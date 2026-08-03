"""Tavily 웹 검색 결과를 연구실 도우미용 근거 형식으로 정규화한다."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from labrag.config import Settings

logger = logging.getLogger("uvicorn.error")


# 예전에는 URL 도메인으로 출처를 `공식/학술`·`전문/보도`·`경험/비공식` 세 등급으로
# 나눠 프롬프트와 출처 목록에 넣었다. 실제 웹 결과 30건으로 재보니 가장 큰 등급
# (`경험/비공식` 16건)이 맞을 때보다 틀릴 때가 많았고, 오류가 한쪽으로 쏠렸다 —
# korea.kr(정책브리핑)·krei.re.kr(국책연구기관)·mdpi.com(학술지)이 전부 "개인 경험"
# 으로 강등됐다. `.go.kr` 만 보고 `.re.kr`·`.or.kr`·학술 출판사를 못 잡기 때문이다.
#
# 시스템 프롬프트가 `경험/비공식` 은 사실로 단정하지 말라고 지시하므로, 이 오분류는
# 정부 발표를 블로그 의견처럼 얼버무리게 만들었다. 도메인 목록을 늘리는 방식은 끝이
# 없어서, 등급을 없애고 URL 을 그대로 보여준 뒤 모델이 판단하게 한다.
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
# Tavily 하드 리밋. 넘기면 응답이 `Query is too long. Max query length is 400
# characters.` 와 함께 HTTP 400 이다 — 실패한 호출도 쿼터를 쓰므로 보내기 전에 자른다.
TAVILY_MAX_QUERY_CHARS = 400


ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
# 실제 웹 페이지 제목은 이 길이를 넘지 않는다. 넘는 것은 프롬프트를 밀어내려는 쪽에
# 가깝다 — 근거 목록 한 줄로 쓸 수 있는 만큼만 남긴다.
MAX_TITLE_CHARS = 200
# 웹 본문을 그대로 노출하는 상한. 프롬프트 근거 블록과 /search 응답이 같은 값을
# 쓴다 — 한쪽만 늘리면 "프롬프트에 넣은 근거"와 "점검용으로 본 근거"가 달라진다.
WEB_CONTENT_PREVIEW_CHARS = 1500

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# Tavily 호출량 감시용 누적 카운터. 키 미설정으로 호출 자체를 건너뛴 경우는 calls 에
# 세지 않는다 — 세면 실제 쿼터 사용량과 어긋난다.
_ZERO_METRICS = {
    "calls": 0, "ok": 0, "empty": 0, "errors": 0, "skipped_too_long": 0,
}
_metrics = dict(_ZERO_METRICS)


def web_search_metrics() -> dict[str, int]:
    """누적 Tavily 호출 집계 사본."""
    return dict(_metrics)


def reset_web_search_metrics() -> None:
    _metrics.update(_ZERO_METRICS)


def _fail(reason: str) -> None:
    """실패를 카운트하고 사유를 남긴다.

    호출자에게는 계속 빈 결과를 돌려주지만(웹 장애가 채팅을 실패시키면 안 된다),
    그렇다고 조용히 삼키면 쿼터 소진이 '검색 결과 없음'과 구분되지 않는다.
    """
    _metrics["errors"] += 1
    logger.warning("[web] Tavily 호출 실패: %s", reason)


def _clean_url(raw: str) -> str:
    """근거로 쓸 수 없는 URL 은 빈 문자열로 만들어 호출자가 버리게 한다.

    출처 목록은 마크다운 링크로 렌더링되므로 `javascript:`·`data:` 스킴이 그대로
    나가면 클릭 가능한 위험 링크가 된다. 공백이 남은 URL 도 마크다운 링크를
    깨뜨리므로 버린다 — Tavily 가 정상 URL 을 주는 한 걸리지 않는 이중 방어다.
    """
    text = _CONTROL_CHARS.sub("", raw).strip()
    if not text or re.search(r"\s", text):
        return ""
    parsed = urlparse(text)
    if parsed.scheme.lower() not in ALLOWED_URL_SCHEMES or not parsed.hostname:
        return ""
    return text


def _clean_title(raw: str) -> str:
    """제목은 반드시 한 줄로 만든다.

    근거 블록은 `[W1] 제목` / `URL:` / `발췌:` 를 줄 단위로 쌓는다. 제목에 줄바꿈이
    남아 있으면 그 안에서 `URL:` 줄이나 `[W2]` 항목을 통째로 만들어낼 수 있다 —
    블로그가 정부 주소를 자기 출처인 것처럼 끼워 넣는 식이다. 제목이 줄을 만들지
    못하게 막는 것이 그 위조를 끊는 지점이다.
    """
    collapsed = _CONTROL_CHARS.sub(" ", raw)
    return re.sub(r"\s+", " ", collapsed).strip()[:MAX_TITLE_CHARS].strip()


@dataclass(frozen=True)
class WebHit:
    title: str
    url: str
    content: str
    score: float


async def search_web(
    query: str,
    client: httpx.AsyncClient,
    settings: Settings,
) -> list[WebHit]:
    """Tavily 검색을 수행하고, 장애나 미설정 때는 조용히 빈 결과를 반환한다."""
    api_key = settings.tavily_api_key.strip()
    if not settings.web_search_enabled or not api_key:
        return []

    if len(query) > TAVILY_MAX_QUERY_CHARS:
        _metrics["skipped_too_long"] += 1
        logger.warning(
            "[web] query %d자로 상한(%d) 초과 — Tavily 호출을 건너뜀",
            len(query), TAVILY_MAX_QUERY_CHARS,
        )
        return []

    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": min(max(settings.web_search_max_results, 1), 5),
    }
    _metrics["calls"] += 1
    try:
        response = await client.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=settings.web_search_timeout,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        # 상태 코드를 남긴다. 429/402 는 쿼터 소진이라 '결과 없음'과 대응이 전혀
        # 다르다 — 본문은 키가 섞여 나올 수 있으니 절대 로그에 넣지 않는다.
        _fail(f"http_{exc.response.status_code}")
        return []
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        _fail(type(exc).__name__)
        return []
    if not isinstance(body, dict):
        _fail("bad_payload")
        return []
    rows = body.get("results", [])
    if not isinstance(rows, list):
        _fail("bad_payload")
        return []

    hits: list[WebHit] = []
    for row in rows[:payload["max_results"]]:
        if not isinstance(row, dict):
            continue
        title = _clean_title(str(row.get("title") or ""))
        url = _clean_url(str(row.get("url") or ""))
        if not title or not url:
            continue
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        hits.append(WebHit(
            title=title,
            url=url,
            content=str(row.get("content") or "").strip(),
            score=score,
        ))
    _metrics["ok" if hits else "empty"] += 1
    return hits
