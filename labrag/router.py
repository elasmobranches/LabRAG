from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .intent import parse_query_intent

RouteMode = Literal["auto", "general", "rag", "location", "web", "rag_web"]
VALID_MODES = {"auto", "general", "rag", "location", "web", "rag_web"}

LOCATION_HINT = re.compile(
    r"(어디|어느\s*폴더|위치|경로|링크|찾아\s*줘|찾아\s*주|"
    r"\.(?:pdf|docx?|xlsx?|pptx?|hwp|hwpx|csv|txt)\b)",
    re.IGNORECASE,
)
INTERNAL_HINT = re.compile(
    r"(우리\s*(?:연구실|랩)|연구실|랩미팅|과제|실험|레이블링|라벨링|"
    r"프로젝트|연구비|회의록|슬랙|slack|채널|drive|드라이브|내부\s*문서|"
    r"논문|데이터셋|Conference|"
    r"출장|방문|참여(?:자|인원)|담당(?:자|업무)|업무\s*기록|"
    r"다녀\s*온|갔다\s*온|현장\s*테스트)",
    re.IGNORECASE,
)
GENERAL_HINT = re.compile(
    r"(안녕|반가워|고마워|감사|날씨|설명해|개념|번역해|영어로|한국어로|"
    r"작성해|써\s*줘|아이디어|코드|코딩|파이썬|python|일반적으로|장단점|"
    r"여행지|여행\s*추천|추천\s*해|뭐\s*(?:먹|하지)|루틴)",
    re.IGNORECASE,
)
DRIVE_SOURCE_HINT = re.compile(
    r"(?:구글\s*)?(?:drive|드라이브)", re.IGNORECASE
)
SLACK_SOURCE_HINT = re.compile(r"(?:slack|슬랙)", re.IGNORECASE)
LAB_CONTEXT_HINT = re.compile(
    r"(우리|연구실|랩|Conference|구글\s*드라이브|drive|드라이브|slack|슬랙|내부)",
    re.IGNORECASE,
)
CURRENT_WEB_HINT = re.compile(
    r"(?:최신|최근|현재|이번\s*주|요즘).{0,20}"
    r"(?:뉴스|동향|논문|제품|정책|추천)",
    re.IGNORECASE,
)
WEB_PRODUCT_POLICY_HINT = re.compile(
    r"(?:제품|정책)(?:.{0,20}(?:추천|비교|최신|알려|장단점|"
    r"어떻게|바뀌)|$)",
    re.IGNORECASE,
)
WEB_HINT = re.compile(
    r"(웹|인터넷|뉴스|동향|요즘)", re.IGNORECASE
)
WEB_CAPABLE_HINT = re.compile(
    r"(최신|최근|현재|이번\s*주|요즘)", re.IGNORECASE
)
EXPLICIT_WEB_HINT = re.compile(
    r"(?:웹|인터넷)\s*(?:검색|에서도|에서|자료|최신|동향|"
    r"(?:도|을|를)\s*(?:함께|같이|검색|찾아))",
    re.IGNORECASE,
)
# 출처를 웹이라고 콕 집어 말한 형태. `웹 크롤러`처럼 웹이 주제일 뿐인 경우와
# 구별해야 해서 EXPLICIT_WEB_HINT 보다 좁게 잡는다 — 뒤에 오는 말까지 본다.
WEB_SOURCE_HINT = re.compile(
    r"(?:웹|인터넷)\s*(?:검색|에서도|에서|"
    r"(?:도|을|를)\s*(?:함께|같이|검색|찾아))", re.IGNORECASE
)


OPENWEBUI_TASK_PREFIX = re.compile(r"^\s*###\s*task\s*:", re.IGNORECASE)
OPENWEBUI_GENERATION_PROMPT = re.compile(
    r"^\s*(?:generate|create)\s+(?:a\s+)?(?:concise\s+)?"
    r"(?:title|tags?|follow[- ]?up\s+questions?|search\s+queries?)\b",
    re.IGNORECASE,
)

_HYBRID_WEB_FOR_INTERNAL = re.compile(
    r"(?:웹|인터넷)\s*(?:검색(?:으로)?|에서도|에서)|"
    r"(?:웹|인터넷)\s*(?:도|을|를)(?=\s*(?:함께|같이|검색|찾아))|"
    r"(?:웹|인터넷)(?=\s*(?:자료|최신|동향))",
    re.IGNORECASE,
)
_HYBRID_PROJECT_JOIN = re.compile(
    r"(?:과제|프로젝트)\s*(?:와|과|및|그리고)\s*(?=(?:웹|인터넷))",
    re.IGNORECASE,
)
_HYBRID_INTERNAL_FOR_WEB = re.compile(
    r"(?:우리\s*)?(?:연구실|랩)\s*|"
    r"(?:구글\s*)?(?:드라이브|drive)\s*(?:와|과|및|그리고)?\s*|"
    r"(?:슬랙|slack)\s*(?:와|과|및|그리고)?\s*|"
    r"내부\s*(?:자료|문서)?\s*",
    re.IGNORECASE,
)
_HYBRID_SOURCE_CLAUSES = re.compile(
    r"^(?P<internal>.+)\s*(?:와|과|및|그리고)\s*"
    r"(?P<web>(?:웹|인터넷).*)$",
    re.IGNORECASE,
)
_HYBRID_WEB_FIRST_CLAUSES = re.compile(
    r"^(?P<web>(?:웹|인터넷).+?)\s*(?:와|과|및|그리고)\s*"
    r"(?P<internal>(?:(?:우리\s*)?(?:연구실|랩)|"
    r"(?:구글\s*)?(?:드라이브|drive)|(?:슬랙|slack)|내부).*)$",
    re.IGNORECASE,
)


def _clean_search_query(query: str) -> str:
    query = re.sub(r"\s+", " ", query).strip()
    query = re.sub(r"^(?:와|과|및|그리고)\s*", "", query)
    return query.strip(" ,")


def _enrich_internal_summary_query(query: str) -> str:
    """짧은 과제명만 남은 요약 요청은 숫자 라벨보다 서술형 문서를 우선하게 한다."""
    topic = _clean_search_query(_HYBRID_INTERNAL_FOR_WEB.sub("", query))
    if re.search(r"(?:과제|프로젝트)$", topic):
        return f"{topic} 회의록 연구 내용"
    return query


def split_hybrid_queries(query: str) -> tuple[str, str]:
    """복합 요청을 내부 저장소용과 웹 검색용 문장으로 가볍게 정리한다.

    LLM 재작성 호출을 추가하지 않고 출처 지시어만 걷어낸다. 뜻을 새로 만들지 않아
    실패해도 원 질문에 가까우며, 두 검색기가 서로의 출처 단어에 끌리는 현상만 막는다.
    """
    web_first = _HYBRID_WEB_FIRST_CLAUSES.match(query.strip())
    clauses = _HYBRID_SOURCE_CLAUSES.match(query.strip())
    web_override = None
    if web_first:
        internal = _clean_search_query(web_first.group("internal"))
        web_override = _clean_search_query(web_first.group("web"))
    elif clauses:
        internal_clause = _clean_search_query(clauses.group("internal"))
        internal_topic = _clean_search_query(
            _HYBRID_INTERNAL_FOR_WEB.sub("", internal_clause)
        )
        if internal_topic:
            internal = internal_clause
        else:
            shared = _clean_search_query(
                _HYBRID_WEB_FOR_INTERNAL.sub("", clauses.group("web"))
            )
            internal = _clean_search_query(f"{internal_clause} {shared}")
    else:
        internal = _clean_search_query(_HYBRID_WEB_FOR_INTERNAL.sub("", query))
    web = web_override
    if web is None:
        web = _HYBRID_PROJECT_JOIN.sub("", query)
        web = _clean_search_query(_HYBRID_INTERNAL_FOR_WEB.sub("", web))
    internal = _enrich_internal_summary_query(internal)
    return internal or query, web or query


def _is_openwebui_task_prompt(query: str) -> bool:
    """Open WebUI 가 대화 턴마다 자동으로 보내는 제목·태그 생성 요청인가.

    제목·태그·후속질문·검색어 생성 프롬프트가 모두 이 머리말로 시작한다
    (Open WebUI `config.py` 의 `DEFAULT_*_GENERATION_PROMPT_TEMPLATE`).

    Open WebUI 는 원래 `metadata: {"task": "title_generation"}` 를 붙이지만 외부 API
    로 보내기 직전에 `payload.pop("metadata")` 로 떼어낸다. 따라서 서버에는 표시
    없는 긴 글만 도착하고, 그 안의 대화 내용 때문에 실제 질문으로 오인될 수 있다.

    사용자 질문이 이 머리말로 시작할 일은 없으므로 이것으로 구분한다. 관리자가
    Open WebUI 에서 프롬프트 템플릿을 바꾸면 이 단서가 사라지니, 그때는 연결 설정에
    커스텀 헤더 `{{TASK}}` 를 추가하는 방법으로 바꿔야 한다.
    """
    return bool(
        OPENWEBUI_TASK_PREFIX.search(query)
        or OPENWEBUI_GENERATION_PROMPT.search(query)
    )


@dataclass(frozen=True)
class RouteDecision:
    mode: RouteMode
    reason: str
    probe_required: bool = False
    probe_score: float | None = None


def classify_query(query: str, requested_mode: str = "auto") -> RouteDecision:
    if requested_mode not in VALID_MODES:
        raise ValueError(f"지원하지 않는 mode: {requested_mode}")
    if requested_mode != "auto":
        return RouteDecision(requested_mode, "forced")  # type: ignore[arg-type]
    # 사용자가 보낸 질문이 아니라 Open WebUI 의 부가 작업 요청이면 검색하지 않는다.
    # 다른 규칙보다 먼저 봐야 한다 — 본문에 대화 전체가 들어 있어 아래 힌트들에
    # 전부 걸린다.
    if _is_openwebui_task_prompt(query):
        return RouteDecision("general", "openwebui_task")
    try:
        intent = parse_query_intent(query)
    except Exception:
        intent = None
    explicit_web = bool(EXPLICIT_WEB_HINT.search(query))
    if intent is not None and set(intent.sources) == {"drive", "slack"}:
        if explicit_web:
            return RouteDecision("rag_web", "explicit_internal_web")
        return RouteDecision("rag", "explicit_multi_source")
    if intent is not None and intent.task == "location":
        return RouteDecision("location", "location_hint")
    if DRIVE_SOURCE_HINT.search(query) and SLACK_SOURCE_HINT.search(query):
        if explicit_web:
            return RouteDecision("rag_web", "explicit_internal_web")
        return RouteDecision("rag", "explicit_multi_source")
    # 내부 출처와 웹을 함께 지정한 요청은 "찾아 줘" 같은 약한 위치 힌트보다 우선한다.
    # 진짜 위치 질문(어디·위치·경로·확장자)은 위 intent.task == "location" 에서 이미
    # 걸러졌으므로 여기까지 오지 않는다 — intent 의 LOCATION_PATTERN 에는 "찾아 줘" 가
    # 없어서, 이 지점에 남는 것은 내용 질문에 붙은 약한 힌트뿐이다.
    if INTERNAL_HINT.search(query) and explicit_web:
        return RouteDecision("rag_web", "explicit_internal_web")
    # 내부 출처를 함께 말하지 않고 웹만 지목한 요청. 위와 같은 이유로 `찾아 줘`
    # 보다 먼저 판정한다 — 예전에는 location 으로 갔다가 파일을 못 찾고 rag_web 으로
    # 넘어와서, 사용자가 웹이라고 말했는데도 위치 검색과 내부 검색을 둘 다 헛돌았다.
    if WEB_SOURCE_HINT.search(query):
        return RouteDecision("web", "web_source_hint")
    if LOCATION_HINT.search(query):
        return RouteDecision("location", "location_hint")
    if (
        CURRENT_WEB_HINT.search(query)
        and not LAB_CONTEXT_HINT.search(query)
    ):
        return RouteDecision("web", "web_hint")
    if INTERNAL_HINT.search(query):
        return RouteDecision("rag", "internal_hint")
    if WEB_HINT.search(query) or WEB_PRODUCT_POLICY_HINT.search(query):
        return RouteDecision("web", "web_hint")
    if WEB_CAPABLE_HINT.search(query):
        return RouteDecision("auto", "ambiguous_web", probe_required=True)
    if GENERAL_HINT.search(query):
        return RouteDecision("general", "general_hint")
    return RouteDecision("auto", "ambiguous", probe_required=True)


def probe_internal_relevance(
    query,
    models,
    drive_store,
    *,
    slack_store=None,
    roots=None,
    candidates=3,
    threshold=0.45,
):
    del threshold  # 임계값 판정은 resolve_probe에서 한 번만 한다.
    try:
        vector = models.embed_one(query)
    except Exception:
        return 0.0, True

    scores: list[float] = []
    successes = 0
    for store, source_roots in ((drive_store, roots), (slack_store, None)):
        if store is None:
            continue
        try:
            hits = store.search(vector, limit=candidates, roots=source_roots)
            successes += 1
            scores.extend(float(hit.score) for hit in hits)
        except Exception:
            continue
    if not successes:
        return 0.0, True
    return max(scores, default=0.0), False


def resolve_probe(
    decision: RouteDecision,
    score: float,
    threshold: float,
    error: bool = False,
) -> RouteDecision:
    if not decision.probe_required:
        return decision
    if error:
        if decision.reason == "ambiguous_web":
            return RouteDecision("web", "probe_web", probe_score=score)
        return RouteDecision("general", "probe_error", probe_score=score)
    if score >= threshold:
        return RouteDecision("rag", "probe_relevant", probe_score=score)
    if decision.reason == "ambiguous_web":
        return RouteDecision("web", "probe_web", probe_score=score)
    return RouteDecision("general", "probe_low", probe_score=score)
