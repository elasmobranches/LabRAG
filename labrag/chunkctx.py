"""청크별 맥락 생성 (contextual retrieval) — 문서 크기에 따라 범위를 바꾼다.

## 왜 청크 단위인가

파일 단위 요약을 그 파일의 모든 청크에 똑같이 붙이는 방식은 실측에서 실패했다
(2026-07-30: 넓은 탐색형 질의의 top-10 고유 파일 수가 6개 → 1개로 붕괴). 같은
문자열이 같은 파일 청크들을 서로 끌어당겼기 때문이다. 청크마다 **다른** 맥락을
붙이면 그 실패 모드를 피한다 — 실측 결과 다양성 4.5/10 로 기존 5.2/10 을 거의
보존하면서 표적 사례 MRR 은 0.265 → 0.387 로 올랐다.

## 왜 문서 전문을 항상 주지 않는가

40파일 파일럿에서 EnergyPlus 매뉴얼 같은 초대형 다주제 문서는 맥락을 붙였을 때
오히려 **악화**했다(정답 청크 3위 → 13위). 문서 전문을 프롬프트에 넣으면
서로 무관한 주제가 섞여 들어가 대상 청크가 희석되고, 문서의 대표 주제가 대상
청크의 주제를 덮어쓴다. 그래서 문서가 크면 범위를 좁힌다:

    문서 전체가 예산 안에 들어옴   → full    (짧고 단일 주제)
    같은 섹션만                    → section (구조화된 중대형)
    앞뒤 몇 청크만                 → local   (초대형 다주제)

## 숫자만 있는 파일

YOLO 라벨 좌표처럼 처음부터 끝까지 숫자인 파일은 문서를 얼마나 보여줘도 의미가
생기지 않는다. 이런 파일의 의미는 파일명·폴더 경로·표 헤더에서 온다. 그래서
범위와 무관하게 **항상** 파일명·폴더·섹션을 넣고, 파일 앞부분(표 헤더가 있는
곳)을 같이 넣는다.
"""
from __future__ import annotations

import re

import httpx

from .config import settings

# 프롬프트 예산(문자). 한국어는 1자≈1토큰에 가까워서 문자로 잡아도 보수적이다.
# 문서 전문은 예산의 60% 이내일 때만 허용한다(나머지는 지시문·대상 청크·출력용).
PROMPT_BUDGET_CHARS = 12000
FULL_DOC_MAX_CHARS = 7200
SECTION_MAX_CHARS = 5000
LOCAL_NEIGHBORS = 2          # local 범위에서 앞뒤로 볼 청크 수
HEAD_CHARS = 700             # 파일 앞부분(표 헤더 등) 발췌 길이
TARGET_MAX_CHARS = 2000      # 대상 청크를 프롬프트에 넣을 때 상한
MAX_CONTEXT_CHARS = 400      # 생성된 맥락 문장 길이 상한
MIN_CONTEXT_CHARS = 8

SCOPES = ("full", "section", "local")

_PROMPT = """다음은 연구실 문서의 일부다.

파일명: {name}
폴더: {root}
문서 내 경로: {path}
{section_line}
{head_block}<문서{scope_note}>
{body}
</문서>

아래 <대상> 부분이 이 문서에서 어떤 맥락에 있는지 한국어 1~2문장으로 짧게 써라.

<대상>
{target}
</대상>

규칙:
1. 이 대상을 검색으로 찾을 때 도움될 정보만 써라 — 무엇에 관한 데이터·내용인지, 어떤 표·절·주제에 속하는지.
2. 대상에 지시대명사("이 방법", "위 표")나 약어가 있으면 문서를 보고 실제 대상으로 풀어서 명시해라.
3. 표의 숫자만 있는 대상이라면 각 열이 무엇인지 문서에서 찾아 밝혀라.
4. 문서·파일명·폴더에서 확인되지 않는 사실은 절대 만들지 마라. 연도·사람·과제명을 추측해 넣지 마라.
5. 설명만 출력해라. "이 대상은" 같은 군더더기 서두 없이 바로."""


def _clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …"


def choose_scope(chunk_texts: list[str], index: int) -> str:
    """문서 크기를 보고 맥락 범위를 정한다."""
    total = sum(len(t) for t in chunk_texts)
    if total <= FULL_DOC_MAX_CHARS:
        return "full"
    return "section" if index is not None else "local"


def build_prompt(
    *, file_name: str, root: str, path: str,
    chunk_texts: list[str], sections: list[str | None], index: int,
) -> tuple[str, str]:
    """(프롬프트, 사용한 범위)."""
    target = _clip(chunk_texts[index], TARGET_MAX_CHARS)
    total = sum(len(t) for t in chunk_texts)
    section = sections[index] if index < len(sections) else None

    if total <= FULL_DOC_MAX_CHARS:
        scope = "full"
        body_parts = chunk_texts
        note = " 전체"
    else:
        # 같은 섹션만 모아본다. 섹션이 없거나 그것도 너무 크면 앞뒤 이웃만.
        same = [j for j, s in enumerate(sections)
                if section is not None and s == section]
        if same and sum(len(chunk_texts[j]) for j in same) <= SECTION_MAX_CHARS:
            scope = "section"
            body_parts = [chunk_texts[j] for j in same]
            note = f" 중 '{section}' 절"
        else:
            scope = "local"
            lo = max(0, index - LOCAL_NEIGHBORS)
            hi = min(len(chunk_texts), index + LOCAL_NEIGHBORS + 1)
            body_parts = chunk_texts[lo:hi]
            note = f" 중 {lo + 1}~{hi}번째 조각"

    body = _clip("\n".join(body_parts), FULL_DOC_MAX_CHARS)

    # 범위를 좁혔을 때는 파일 앞부분(표 헤더·제목이 있는 곳)을 따로 붙인다.
    # 숫자만 있는 표 파일은 이 헤더가 유일한 의미 단서다.
    head_block = ""
    if scope != "full" and chunk_texts:
        head = _clip(chunk_texts[0], HEAD_CHARS)
        if head and head not in body:
            head_block = f"<문서 앞부분>\n{head}\n</문서 앞부분>\n\n"

    prompt = _PROMPT.format(
        name=file_name, root=root, path=path,
        section_line=f"현재 절: {section}" if section else "",
        head_block=head_block, scope_note=note, body=body, target=target,
    )
    return prompt, scope


_BAD_PREFIX = re.compile(r"^(이 대상은|이 청크는|이 발췌는|해당 대상은)\s*")


def validate(text: str | None) -> str | None:
    """생성 결과 검증 — 비었거나 너무 길거나 형식이 깨진 건 버린다."""
    if not text:
        return None
    t = text.strip().strip('"').strip()
    t = _BAD_PREFIX.sub("", t).strip()
    if len(t) < MIN_CONTEXT_CHARS:
        return None
    if len(t) > MAX_CONTEXT_CHARS:
        t = t[:MAX_CONTEXT_CHARS].rsplit(" ", 1)[0] + " …"
    if t.upper().startswith("SKIP"):
        return None
    return t


def generate(
    client: httpx.Client, *, file_name: str, root: str, path: str,
    chunk_texts: list[str], sections: list[str | None], index: int,
    max_tokens: int = 140,
) -> tuple[str | None, str]:
    """(맥락 문장 또는 None, 사용한 범위). 실패는 예외 대신 None 이다 —
    한 청크의 생성 실패가 파일 전체나 빌드를 멈추게 하면 안 된다."""
    prompt, scope = build_prompt(
        file_name=file_name, root=root, path=path,
        chunk_texts=chunk_texts, sections=sections, index=index,
    )
    try:
        r = client.post(f"{settings.vllm_url}/chat/completions", json={
            "model": settings.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        if r.status_code != 200:
            return None, scope
        return validate(r.json()["choices"][0]["message"]["content"]), scope
    except Exception:
        return None, scope
