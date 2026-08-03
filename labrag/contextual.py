"""파일 단위 contextual enrichment — 청크가 문서 전체 맥락 없이 고립되는 문제를 줄인다.

## 왜 청크 단위가 아니라 파일 단위인가

Anthropic 의 contextual retrieval 원안은 청크마다 LLM으로 "문서 전체에서 이 청크의
위치"를 생성한다. 이 코퍼스는 인덱싱된 파일 7,167개, 청크 241,221개 — 청크 단위로
하면 로컬 vLLM 생성 모델 하나로 감당하기 어렵다. 파일 단위(파일당 1회 호출)로
낮추면 비용이 33배 줄고, "이 문서가 뭔지" 정도의 정보는 여전히 준다.

## 왜 요약을 짧고 중립적으로 강제하는가

파일 요약을 그 파일의 모든 청크에 그대로 붙이면, 같은 파일의 청크들이 서로 더
비슷해져 버린다. 실측으로 이미 고친 문제(§27 "최신 랩미팅"이 날짜 하나 차이를
구분 못해 엉뚱한 문서로 쏠림)와 같은 종류의 편향을 임베딩 쪽에서 재도입할 위험이
있다 — 그래서 날짜·이름을 나열하지 말고 "문서 종류와 역할"만 짧게 쓰게 한다.
"""
from __future__ import annotations

import httpx

from .config import settings
from .drive import DriveFile
from .parse import Block

MIN_SYNOPSIS_TEXT = 200      # 이보다 앞부분 텍스트가 적으면 판단할 근거가 부족
SAMPLE_CHARS = 1500          # 프롬프트에 넣을 문서 앞부분 상한
MAX_SYNOPSIS_TOKENS = 100

SYNOPSIS_PROMPT = """다음은 어느 연구실의 구글 드라이브 문서 앞부분 발췌야.

파일명: {name}
경로: {root}
종류: {category}

<발췌>
{text}
</발췌>

이 문서가 무엇이고 어떤 용도인지 1~2문장으로 짧게 설명해줘. 규칙:

1. 발췌에 실제로 나온 내용만 근거로 써 — 없는 사실을 지어내지 마.
2. 날짜·사람 이름·수치를 나열하지 마. "이 문서의 종류와 역할"만 설명해
   (예: "○○ 프로젝트의 진행 상황을 정리한 회의록", "온실 에너지 시뮬레이션 설정
   방법을 설명하는 매뉴얼의 일부"). 구체적인 날짜나 이름을 넣으면 이 문서의 다른
   부분과 헷갈리게 된다.
3. 발췌로 판단이 안 서면 정확히 SKIP 이라고만 출력해.
4. 설명만 출력해. 따옴표나 "이 문서는" 같은 군더더기 서두 없이 바로 써."""


def _sample_text(blocks: list[Block]) -> str:
    parts: list[str] = []
    total = 0
    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        parts.append(t)
        total += len(t)
        if total >= SAMPLE_CHARS:
            break
    return "\n".join(parts)[:SAMPLE_CHARS]


def generate_file_synopsis(client: httpx.Client, f: DriveFile, blocks: list[Block]) -> str | None:
    """파일 전체를 대표하는 짧은 중립적 요약. 실패하거나 판단이 안 서면 None.

    호출 실패가 인덱싱을 막으면 안 되므로 예외를 전부 삼킨다 — 이 파일은 그냥
    기존처럼(맥락 없이) 인덱싱된다.
    """
    text = _sample_text(blocks)
    if len(text) < MIN_SYNOPSIS_TEXT:
        return None
    try:
        r = client.post(
            f"{settings.vllm_url}/chat/completions",
            json={
                "model": settings.vllm_model,
                "messages": [{
                    "role": "user",
                    "content": SYNOPSIS_PROMPT.format(
                        name=f.name, root=f.root, category=f.category, text=text,
                    ),
                }],
                "temperature": 0.2,
                "max_tokens": MAX_SYNOPSIS_TOKENS,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        if r.status_code != 200:
            return None
        out = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    out = out.strip().strip('"').strip()
    if not out or out.upper().startswith("SKIP") or len(out) < 10:
        return None
    return out
