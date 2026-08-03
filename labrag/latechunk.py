"""Late chunking — 문서를 한 번에 통과시켜 청크별 벡터를 뽑는다.

## 왜 이게 되는가 (실측 2026-07-30)

지금 쓰는 임베딩 서버(`vllm-embed`)는 `tok_pooling_type='ALL'`이 기본값이라
`/pooling` + `task=token_embed` 로 **토큰별 2560차원 벡터**를 이미 준다. 새 컨테이너·
모델 교체·양방향(is_causal=false) 설정이 전부 불필요하다 (양방향은 vLLM 0.15.1/
0.20.0 둘 다 pooling 서빙과 조합하면 KV 캐시 할당 단계에서 assert 로 죽는다).

## 무엇을 얻고 무엇을 못 얻는가

이 모델은 causal 이다 — 실측으로 확인했다(앞 청크의 토큰 벡터가 뒤 청크 유무와
무관하게 cos 0.9999 로 동일). 그래서 얻는 것은 **좌측 맥락**이다: 청크 N 이
1..N-1 을 본다. 표 헤더가 앞 청크에 있고 숫자만 뒤 청크에 있는 경우처럼, 앞에
나온 정의·헤더·선행사를 뒤 청크가 가져오는 상황이 정확히 이 범위다.

실측 (표 헤더는 앞 청크, 숫자만 뒤 청크):

    질의 "토마토 병해 분류 모델의 F1 점수는 얼마인가"
      기존(청크 단독 임베딩)   cos 0.445
      late chunking            cos 0.704   ← 표적 사례
    무관 질의(대조군)
      기존                     cos 0.190
      late chunking            cos 0.221   ← 거의 안 오름 = 신호가 늘었다는 뜻

## 왜 청크 경계에 EOS 를 넣는가

Qwen3-Embedding 은 **마지막 토큰**을 pooling 하도록 학습됐고, 단독 임베딩 때는
끝에 EOS 가 붙어 그 EOS 를 pooling 한다. 문서를 이어붙일 때 청크 경계마다
EOS 를 넣어주면 각 청크의 pooling 위치도 EOS 가 되어 학습 분포와 맞는다.
실측: 경계 EOS 없이는 단독 임베딩과 cos 0.774, 넣으면 0.811 로 올라간다.

입력 텍스트 자체는 오늘 쓰는 `Chunk.embed_text` 를 그대로 이어붙인 것이라,
바뀌는 것은 attention 이 보는 범위뿐이다.
"""
from __future__ import annotations

import math

import httpx

from .config import settings

EOS = "<|endoftext|>"
EOS_TOKEN_ID = 151643        # Qwen3 계열 <|endoftext|>

# 8192(max_model_len) 에서 여유를 둔다 — 윈도우 경계 계산 오차로 요청이
# 거부되면 그 파일 전체가 실패하므로 보수적으로 잡는다.
MAX_WINDOW_TOKENS = 7600
# 윈도우가 나뉠 때 앞 윈도우의 마지막 몇 청크를 좌측 맥락으로 다시 넣는가.
# 이 청크들은 맥락으로만 쓰고 벡터는 재사용하지 않는다(이미 앞 윈도우에서 냈다).
CONTEXT_CARRY_CHUNKS = 2


class LateChunkError(RuntimeError):
    pass


def _post(client: httpx.Client, path: str, payload: dict) -> dict:
    r = client.post(f"{settings.embed_url.rstrip('/')}{path}", json=payload)
    r.raise_for_status()
    return r.json()


def tokenize(client: httpx.Client, text: str) -> list[int]:
    """토큰 ID 목록. add_special_tokens=False 로 우리가 넣은 EOS 만 남게 한다."""
    d = _post(client, "/tokenize", {
        "model": "embed", "prompt": text, "add_special_tokens": False,
    })
    return d["tokens"]


def token_embed(client: httpx.Client, text: str) -> list[list[float]]:
    """토큰별 임베딩. tokenize 와 토큰 수가 정확히 맞아야 하므로 특수토큰 추가 금지."""
    d = _post(client, "/pooling", {
        "model": "embed", "input": text, "task": "token_embed",
        "add_special_tokens": False,
    })
    return d["data"][0]["data"]


def _l2(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def _chunk_token_counts(client: httpx.Client, texts: list[str]) -> list[int]:
    """각 청크(본문+EOS)의 토큰 수.

    문서 전체를 한 번 tokenize 해서 EOS 위치로 나누는 게 호출 수가 적지만,
    /tokenize 도 길이 제한에 걸릴 수 있어 실패하면 청크별로 물어본다.
    """
    doc = "".join(t + EOS for t in texts)
    try:
        ids = tokenize(client, doc)
        bounds = [i for i, t in enumerate(ids) if t == EOS_TOKEN_ID]
        if len(bounds) == len(texts):
            counts, prev = [], -1
            for b in bounds:
                counts.append(b - prev)
                prev = b
            return counts
    except Exception:
        pass
    return [len(tokenize(client, t + EOS)) for t in texts]


def _windows(counts: list[int]) -> list[tuple[int, int]]:
    """청크를 MAX_WINDOW_TOKENS 안에 들어가는 [시작, 끝) 구간으로 묶는다."""
    out: list[tuple[int, int]] = []
    i = 0
    n = len(counts)
    while i < n:
        total = 0
        j = i
        while j < n and (total + counts[j] <= MAX_WINDOW_TOKENS or j == i):
            total += counts[j]
            j += 1
        out.append((i, j))
        i = j
    return out


def late_chunk_vectors(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    """청크 텍스트 목록 → 청크별 late-chunked 벡터 (입력 순서 유지, L2 정규화).

    texts 는 보통 `[c.embed_text for c in chunks]` 다 — 즉 오늘 각 청크를 따로
    임베딩할 때 쓰는 그 텍스트를 그대로 넘긴다.
    """
    if not texts:
        return []
    counts = _chunk_token_counts(client, texts)
    if len(counts) != len(texts):
        raise LateChunkError(f"토큰 수 산출 실패: {len(counts)} vs 청크 {len(texts)}")

    vectors: list[list[float] | None] = [None] * len(texts)

    for start, end in _windows(counts):
        # 좌측 맥락으로 앞 청크 몇 개를 같이 넣는다(벡터는 안 씀).
        ctx_start = max(0, start - CONTEXT_CARRY_CHUNKS)
        while ctx_start < start and sum(counts[ctx_start:end]) > MAX_WINDOW_TOKENS:
            ctx_start += 1

        window_texts = texts[ctx_start:end]
        doc = "".join(t + EOS for t in window_texts)
        toks = token_embed(client, doc)
        ids = tokenize(client, doc)
        if len(toks) != len(ids):
            raise LateChunkError(
                f"토큰 임베딩 {len(toks)}개 vs 토큰 {len(ids)}개 불일치 — "
                f"add_special_tokens 처리가 서버와 어긋났다"
            )
        eos_pos = [i for i, t in enumerate(ids) if t == EOS_TOKEN_ID]
        if len(eos_pos) != len(window_texts):
            raise LateChunkError(
                f"EOS 경계 {len(eos_pos)}개 vs 청크 {len(window_texts)}개 불일치"
            )
        for k, pos in enumerate(eos_pos):
            idx = ctx_start + k
            if idx >= start:                     # 맥락용 청크는 건너뛴다
                vectors[idx] = _l2(toks[pos])

    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        raise LateChunkError(f"벡터를 못 만든 청크: {missing[:5]} (총 {len(missing)}개)")
    return vectors  # type: ignore[return-value]
