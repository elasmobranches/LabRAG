"""임베딩·리랭커 클라이언트.

## ⚠️ 리랭커는 vLLM의 /v1/rerank 를 쓰면 안 된다

Qwen3-Reranker 는 "yes/no 로만 답하라"는 지시문이 포함된 특정 프롬프트 형식으로
학습된 모델이다. vLLM 의 `/v1/rerank` 엔드포인트는 질의와 문서를 단순히 이어붙이기만
해서 모델이 학습된 분포를 벗어난다. 영어는 어휘 신호가 강해 버티지만 한국어는 무너진다.

실측 (질의: "토마토 병해 분류 방법"):

    문서                          /v1/rerank      /classify + 템플릿
    스테레오 카메라 비용 절감       0.277 (1위)     0.0000434
    토마토 잎 병해 CNN 분류          0.049 (3위)     0.99939 (1위)   ← 정답
    4족 보행 로봇 계단 등반         0.103           0.0000187

리랭커가 LLM 에 넘길 문서를 결정하므로, 이 차이가 RAG 전체의 정확도를 좌우한다.
그래서 여기서 템플릿을 직접 만들어 `/classify` 로 보낸다.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from .config import settings

# Qwen3-Reranker 공식 프롬프트 형식. 한 글자도 바꾸지 말 것 — 학습 분포와 맞아야 한다.
_RERANK_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
DEFAULT_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def rerank_prompt(query: str, doc: str, instruct: str = DEFAULT_INSTRUCT) -> str:
    return (
        f"<|im_start|>system\n{_RERANK_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n<Instruct>: {instruct}\n<Query>: {query}\n"
        f"<Document>: {doc}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@dataclass
class Scored:
    index: int          # 원본 리스트에서의 위치
    score: float


class Models:
    """임베딩·리랭커 HTTP 클라이언트.

    인덱싱(수천 건 배치)과 질의(수십 건)를 같은 코드로 다룬다.
    """

    def __init__(
        self,
        embed_url: str | None = None,
        rerank_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.embed_url = (embed_url or settings.embed_url).rstrip("/")
        self.rerank_url = (rerank_url or settings.rerank_url).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Models":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ 임베딩

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """텍스트를 벡터로. 입력 순서를 그대로 유지한다."""
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self._client.post(
                f"{self.embed_url}/v1/embeddings",
                json={"model": "embed", "input": batch},
            )
            r.raise_for_status()
            data = r.json()["data"]
            # API 가 순서를 보장하지 않을 수 있으니 index 로 정렬한다
            data.sort(key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        """임베딩 차원. Qdrant 컬렉션 생성에 필요하다.

        하드코딩하지 않는 이유: 임베딩 모델을 바꾸면 차원이 달라지고,
        컬렉션 차원이 어긋나면 전체 재인덱싱이 필요해진다. 매번 실측한다.
        """
        if not hasattr(self, "_dim"):
            self._dim = len(self.embed_one("dimension probe"))
        return self._dim

    # ------------------------------------------------------------ 리랭킹

    def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int | None = None,
        instruct: str = DEFAULT_INSTRUCT,
        batch_size: int = 16,
    ) -> list[Scored]:
        """질의와의 관련도로 문서를 재정렬한다 (점수 내림차순).

        /classify 에 공식 템플릿을 적용해 보낸다 — 모듈 최상단 설명 참고.
        """
        scores: list[Scored] = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            r = self._client.post(
                f"{self.rerank_url}/classify",
                json={
                    "model": "rerank",
                    "input": [rerank_prompt(query, d, instruct) for d in batch],
                },
            )
            r.raise_for_status()
            for item in r.json()["data"]:
                probs = item["probs"]
                # num_classes=1 이면 probs 는 "yes" 확률 하나.
                # 2 클래스로 나오는 버전이라면 마지막 값(yes)을 쓴다.
                score = float(probs[-1]) if isinstance(probs, list) else float(probs)
                scores.append(Scored(index=i + item["index"], score=score))
        scores.sort(key=lambda s: -s.score)
        return scores[:top_k] if top_k else scores

    # ------------------------------------------------------------ 헬스체크

    def health(self) -> dict[str, str]:
        out = {}
        for name, url in (("embed", self.embed_url), ("rerank", self.rerank_url)):
            try:
                r = self._client.get(f"{url}/health", timeout=5)
                out[name] = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
            except Exception as e:
                out[name] = f"{type(e).__name__}"
        return out
