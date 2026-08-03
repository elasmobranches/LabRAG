"""파일 단위 문서 synopsis 벡터 — 청크 검색과 병렬로 도는 "문서 발견" 계층.

## 왜 청크 임베딩에 안 넣고 별도 컬렉션인가

파일럿(2026-07-30)에서 파일 요약을 모든 청크 임베딩에 그대로 붙이는 방식을 실측했다.
승리 사례도 있었다 — "토마토 재배 관련 객체 탐지 데이터가 어디 있어?"(원문이 YOLO 라벨
좌표 숫자뿐이라 "토마토"라는 단어가 전혀 없는 파일)를 top-10 밖에서 1위로 끌어올렸다.
그런데 부작용도 실측됐다 — "토마토 관련 연구 자료 알려줘" 같은 넓은 탐색형 질의에서
그 파일의 청크들이 서로 너무 비슷해져 top-10을 통째로 점유했다(고유 파일 수 6/10 →
1/10). 청크 임베딩은 그대로 두고, 파일 단위 synopsis 를 별도 벡터(파일당 1개)로
인덱싱해 청크 검색과 병렬로 돌린 뒤 후보만 합치면, 청크 검색의 다양성을 해치지
않으면서 "파일명·본문 모두 불투명한 파일"도 찾을 수 있다 — 이미 만들어둔 하이브리드
lexical 문서 검색(catalog.py)과 같은 패턴이다. 다른 점은 lexical 은 파일명/경로
문자열 매칭이고, 이건 LLM 요약의 의미 임베딩 매칭이라는 것.

## 왜 Store 를 그대로 재사용하는가

`Store`의 create/exists/count/search 는 Chunk 에 종속적이지 않다 — payload 딕셔너리를
그대로 감싸는 `Hit`만 다룬다. upsert 만 `Chunk` 타입에 매여 있어서, 문서 포인트는
여기서 직접 qdrant_client 로 upsert 한다.
"""
from __future__ import annotations

import uuid

from qdrant_client import models as qm

from .config import settings
from .store import DENSE, Store

DOC_COLLECTION = f"{settings.collection}_docsyn"

# synopsis 프롬프트나 생성 모델을 바꾸면 이 값을 올린다. mod_time 만 보고 증분
# 스킵하면, 프롬프트를 고쳐도 파일이 안 바뀐 이상 옛 synopsis가 그대로 남아
# 새/구 버전이 조용히 섞인다.
PIPELINE_VERSION = "v1"

# chunk.py 와 다른 네임스페이스를 써서 포인트 ID가 절대 겹치지 않게 한다
_POINT_NAMESPACE = uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")


def doc_point_id(file_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"docsyn:{file_id}"))


def existing_fingerprints(store: Store) -> dict[str, tuple[str, str, str, str]]:
    """이미 synopsis 가 있는 file_id -> (mod_time, path, root, pipeline_version).

    build_docsynopsis.py 의 증분 판단에 쓴다. mod_time 만 보면 놓치는 두 가지를
    같이 본다 — path/root 가 바뀌면 synopsis 의 근거(예: 폴더명에 있던 프로젝트명)
    가 달라질 수 있고, pipeline_version 이 바뀌면 파일이 그대로여도 새 프롬프트로
    다시 만들어야 한다.
    """
    out: dict[str, tuple[str, str, str, str]] = {}
    offset = None
    while True:
        pts, offset = store.client.scroll(
            collection_name=store.collection, limit=1000, offset=offset,
            with_payload=["file_id", "mod_time", "path", "root", "pipeline_version"],
        )
        for p in pts:
            pl = p.payload or {}
            fid = pl.get("file_id")
            if fid:
                out[fid] = (pl.get("mod_time", ""), pl.get("path", ""),
                           pl.get("root", ""), pl.get("pipeline_version", ""))
        if offset is None:
            break
    return out


def upsert_doc_synopsis(store: Store, *, file_id: str, file_name: str, root: str,
                        path: str, mod_time: str, category: str,
                        synopsis: str, vector: list[float]) -> None:
    store.client.upsert(
        collection_name=store.collection,
        points=[qm.PointStruct(
            id=doc_point_id(file_id),
            vector={DENSE: vector},
            payload={
                "file_id": file_id, "file_name": file_name, "root": root,
                "path": path, "mod_time": mod_time, "category": category,
                "synopsis": synopsis, "pipeline_version": PIPELINE_VERSION,
            },
        )],
        wait=True,
    )


def delete_doc_synopsis(store: Store, file_ids: list[str]) -> None:
    if not file_ids:
        return
    store.client.delete(
        collection_name=store.collection,
        points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="file_id", match=qm.MatchAny(any=file_ids))
        ])),
        wait=True,
    )
