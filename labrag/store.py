"""Qdrant 벡터 저장소.

## 컬렉션 차원을 하드코딩하지 않는 이유

Qwen3-Embedding-4B 는 2560차원이지만 이 값을 코드에 박지 않는다. 임베딩 모델을
바꾸면 차원이 달라지고, 컬렉션 차원이 어긋나면 upsert 가 통째로 실패한다.
컬렉션을 만들 때 실제 엔드포인트에서 한 번 측정해서 쓰고, 그 값을 컬렉션에
기록해둔다. 나중에 모델을 바꾸면 불일치를 즉시 감지할 수 있다.

## 나중에 하이브리드 검색으로 확장하는 경로

지금은 dense(의미) 검색만 쓴다. 논문 제목·모델명·수치 같은 정확한 문자열은
dense 가 놓칠 수 있어서 sparse(BM25 계열)를 더하는 게 좋지만, 아무것도 동작하지
않는 상태에서 하이브리드부터 만들면 문제 원인을 못 찾는다.

Qdrant 는 기존 컬렉션에 sparse 벡터를 나중에 추가할 수 있으므로(named vector),
지금 dense 로만 인덱싱해도 재임베딩 없이 확장할 수 있다. 막다른 길이 아니다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .chunk import Chunk
from .config import settings

DENSE = "dense"          # named vector — sparse 를 나중에 추가할 여지를 둔다
CHANNEL_NAMES_CACHE_TTL_SECONDS = 300.0


@dataclass
class Hit:
    score: float
    payload: dict

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def citation(self) -> str:
        return self.payload.get("citation", "")

    @property
    def url(self) -> str:
        return self.payload.get("file_url", "")


class Store:
    def __init__(self, url: str | None = None, collection: str | None = None) -> None:
        self.client = QdrantClient(url=url or settings.qdrant_url, timeout=120)
        self.collection = collection or settings.collection

    # ------------------------------------------------------------ 컬렉션

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def create(self, dim: int, recreate: bool = False) -> None:
        if self.exists():
            if not recreate:
                self._check_dim(dim)
                return
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            },
        )
        # 삭제·필터에 쓰는 필드에만 인덱스를 만든다 (인덱스는 공짜가 아니다)
        for field, schema in (
            ("file_id", models.PayloadSchemaType.KEYWORD),   # 파일 단위 삭제/갱신
            ("root", models.PayloadSchemaType.KEYWORD),      # 폴더별 검색 제한
            ("page", models.PayloadSchemaType.INTEGER),
        ):
            self.client.create_payload_index(
                collection_name=self.collection, field_name=field, field_schema=schema
            )

    def _check_dim(self, dim: int) -> None:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        actual = vectors[DENSE].size if isinstance(vectors, dict) else vectors.size
        if actual != dim:
            raise ValueError(
                f"컬렉션 차원 불일치: '{self.collection}'은 {actual}차원인데 "
                f"임베딩 모델은 {dim}차원을 낸다. 임베딩 모델을 바꿨다면 "
                f"recreate=True 로 재생성하고 전체를 다시 인덱싱해야 한다."
            )

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------ 쓰기

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               batch_size: int = 128) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"청크 {len(chunks)}개 vs 벡터 {len(vectors)}개 — 개수 불일치")
        for i in range(0, len(chunks), batch_size):
            cs, vs = chunks[i:i + batch_size], vectors[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(id=c.id, vector={DENSE: v}, payload=c.payload())
                    for c, v in zip(cs, vs)
                ],
                wait=True,
            )

    def delete_file(self, file_ids: list[str]) -> None:
        """이 파일들의 청크를 전부 지운다.

        파일이 수정되면 청크 수가 줄어들 수 있다. 새 청크만 upsert 하면
        예전의 남는 청크가 유령처럼 검색에 계속 잡히므로, 재인덱싱 전에 먼저 지운다.
        """
        if not file_ids:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="file_id",
                                          match=models.MatchAny(any=file_ids))
                ])
            ),
            wait=True,
        )

    def delete_root(self, roots: list[str]) -> None:
        """폴더(root) 단위로 청크를 전부 지운다.

        "일단 다 넣고 나중에 빼자"를 실제로 가능하게 하는 함수다.
        root 에 페이로드 인덱스가 있으므로 수만 개여도 빠르다.
        """
        if not roots:
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(key="root", match=models.MatchAny(any=roots))
                ])
            ),
            wait=True,
        )

    def count_root(self, root: str) -> int:
        return self.client.count(
            self.collection, exact=True,
            count_filter=models.Filter(must=[
                models.FieldCondition(key="root", match=models.MatchValue(value=root))
            ]),
        ).count

    # ------------------------------------------------------------ 검색

    def _filter(self, roots: list[str] | None) -> models.Filter | None:
        if not roots:
            return None
        return models.Filter(must=[
            models.FieldCondition(key="root", match=models.MatchAny(any=roots))
        ])

    def search(self, vector: list[float], limit: int = 50,
               roots: list[str] | None = None,
               file_ids: list[str] | None = None) -> list[Hit]:
        flt = self._filter(roots)
        if file_ids:
            cond = models.FieldCondition(key="file_id", match=models.MatchAny(any=file_ids))
            flt = models.Filter(must=[*(flt.must if flt else []), cond])
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [Hit(score=p.score, payload=p.payload or {}) for p in res.points]

    def channel_names(self) -> tuple[str, ...]:
        """이 컬렉션에 있는 Slack 채널 이름 목록.

        질문에 적힌 채널명을 실제 이름으로 되돌리는 데 쓴다. 실측 2,658 포인트에서
        전체 스캔이 30ms 라 질문마다 돌려도 못 쓸 정도는 아니지만, 검색 지연에
        그대로 얹히므로 5분 동안 캐시한다. 주간 인덱싱은 별도 프로세스에서 돌아가므로
        무기한 캐시하면 새 채널이 서비스 재시작 전까지 보이지 않는 문제가 생긴다.
        """
        now = time.monotonic()
        cached = getattr(self, "_channel_names", None)
        cached_at = getattr(self, "_channel_names_cached_at", 0.0)
        if cached is not None and now - cached_at < CHANNEL_NAMES_CACHE_TTL_SECONDS:
            return cached
        names: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=1024,
                offset=offset,
                with_payload=["channel_name"],
                with_vectors=False,
            )
            for point in points:
                name = (point.payload or {}).get("channel_name")
                if name:
                    names.add(str(name))
            if offset is None:
                break
        self._channel_names = tuple(sorted(names))
        self._channel_names_cached_at = now
        return self._channel_names

    def latest_slack_threads(self, limit: int = 50) -> list[Hit]:
        """Slack 컬렉션 전체에서 thread_ts 기준 최신 스레드를 반환한다."""
        latest: dict[tuple[str, str], dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                channel_id = str(payload.get("channel_id", ""))
                thread_ts = str(payload.get("thread_ts", ""))
                if not channel_id or not thread_ts:
                    continue
                key = (channel_id, thread_ts)
                current = latest.get(key)
                if current is None or int(payload.get("index", 0)) < int(
                    current.get("index", 0)
                ):
                    latest[key] = payload
            if offset is None:
                break

        def timestamp(payload: dict) -> float:
            try:
                return float(payload.get("thread_ts", 0))
            except (TypeError, ValueError):
                return 0.0

        ordered = sorted(latest.values(), key=timestamp, reverse=True)
        return [Hit(score=0.0, payload=payload) for payload in ordered[:limit]]

    def search_grouped(self, vector: list[float], groups: int = 25,
                       per_file: int = 3,
                       roots: list[str] | None = None) -> list[Hit]:
        """파일별로 묶어서 검색 — 한 문서가 후보를 독점하지 못하게 한다.

        ## 왜 필요한가

        인덱스에는 청크 수가 압도적인 문서가 섞인다. 실측: EnergyPlus 소프트웨어
        매뉴얼 하나가 8,940 청크, 학회 논문집 하나가 7,161 청크,
        `[Workspace]/LabMeeting/김연구/thesis` 폴더 하나가 전체 인덱스의 33%.

        평범한 검색은 상위 50개를 점수순으로 뽑는데, 질의가 그 거대 문서의 주제와
        가까우면 50개가 **그 문서 한 권으로 채워진다**. 예: "온실 에너지 시뮬레이션
        설정" → EnergyPlus 매뉴얼 8,940개 중 50개. 그러면 연구실이 실제로 작성한
        작업 노트는 후보에 아예 들어오지 못하고, 리랭커는 후보에 없는 것을 고를 수 없다.

        Qdrant 의 group 검색으로 **파일당 최대 per_file 개**만 가져오면, 후보가
        최소 groups 개의 서로 다른 문서에 걸치게 된다. 청크 수가 많은 문서가
        유리해지는 편향이 사라진다.
        """
        res = self.client.query_points_groups(
            collection_name=self.collection,
            query=vector,
            using=DENSE,
            group_by="file_id",
            limit=groups,
            group_size=per_file,
            query_filter=self._filter(roots),
            with_payload=True,
        )
        hits: list[Hit] = []
        for g in res.groups:
            for p in g.hits:
                hits.append(Hit(score=p.score, payload=p.payload or {}))
        hits.sort(key=lambda h: -h.score)
        return hits

    def get_neighbors(self, file_id: str, index: int, radius: int = 1) -> list[Hit]:
        """같은 파일의 앞뒤 청크를 index 순번으로 정확히 가져온다 (벡터 검색 아님).

        문맥 확장(small-to-big)용 — 리랭커가 고른 청크 하나만으로는 문맥이 부족할
        때(대명사·표 제목이 앞 청크에 있는 등), 검색으로 다시 찾는 대신 같은 파일
        안에서 바로 옆 청크를 정확히 지목해서 가져온다.
        """
        flt = models.Filter(must=[
            models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
            models.FieldCondition(key="index",
                                  range=models.Range(gte=index - radius, lte=index + radius)),
        ])
        points, _ = self.client.scroll(
            collection_name=self.collection, scroll_filter=flt,
            limit=2 * radius + 4, with_payload=True,
        )
        hits = [Hit(score=0.0, payload=p.payload or {}) for p in points]
        hits.sort(key=lambda h: h.payload.get("index", 0))
        return hits

    def latest_file_in_root(self, root: str) -> dict | None:
        """이 폴더 전체에서 mod_time(드라이브 수정 시각)이 가장 최신인 파일 payload.

        "가장 최근 X" 질문에서 X가 어느 폴더 얘기인지는 알아냈지만(의미검색으로),
        그 폴더 안에서 진짜 최신 파일을 찾는 건 다시 의미검색에 맡기면 안 된다 —
        비슷한 양식의 문서(매주 같은 틀로 쓰는 회의록 등)가 많으면 임베딩이
        날짜 숫자 하나 차이를 구분 못 해서 후보 상위권에도 못 드는 파일이 실제
        최신일 수 있다(실측). 폴더 전체를 스캔해 메타데이터로 직접 비교한다.
        """
        flt = models.Filter(must=[
            models.FieldCondition(key="root", match=models.MatchValue(value=root)),
        ])
        best: dict | None = None
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, scroll_filter=flt,
                limit=1000, offset=offset,
                with_payload=["file_id", "file_name", "mod_time"],
            )
            if not points:
                break
            for p in points:
                pl = p.payload or {}
                if best is None or pl.get("mod_time", "") > best.get("mod_time", ""):
                    best = pl
            if offset is None:
                break
        return best

    def list_files_in_root(self, root: str) -> list[dict]:
        """이 폴더에 인덱싱된 파일들의 고유 이름 목록 (mod_time 최신순).

        "이 폴더에 뭐가 있어?" 는 검색 질의가 아니라 목록 요청이다 — 의미검색은
        "질문과 비슷한 조각"을 찾을 뿐 "이 폴더의 파일 목록"을 돌려주지 않는다.
        (이미지·영상처럼 본문이 없어 인덱싱 안 된 파일은 여기 안 잡힌다 — 그건
        `mode=listing` 폴더의 "위치 정보" 청크를 따로 봐야 한다.)
        """
        flt = models.Filter(must=[
            models.FieldCondition(key="root", match=models.MatchValue(value=root)),
        ])
        by_name: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, scroll_filter=flt,
                limit=1000, offset=offset,
                with_payload=["file_name", "mod_time"],
            )
            if not points:
                break
            for p in points:
                pl = p.payload or {}
                name = pl.get("file_name")
                if not name:
                    continue
                if name not in by_name or pl.get("mod_time", "") > by_name[name]:
                    by_name[name] = pl.get("mod_time", "")
            if offset is None:
                break
        return sorted(
            ({"file_name": n, "mod_time": t} for n, t in by_name.items()),
            key=lambda d: d["mod_time"], reverse=True,
        )

    def list_subfolders_in_root(self, root: str) -> list[dict]:
        """이 root 바로 아래 1단계 하위폴더 목록 (mod_time 최신순).

        "최근 3년간 제출한 논문들 알려줘" 같은 질문은 파일 하나가 아니라 "폴더
        구조 자체"가 답이다 — `[Workspace]/논문` 아래 1단계 폴더(예: "Agronomy(published
        Feb 2024)")가 논문 프로젝트 하나씩과 대응한다. `list_files_in_root`가
        파일명을 세는 것과 같은 문제를 겪는다 — 의미검색으로는 "이 폴더 밑에
        어떤 하위폴더들이 있는지"를 답할 수 없다.
        """
        flt = models.Filter(must=[
            models.FieldCondition(key="root", match=models.MatchValue(value=root)),
        ])
        # Qdrant 포인트는 청크 단위다 — 그냥 개수를 세면 "파일 수"가 아니라
        # "청크 수"가 된다(실측: 파일 124개짜리 폴더가 청크 3412개로 잘못
        # 집계됐다, Codex 리뷰 지적). file_id 집합으로 중복 제거해야 한다.
        by_folder: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection, scroll_filter=flt,
                limit=1000, offset=offset,
                with_payload=["path", "mod_time", "file_id"],
            )
            if not points:
                break
            for p in points:
                pl = p.payload or {}
                path = pl.get("path", "")
                top, sep, _ = path.partition("/")
                if not sep:
                    top = "(루트 직속 파일)"
                entry = by_folder.setdefault(top, {"file_ids": set(), "latest": ""})
                fid = pl.get("file_id")
                if fid:
                    entry["file_ids"].add(fid)
                mt = pl.get("mod_time", "")
                if mt > entry["latest"]:
                    entry["latest"] = mt
            if offset is None:
                break
        out = [
            {"name": name, "mod_time": info["latest"], "n_files": len(info["file_ids"])}
            for name, info in by_folder.items()
        ]
        return sorted(out, key=lambda d: d["mod_time"], reverse=True)

    def chunks_of_file(self, file_id: str, limit: int = 200) -> list[Hit]:
        """한 파일의 청크를 순번(index)대로 전부 가져온다 (벡터 검색 아님).

        Qdrant scroll 은 반환 순서를 payload["index"] 순서로 보장하지 않는다.
        한 번의 scroll 호출로 limit 개만 받으면(예전 구현) 청크 수가 limit 를
        넘는 파일에서 "앞 N개"가 아니라 "임의의 N개"를 받은 뒤 정렬하는 꼴이 되어
        중간·끝 청크가 조용히 누락될 수 있다(Codex 리뷰 지적). offset 으로
        끝까지 모은 뒤 정렬하고 나서 limit 로 자르면 최소한 "실제 index 상 앞
        N개"라는 보장이 생긴다.
        """
        flt = models.Filter(must=[
            models.FieldCondition(key="file_id", match=models.MatchValue(value=file_id)),
        ])
        points = []
        offset = None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection, scroll_filter=flt,
                limit=1000, offset=offset, with_payload=True,
            )
            points.extend(batch)
            if offset is None:
                break
        hits = [Hit(score=0.0, payload=p.payload or {}) for p in points]
        hits.sort(key=lambda h: h.payload.get("index", 0))
        return hits[:limit]

    def stats(self) -> dict:
        if not self.exists():
            return {"exists": False}
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        dim = vectors[DENSE].size if isinstance(vectors, dict) else vectors.size
        return {
            "exists": True,
            "points": info.points_count,
            "dim": dim,
            "status": str(info.status),
        }
