"""manifest.sqlite 문서 카탈로그 — 청크가 아니라 "파일" 단위 검색.

## 왜 필요한가

Qdrant는 청크 단위 dense 검색만 한다. "연구실 인원 정보.xlsx"처럼 파일명 자체가
정답을 강하게 알려주는 질문은, 질문의 표현이 문서 본문과 다르면(예: "2026년
기준"이라는 시점 표현이 실제 셀 내용과 안 겹침) 임베딩 후보에서 아예 탈락할 수
있다 (실측: `2026년 기준 랩 구성원들 알려줘`에 이 파일이 dense 상위 20위 안에도
없었다). manifest.sqlite 는 이미 모든 파일의 이름·경로를 갖고 있으니, 이 정보로
dense 후보를 보강한다.

## 왜 Qdrant 페이로드에 full-text 인덱스를 만들지 않았는가

파일명·경로가 청크마다 중복 저장돼야 하고, 한국어 파일명 토큰화를 실측 없이
믿기 어렵다. manifest 는 이미 파일 단위 레코드라 검색 로직과 점수를 Python에서
명확히 통제할 수 있다.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# 요청 표현·조사·연도·너무 일반적인 명사는 파일명 후보 키워드로 가치가 낮다.
_STOPWORDS = {
    "알려줘", "보여줘", "찾아줘", "말해줘", "요약해줘", "설명해줘", "정리해줘",
    "에", "에서", "의", "으로", "기준", "대한", "관련", "이", "가", "은", "는", "들",
    "우리", "이거", "그거", "해당", "최근", "최신", "현재", "올해", "그리고",
    "내용", "자료", "정보", "파일", "문서", "목록", "무엇", "어디", "누구", "언제", "뭐",
}
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_YEAR_RE = re.compile(r"^\d{4}년?$|^\d{4}년도$")


def normalize(text: str) -> str:
    """질의와 manifest의 name/path에 같은 정규화를 적용해야 비교가 된다."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\\/_\-.()\[\]{}／·]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _root_segments(root: str) -> tuple[str, ...]:
    """폴더 경로를 계층 그대로 유지한 채 정규화한다 ("[Workspace]/연구실" → ("lab", "연구실")).

    ``normalize()``는 "/"를 공백으로 뭉개버려서 "lab 연구실"처럼 계층 정보를 잃는다.
    그 문자열로 부분일치를 하면 "lab"이라는 prefix 하나가 다른 폴더까지 다 잡아먹는
    문제가 생긴다(Codex 리뷰 지적). 경로 비교는 구간(segment) 단위로 해야 한다.
    """
    return tuple(normalize(p) for p in root.split("/") if p.strip())


def _root_within(doc_segments: tuple[str, ...], filter_root: str) -> bool:
    """doc의 root가 filter_root 폴더 아래(또는 그 자신)에 있는가 — 계층 기준, 부분문자열 아님."""
    filt = _root_segments(filter_root)
    return doc_segments[:len(filt)] == filt


_TRAILING_PARTICLES = ("들", "은", "는", "이", "가", "을", "를", "에서", "에", "의", "으로", "로")


def extract_lexical_terms(query: str) -> list[str]:
    """질문에서 파일명 검색에 쓸 키워드를 뽑는다 (규칙 기반, LLM 호출 없음).

    완전한 형태소 분석이 목표가 아니다 — dense 검색이 놓친 문서를 후보에 추가하는
    것으로 충분하다. 요청 표현("알려줘")·조사·단독 연도·너무 일반적인 명사는
    제거하고, 남는 한글/영문/숫자 토큰(2자 이상)만 쓴다. 한국어는 조사가 명사에
    바로 붙어 하나의 토큰이 되므로("구성원들"), 흔한 조사를 뗀 변형도 추가한다 —
    형태소 분석기 없이 싼 값에 얻을 수 있는 recall 이다.
    """
    norm = normalize(query)
    tokens = _TOKEN_RE.findall(norm)
    terms = []
    for t in tokens:
        if t in _STOPWORDS or t.isdigit() or len(t) < 2 or _YEAR_RE.match(t):
            continue
        terms.append(t)
        for p in _TRAILING_PARTICLES:
            if t.endswith(p) and len(t) - len(p) >= 2:
                stripped = t[: -len(p)]
                if stripped not in _STOPWORDS:
                    terms.append(stripped)
                break
    return list(dict.fromkeys(terms))  # 순서 보존 dedup — 같은 term이 점수를 중복 가산하지 않게


@dataclass(frozen=True)
class DocumentRecord:
    file_id: str
    name: str
    path: str
    root: str
    category: str
    mod_time: str
    n_chunks: int
    status: str = "indexed"
    error: str | None = None


@dataclass(frozen=True)
class ScoredDocument:
    doc: DocumentRecord
    score: float
    reasons: tuple[str, ...]


class ManifestCatalog:
    """manifest.sqlite 를 읽기 전용으로 열어 파일 단위 후보를 찾는다.

    서버 시작 시 한 번 메모리에 로딩한다 — 재인덱싱 중에도 서버가 안전하게 읽을
    수 있고(같은 커넥션을 계속 들고 있지 않음), 매 요청마다 SQL을 여러 번 돌리는
    것보다 빠르고 예측 가능하다. 대신 재인덱싱 뒤 새 파일을 반영하려면 서버
    재시작이 필요하다 — 지금은 이 정도로 충분하고, 자동 갱신은 필요해지면 추가.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._docs: list[DocumentRecord] = []
        # 정규화된 문자열을 미리 계산해둔다 — 요청마다 반복 계산하지 않기 위해.
        self._norm_name: list[str] = []
        self._norm_path: list[str] = []
        self._root_segs: list[tuple[str, ...]] = []
        self.reload()

    def reload(self) -> None:
        uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # 'indexed'뿐 아니라 'skipped'도 로딩한다 — 스캔본 PDF처럼 내용은
            # 못 뽑았어도 파일명·위치는 manifest 에 있다. 실측: "예시 서비스 서비스"를
            # 물으면 (연예시 서비스트)예시 서비스_서비스소개서.pdf(스캔본, OCR 필요, n_chunks=0)
            # 가 'indexed'만 보던 기존 쿼리에서는 아예 존재하지 않는 파일처럼
            # 취급돼, 완전히 무관한 파일로 답이 나갔다. 검색 후보로 쓰지는
            # 않되(청크가 없으니 본문을 못 준다) "위치 안내"용으로는 쓴다
            # (rag.py 의 _unindexed_notice, search_documents 의 statuses 파라미터).
            rows = conn.execute(
                "SELECT file_id, name, path, root, category, mod_time, n_chunks, "
                "status, error FROM files WHERE status IN ('indexed', 'skipped')"
            ).fetchall()
        finally:
            conn.close()
        self._docs = [
            DocumentRecord(r["file_id"], r["name"], r["path"], r["root"],
                          r["category"], r["mod_time"], r["n_chunks"],
                          r["status"], r["error"])
            for r in rows
        ]
        self._norm_name = [normalize(d.name) for d in self._docs]
        self._norm_path = [normalize(d.path) for d in self._docs]
        self._root_segs = [_root_segments(d.root) for d in self._docs]

    def __len__(self) -> int:
        return len(self._docs)

    def search_documents(self, terms: list[str], *, allowed_roots: list[str] = (),
                         preferred_roots: list[str] = (),
                         filename_hints: list[str] = (), limit: int = 5,
                         statuses: tuple[str, ...] = ("indexed",)) -> list[ScoredDocument]:
        """파일명·경로 기준으로 후보 파일을 찾는다 (dense 검색과 별개).

        점수는 절대값보다 상대 우선순위가 중요하다: 파일명 구절 일치 > 파일명
        토큰 일치 > 경로 일치. 경로만 맞아서는(예: "[Workspace]/연구실" 아래 파일이면
        전부) 높은 점수를 주지 않는다 — preferred_roots 는 tie-break 보조 신호.

        ``allowed_roots``는 preferred_roots와 다르다 — 점수 가산이 아니라 하드
        필터다. 요청이 특정 폴더로 범위를 좁혔는데(roots=[...]) canonical/lexical이
        그 밖의 폴더 파일을 돌려주면 안 된다(Codex 리뷰에서 지적된 scope 위반).

        ``statuses``는 기본 ("indexed",)만 본다 — 기존 하이브리드/canonical 호출은
        내용이 있는 파일만 대상으로 해야 하므로 그대로 둔다. "skipped"도 보고
        싶으면(위치 안내용) 명시적으로 넘겨야 한다.
        """
        if not terms and not filename_hints:
            return []
        norm_hints = list(dict.fromkeys(normalize(h) for h in filename_hints if h))
        allowed = list(dict.fromkeys(allowed_roots))
        preferred = list(dict.fromkeys(preferred_roots))
        joined_terms = " ".join(terms)

        scored: list[ScoredDocument] = []
        for doc, name_n, path_n, root_segs in zip(self._docs, self._norm_name, self._norm_path, self._root_segs):
            if doc.status not in statuses:
                continue
            if allowed and not any(_root_within(root_segs, r) for r in allowed):
                continue
            score = 0.0
            reasons: list[str] = []
            if terms and joined_terms in name_n:
                score += 60
                reasons.append("파일명에 질의 구절 포함")
            for t in terms:
                if t in name_n:
                    score += 20
                    reasons.append(f"파일명에 '{t}' 포함")
                elif t in path_n:
                    score += 5
                    reasons.append(f"경로에 '{t}' 포함")
            for hint in norm_hints:
                if hint and hint in name_n:
                    score += 30
                    reasons.append(f"파일명 힌트 '{hint}' 일치")
            for root in preferred:
                if root and _root_within(root_segs, root):
                    score += 30
                    reasons.append("선호 경로 일치")
            if score > 0:
                scored.append(ScoredDocument(doc, score, tuple(reasons)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    def get_by_file_id(self, file_id: str) -> DocumentRecord | None:
        for d in self._docs:
            if d.file_id == file_id:
                return d
        return None


# ---------------------------------------------------------------- canonical

@dataclass(frozen=True)
class CanonicalRecord:
    """"이 의도는 이 폴더/파일명 계열을 우선 찾는다"는 탐색 힌트.

    특정 file_id 를 하드코딩하지 않는다 — 파일이 바뀌거나 새로 추가돼도
    안 깨지게, 의미·경로·파일명 힌트만 설정에 담는다 (config/canonical_records.json).
    """
    key: str
    intent_terms: tuple[str, ...]
    context_terms: tuple[str, ...]
    exclude_if_present: tuple[str, ...]
    preferred_roots: tuple[str, ...]
    filename_hints: tuple[str, ...]
    operation: str


def load_canonical_config(path: Path) -> dict[str, CanonicalRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, CanonicalRecord] = {}
    for key, rec in data.get("records", {}).items():
        out[key] = CanonicalRecord(
            key=key,
            intent_terms=tuple(rec.get("intent_terms", [])),
            context_terms=tuple(rec.get("context_terms", [])),
            exclude_if_present=tuple(rec.get("exclude_if_present", [])),
            preferred_roots=tuple(rec.get("preferred_roots", [])),
            filename_hints=tuple(rec.get("filename_hints", [])),
            operation=rec.get("operation", "canonical_record"),
        )
    return out


def detect_canonical_intent(query: str, config: dict[str, CanonicalRecord]) -> CanonicalRecord | None:
    """질문이 어떤 canonical intent에 해당하는지 규칙 기반으로 판단한다.

    "구성원"이라는 단어 하나만 보고 바로 매칭하면 안 된다 — "구성원들이 쓴 논문",
    "구성원별 회의록"처럼 명부 조회가 아닌 질문까지 오분류할 위험이 있다(Codex
    자문에서 지적된 우려사례). `intent_terms` 중 하나 **+ `context_terms` 중
    하나**가 같이 있어야 하고, `exclude_if_present`(논문/회의록/학회 등, 이 intent가
    아니라는 강한 신호)에 걸리면 후보에서 뺀다.
    """
    norm = normalize(query)
    for rec in config.values():
        if not any(normalize(t) in norm for t in rec.intent_terms):
            continue
        if rec.context_terms and not any(normalize(t) in norm for t in rec.context_terms):
            continue
        if any(normalize(t) in norm for t in rec.exclude_if_present):
            continue
        return rec
    return None
