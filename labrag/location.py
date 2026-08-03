"""위치 찾기 — "그 파일 어디 있어?" 는 검색이 아니라 목록 조회다.

## 왜 별도 기능인가

이 시스템의 실사용 목적 1순위는 **파일·폴더가 어디 있는지 찾는 것**이다(사용자 확인,
2026-07-30). 그런데 기존 경로는 청크(문서 토막) 검색으로 답을 만들기 때문에 위치
질문에 구조적으로 약하다. 실측:

    "연구비 정산 서류 어디 있어?"
      기존: 국가연구개발혁신법 매뉴얼을 인용해 법령 해설을 하고
            "자료에는 정산 서류 파일 위치는 명시되어 있지 않습니다" 로 끝냈다.
            (연구실에 [Workspace]/연구비/2023 연구비 내역.xlsx 가 실제로 있는데도)

원인은 데이터가 아니라 **경쟁 단위**였다. 파일 요약 계층은 정답을 1위로 찾아냈는데
(0.548), 청크 후보와 같은 통에 넣고 점수순으로 줄 세우니 법령 매뉴얼 청크(0.562)에
밀렸다. 단위가 다른 두 신호를 한 통에서 경쟁시킨 것이 잘못이었다.

## 설계

파일 단위 신호만 쓴다. 청크는 **아예 보지 않는다**.

  1. 파일명·경로 글자 일치 (`catalog.py`, manifest.sqlite, LLM 0회)
  2. 파일 요약 의미 일치 (`lab_docs_docsyn`, 파일당 벡터 1개)

점수 크기가 다른 두 순위는 RRF(순위 역수 합)로 합친다 — 점수 단위를 억지로 맞추지
않아도 되고 임의 가중치를 안 만든다.

**위치는 manifest 의 실제 root/path 를 그대로 낸다.** 파일 요약은 AI 가 쓴 것이라
틀릴 수 있으므로 찾는 데만 쓰고 위치의 근거로 쓰지 않는다(Codex 지적). 답변도
생성 모델을 거치지 않고 코드가 직접 만든다 — 경로를 옮겨 쓰다 틀릴 여지를 없앤다.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from .catalog import DocumentRecord, ManifestCatalog, extract_lexical_terms
from .models import Models
from .store import Store

RRF_K = 60          # 순위 역수 합의 완충 상수 (관례값)
SIGNAL_DEPTH = 40   # 각 신호에서 가져올 파일 수
TOP_FILES = 5       # 기본 출력 개수

# 요약 신호만 걸린 항목은 최고 유사도의 이 비율 미만이면 버린다.
# RRF 점수로는 못 자른다 — 신호가 하나뿐이면 인접 순위 점수가 1/61 vs 1/63 처럼
# 거의 같아서 급락이 안 보인다. 실측("데이터베이스 교재"): 정답 2개가 0.590·0.579
# 인데 무관한 SQLite 노트북들이 0.470·0.449·0.447 로 뚜렷한 골이 있었다.
# 이 비율은 위치 평가셋으로 보정해야 하는 임시값이다.
SYN_ONLY_RATIO = 0.88

# ── 의도 감지 ──────────────────────────────────────────────────────────
# 정규식으로 시작한다(생성 모델 분류는 지연이 늘고, 오분류 로그가 쌓인 뒤에
# 비교하는 게 맞다 — Codex 권고). 대신 두 방향을 다 보고 3상태로 나눈다:
# 위치 표현만 있으면 location, 내용 표현이 섞이면 ambiguous, 위치 표현이
# 아예 없으면 content(= 기존 동작 그대로). 이렇게 하면 위치 표현이 없는
# 질문은 지금과 100% 동일하게 처리된다.
# 파일류를 가리키는 명사. 실사용 질문에서 "목록", "시트", "표", "가이드",
# "영수증" 같은 것도 파일을 뜻하는 말로 쓰였다(사용자 작성 30문항에서 확인).
_ARTIFACT_ALT = (r"파일|자료|서류|문서|폴더|엑셀|보고서|논문|교재|양식|서식|매뉴얼|"
                 r"사진|영상|목록|리스트|시트|표|가이드|영수증|기록|노트|계획서|보고자료")
_ARTIFACT_WORD = rf"(?:{_ARTIFACT_ALT})"
# 확장자가 보이면 특정 파일을 가리키는 것이 거의 확실하다. 다만 확장자만으로
# 판단하면 안 된다 — "lab3_path_planning.py 로 생성된 PNG 파일 이름은?" 처럼
# 내용을 묻는 질문도 확장자를 포함한다. 찾는 동작이 같이 있어야 한다.
_EXT = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|hwpx?|csv|txt|md|ipynb|zip|jpe?g|png|mp4|hwp)\b",
                  re.I)
_SEEK_VERB = r"(?:찾아|찾을|찾고|찾는|찾지|열어|보여|알려|확인)"

# 위치를 묻는 표현. "어디"와 "있"/동사 사이에 조사·부사가 끼는 실제 표현
# ("어디에 있어", "어디에 정리돼 있어")을 잡으려고 사이를 허용한다.
_LOC = re.compile(
    r"어디\s*(?:에|서|엔|에다)?\s*\S{0,6}\s*(?:있|계|보관|저장|정리|들어|담|올려|넣)|"
    r"어딨|"
    rf"어느\s*(?:폴더|경로|디렉|드라이브|{_ARTIFACT_ALT})|"          # 어느 폴더/문서/표…
    rf"{_ARTIFACT_WORD}\s*위치|위치\s*(?:알려|찾|정보|어디)|"
    r"경로\s*(?:알려|어디|가\s*어디)|"
    r"보관\s*(?:돼|되어|된|중)|"
    rf"{_ARTIFACT_WORD}[은는이가를을]?\s*{_SEEK_VERB}|"
    rf"{_ARTIFACT_WORD}[은는이가를]?\s*어디|"
    rf"어디\s*{_ARTIFACT_WORD}")
# "어디에 작성해야 해", "어느 문서에 기록해야 해" — 위치가 아니라 '작업 대상'을 묻는다.
_WRITE = re.compile(
    r"(?:어디|어느\s*\S{1,6})\s*(?:에|에다|를|을)?\s*\S{0,4}\s*"
    r"(?:작성|기록|입력|써야|적어야|넣어야|올려야|제출)|"
    r"(?:작성|기록|입력|제출)\s*(?:해야|하면|할|하는)\s*\S{0,4}\s*(?:곳|문서|파일|양식|표|시트)")
_CONTENT = re.compile(
    r"어떻게|어떤\s*(방법|절차|식|원리)|왜|무슨\s*뜻|의미(가|는)|설명해|"
    r"요약해|정리해|계산|차이(가|는|점)|방법(은|이|을)|절차(는|가|를)|"
    r"내용(은|이|을)\s*(뭐|무엇|알려)")
_ARTIFACT = re.compile(r"파일|자료|서류|문서|폴더|엑셀|보고서")
_SEEK = re.compile(r"찾아|찾을|보여|알려")


def detect_intent(query: str) -> str:
    """"write_target" | "location" | "ambiguous" | "content".

    위치 표현이 아예 없으면 content 로 떨어져 기존 경로가 그대로 쓰인다 —
    기존 96문항 회귀를 막는 안전판이다.
    """
    if _WRITE.search(query):
        return "write_target"
    # 확장자 + 찾는 동작이면 특정 파일을 달라는 요청이다("'깃허브 사용법.docx' 찾아줘")
    if _EXT.search(query) and re.search(_SEEK_VERB, query):
        return "location"
    has_loc = bool(_LOC.search(query))
    has_content = bool(_CONTENT.search(query))
    if has_loc and not has_content:
        return "location"
    if has_loc and has_content:
        return "ambiguous"
    if _ARTIFACT.search(query) and _SEEK.search(query):
        return "ambiguous"
    return "content"


# ── 결과 계약 ──────────────────────────────────────────────────────────

@dataclass
class LocatedFile:
    doc: DocumentRecord
    rrf: float
    lexical_rank: int | None = None
    synopsis_rank: int | None = None
    synopsis_score: float = 0.0
    synopsis: str | None = None
    file_url: str = ""
    evidence: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        """두 신호가 다 걸렸으면 높음. 요약만 걸린 것도 정답일 수 있으니 버리지 않는다."""
        if self.lexical_rank and self.synopsis_rank:
            return "high"
        if self.lexical_rank and self.lexical_rank <= 5:
            return "medium"
        return "low"

    @property
    def location(self) -> str:
        """사람이 읽는 위치. manifest 의 실제 값만 쓴다."""
        if self.doc.path and self.doc.path != self.doc.name:
            return f"{self.doc.root}/{self.doc.path}"
        return self.doc.root


@dataclass
class LocationResult:
    query: str
    files: list[LocatedFile]
    intent: str = "location"
    note: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.files)


# ── "어디에 작성해야 해?" 는 위치가 아니라 작업 대상을 묻는다 ────────────
# 가장 최근 수정 파일을 답으로 내면 위험하다 — 제출본·복사본·결과물이 최신
# 수정일일 수 있다(Codex 지적). 그래서 역할을 나눠 보여주고 단정하지 않는다.
_TEMPLATE_HINT = re.compile(r"양식|서식|template|form|작성용|빈\s?파일|제출서류|샘플|예시", re.I)
_ARCHIVE_HINT = re.compile(r"제출본|완료|최종|복사본|backup|백업|draft|초안|_구|이전|old", re.I)


def classify_write_role(f: LocatedFile) -> str:
    """"양식" | "과거·제출본" | "작업본" — 파일명·경로만 보고 나누는 추정이다."""
    text = f"{f.doc.name} {f.doc.path}"
    if _TEMPLATE_HINT.search(text):
        return "양식"
    if _ARCHIVE_HINT.search(text):
        return "과거·제출본"
    return "작업본"


WRITE_NOTE = ("작성할 파일 후보를 역할별로 나눠 봤어. 파일명·경로로 추정한 것이라 "
              "실제로 어디에 써야 하는지는 파일 안의 작성 규칙이나 담당자 확인이 필요해.")


# ── 검색 ───────────────────────────────────────────────────────────────

def _file_urls(store: Store, file_ids: list[str]) -> dict[str, str]:
    """청크 payload 에 저장된 실제 드라이브 링크를 파일별로 하나씩 가져온다.

    manifest 에는 mime_type 이 없어서 Google 네이티브 문서의 링크 형태를 추정할 수
    없다. 인덱싱 때 이미 계산해 저장해둔 값을 쓰는 것이 정확하다.
    """
    from qdrant_client import models as qm
    out: dict[str, str] = {}
    if not file_ids:
        return out
    try:
        pts, _ = store.client.scroll(
            collection_name=store.collection,
            scroll_filter=qm.Filter(must=[
                qm.FieldCondition(key="file_id", match=qm.MatchAny(any=file_ids))]),
            limit=len(file_ids) * 4, with_payload=["file_id", "file_url"])
        for p in pts:
            pl = p.payload or {}
            fid, url = pl.get("file_id"), pl.get("file_url")
            if fid and url and fid not in out:
                out[fid] = url
    except Exception as exc:
        print(f"[location] 링크 조회 실패, 기본 링크로 대체: {exc}")
    for fid in file_ids:
        if fid not in out:
            primary = fid.split()[0] if fid.split() else fid
            out[fid] = f"https://drive.google.com/file/d/{primary}/view"
    return out


def _trim(files: list[LocatedFile]) -> list[LocatedFile]:
    """상위 몇 개만 남기고, 근거가 약한 꼬리를 버린다.

    "두 신호가 다 걸려야 한다"는 규칙은 쓰면 안 된다 — 파일명을 모르고 묻는 질문에서는
    요약만 맞는 것이 정상적인 정답이다(Codex 지적). 그래서 요약만 걸린 항목끼리
    비교해서 최고 유사도 대비 많이 떨어지는 것만 버린다.
    """
    best_syn = max((f.synopsis_score for f in files), default=0.0)
    kept: list[LocatedFile] = []
    for f in files:
        syn_only = f.lexical_rank is None and f.synopsis_rank is not None
        if syn_only and best_syn and f.synopsis_score < best_syn * SYN_ONLY_RATIO:
            continue
        kept.append(f)
        if len(kept) >= TOP_FILES:
            break
    return kept


def search_files(query: str, models: Models, catalog: ManifestCatalog,
                 docsyn: Store | None, chunk_store: Store,
                 allowed_roots: list[str] | None = None) -> LocationResult:
    """파일 단위 위치 검색. 청크는 보지 않는다."""
    ranks: dict[str, dict] = {}

    terms = extract_lexical_terms(query)
    if terms:
        for i, sd in enumerate(catalog.search_documents(
                terms, allowed_roots=allowed_roots or (),
                limit=SIGNAL_DEPTH, statuses=("indexed", "skipped")), 1):
            e = ranks.setdefault(sd.doc.file_id, {"doc": sd.doc, "ev": []})
            e["lex"] = i
            e["ev"].append(f"이름·경로 일치({', '.join(sd.reasons[:2])})")

    if docsyn is not None:
        try:
            vec = models.embed_one(query)
            for i, h in enumerate(docsyn.search(vec, limit=SIGNAL_DEPTH,
                                                roots=allowed_roots), 1):
                fid = h.payload.get("file_id")
                if not fid:
                    continue
                doc = catalog.get_by_file_id(fid)
                if doc is None:            # manifest 에서 사라진 파일은 안내하지 않는다
                    continue
                e = ranks.setdefault(fid, {"doc": doc, "ev": []})
                e["sem"] = i
                e["sem_score"] = h.score
                e["syn"] = h.payload.get("synopsis", "")
                e["ev"].append(f"요약 의미 일치(유사도 {h.score:.3f})")
        except Exception as exc:
            print(f"[location] 파일 요약 검색 실패, 이름·경로만 사용: {exc}")

    if not ranks:
        return LocationResult(query=query, files=[])

    urls = _file_urls(chunk_store, list(ranks))
    files = [
        LocatedFile(doc=e["doc"],
                    rrf=sum(1.0 / (RRF_K + e[k]) for k in ("lex", "sem") if k in e),
                    lexical_rank=e.get("lex"), synopsis_rank=e.get("sem"),
                    synopsis_score=e.get("sem_score", 0.0),
                    synopsis=e.get("syn"), file_url=urls.get(e["doc"].file_id, ""),
                    evidence=e["ev"])
        for e in ranks.values()
    ]
    files.sort(key=lambda f: -f.rrf)
    return LocationResult(query=query, files=_trim(files))


# ── 답변 만들기 (생성 모델을 쓰지 않는다) ──────────────────────────────

def render(result: LocationResult) -> str:
    """마크다운 답변. 경로를 생성 모델이 다시 쓰지 않게 코드가 직접 만든다."""
    if not result.found:
        return ("이름·경로와 파일 설명으로 찾아봤는데 해당하는 파일을 못 찾았어.\n\n"
                "폴더 이름이나 파일명 일부를 알면 그걸로 다시 물어봐. "
                "내용을 찾는 질문이면 \"~가 뭐야\", \"~어떻게 해\" 처럼 물어보면 "
                "문서 내용을 찾아서 답할게.")

    write_mode = result.intent == "write_target"
    if write_mode:
        lines = [f"작성 대상 후보 {len(result.files)}개를 찾았어.", "",
                 "| 구분 | 파일 | 위치 | 수정일 |", "|---|---|---|---|"]
    else:
        lines = [f"관련 파일 {len(result.files)}개를 찾았어.", "",
                 "| 파일 | 위치 | 수정일 |", "|---|---|---|"]
    for f in result.files:
        name = f"[{f.doc.name}]({f.file_url})" if f.file_url else f.doc.name
        mark = "" if f.doc.status == "indexed" else " ⚠️"
        role = f"{classify_write_role(f)} | " if write_mode else ""
        lines.append(f"| {role}{name}{mark} | `{f.location}` | {f.doc.mod_time[:10]} |")

    lines.append("")
    for i, f in enumerate(result.files, 1):
        bits = [f"**{i}. {f.doc.name}**"]
        if f.synopsis:
            bits.append(f"  {f.synopsis}")
        if f.doc.status != "indexed":
            bits.append(f"  ⚠️ 본문은 검색 대상이 아니야 — {f.doc.error or '사유 미기록'}")
        bits.append(f"  <sub>{f.doc.category} · 조각 {f.doc.n_chunks}개 · "
                    f"{', '.join(f.evidence[:2])}</sub>")
        lines.append("\n".join(bits))

    if result.note:
        lines += ["", f"> {result.note}"]
    lines += ["", "<sub>위치는 드라이브 목록에서 그대로 가져온 값이야. "
                  "수정일은 파일이 마지막으로 바뀐 날이고 문서 내용상의 날짜와는 다를 수 있어.</sub>"]
    return "\n".join(lines)


AMBIGUOUS_NOTE = ("위치를 먼저 찾았어. 내용이나 방법·절차를 알고 싶은 질문이었으면 "
                  "\"~어떻게 해\", \"~내용 알려줘\" 처럼 다시 물어봐.")


# ── 분기 ───────────────────────────────────────────────────────────────

def maybe_locate(query: str, models: Models, catalog: ManifestCatalog | None,
                 docsyn: Store | None, chunk_store: Store,
                 canonical_config: dict | None = None,
                 roots: list[str] | None = None,
                 force: bool = False) -> LocationResult | None:
    """위치 질문이면 LocationResult, 아니면 None(= 기존 내용 검색으로 넘긴다).

    이미 잘 동작하는 기존 처리기(폴더 목록, canonical 파일 조회)가 먼저다 —
    그것들을 여기서 가로채면 회귀가 된다. `retrieve()` 안에 남겨두고 건너뛴다.
    """
    if catalog is None:
        return None

    from .catalog import detect_canonical_intent
    from .rag import LIST_FOLDER_PATTERN

    if not force and LIST_FOLDER_PATTERN.search(query):
        return None                              # "○○폴더에 뭐가 있어" → 기존 처리기
    if (not force and canonical_config
            and detect_canonical_intent(query, canonical_config)):
        return None                              # 구성원 명단 등 → 기존 처리기

    intent = detect_intent(query)
    if intent == "content" and not force:
        return None
    if intent == "content":
        intent = "location"

    result = search_files(query, models, catalog, docsyn, chunk_store,
                          allowed_roots=roots)
    result.intent = intent
    if not force and not result.found and intent == "ambiguous":
        return None       # 애매한 질문인데 파일도 못 찾았으면 내용 검색이 나을 수 있다
    if intent == "ambiguous":
        result.note = AMBIGUOUS_NOTE
    elif intent == "write_target":
        result.note = WRITE_NOTE
    return result


async def maybe_locate_hybrid(
    query: str, models: Models, catalog: ManifestCatalog | None,
    docsyn: Store | None, chunk_store: Store,
    drive_client=None, ancestry=None, canonical_config: dict | None = None,
    roots: list[str] | None = None, timeout: float = 5.0,
    force: bool = False,
):
    """사전 인덱스와 실시간 Drive를 병렬 검색하고 장애 시 기존 결과를 보존한다."""
    if catalog is None:
        return None
    from .catalog import detect_canonical_intent
    from .rag import LIST_FOLDER_PATTERN

    if not force and LIST_FOLDER_PATTERN.search(query):
        return None
    if (not force and canonical_config
            and detect_canonical_intent(query, canonical_config)):
        return None
    intent = detect_intent(query)
    if intent == "content" and not force:
        return None
    if intent == "content":
        intent = "location"

    indexed_task = asyncio.create_task(asyncio.to_thread(
        maybe_locate, query, models, catalog, docsyn, chunk_store,
        canonical_config, roots, force,
    ))
    if drive_client is None or ancestry is None:
        return await indexed_task

    deadline = time.monotonic() + timeout

    async def live_branch():
        items, status = await drive_client.search(query, deadline=deadline, limit=20)
        if status.state != "ok":
            return [], status
        tasks = [
            asyncio.create_task(ancestry.verify(item, deadline=deadline))
            for item in items
        ]
        try:
            locations = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=max(0.001, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()
            from .drive_live import DriveSearchStatus
            return [], DriveSearchStatus("timeout")
        verified = [
            (item, location) for item, location in zip(items, locations)
            if not isinstance(location, Exception) and location.inside_root
        ]
        return verified, status

    live_task = asyncio.create_task(live_branch())
    try:
        indexed, (live_files, live_status) = await asyncio.gather(
            indexed_task,
            asyncio.wait_for(live_task, timeout=max(0.001, deadline - time.monotonic())),
        )
    except Exception:
        live_task.cancel()
        indexed = await indexed_task
        return indexed
    if indexed is None:
        return None
    from .location_merge import merge_location_results
    return merge_location_results(query, intent, indexed, live_files, live_status)


def render_hybrid(result) -> str:
    """하이브리드 위치 결과를 출처와 현재 메타데이터와 함께 렌더링한다."""
    from .location_merge import HybridLocationResult
    if not isinstance(result, HybridLocationResult):
        return render(result)
    if not result.found:
        suffix = ("Google Drive 실시간 확인도 완료했어."
                  if result.live_status.state == "ok"
                  else f"Google Drive 실시간 확인 상태: {result.live_status.state}")
        return f"해당하는 파일을 찾지 못했어.\n\n{suffix}"
    lines = [
        f"관련 파일 {len(result.files)}개를 찾았어.", "",
        "| 파일 | 현재 위치 | 기준일 | 출처 |",
        "|---|---|---|---|",
    ]
    for found in result.files:
        link = f"[{found.name}]({found.file_url})" if found.file_url else found.name
        stamp = (found.modified_time or found.created_time)[:10]
        lines.append(f"| {link} | `{found.path}` | {stamp} | {found.provenance} |")
    lines.append("")
    for i, found in enumerate(result.files, 1):
        lines.append(f"**{i}. {found.name}**  \n<sub>{' · '.join(found.evidence[:3])}</sub>")
    if result.note:
        lines += ["", f"> {result.note}"]
    return "\n".join(lines)
