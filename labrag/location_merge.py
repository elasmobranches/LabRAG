"""사전 문서 인덱스와 실시간 Drive 후보를 파일 ID 기준으로 병합한다."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .ancestry import VerifiedLocation
from .catalog import normalize
from .drive_live import DriveItem, DriveSearchStatus
from .location import LocatedFile, LocationResult, RRF_K

_YEAR = re.compile(r"\b(20\d{2})년?")
_MONTH = re.compile(r"(?:20\d{2})[년\-./ ]+\s*(1[0-2]|0?[1-9])월?")
_CREATED = re.compile(r"작성|생성|만든|올린")
_MODIFIED = re.compile(r"수정|업데이트|갱신")
_RECENT = re.compile(r"최신|최근|가장\s*최근")
_FOLDER = re.compile(r"폴더|디렉터리|모아\s*둔")
_EXT = re.compile(r"\.(?:pdf|docx?|xlsx?|pptx?|hwp|hwpx|csv|txt|md|zip)\b", re.I)


@dataclass
class HybridFile:
    file_id: str
    name: str
    path: str
    file_url: str
    created_time: str
    modified_time: str
    mime_type: str
    indexed: LocatedFile | None
    live: DriveItem | None
    live_path: str = ""
    rrf: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> str:
        if self.indexed and self.live:
            return "문서 인덱스 ✓ · Google Drive 실시간 ✓"
        if self.live:
            return "Google Drive에서 새로 발견 · 문서 인덱스 미반영"
        return "문서 인덱스 ✓"


@dataclass
class HybridLocationResult:
    query: str
    intent: str
    files: list[HybridFile]
    live_status: DriveSearchStatus
    note: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.files)


def _query_filename(query: str) -> str:
    quoted = re.findall(r"""['"]([^'"]+\.[A-Za-z0-9]+)['"]""", query)
    if quoted:
        return normalize(quoted[0])
    match = re.search(r"(\S+\.[A-Za-z0-9]+)", query)
    return normalize(match.group(1)) if match else ""


def _stem(value: str) -> str:
    return normalize(PurePosixPath(value).stem)


def _timestamp_mode(query: str) -> str:
    if _CREATED.search(query):
        return "created"
    if _MODIFIED.search(query):
        return "modified"
    return "modified"


def _name_matches_year(name: str, year: str, month: str = "") -> bool:
    normalized = normalize(name)
    if year in normalized:
        return True
    compact = re.sub(r"\D", "", normalized)
    short = year[2:]
    return compact.startswith(short + month) if month else compact.startswith(short)


def _apply_date_filter(query: str, files: list[HybridFile]) -> list[HybridFile]:
    year = _YEAR.search(query)
    month = _MONTH.search(query)
    if not year:
        return files
    y = year.group(1)
    m = f"{int(month.group(1)):02d}" if month else ""
    mode = _timestamp_mode(query)
    filtered = []
    for f in files:
        stamp = f.created_time if mode == "created" else f.modified_time
        if (
            stamp.startswith(y) and (not m or stamp[5:7] == m)
        ) or _name_matches_year(f.name, y, m):
            filtered.append(f)
    return filtered or files


def merge_location_results(
    query: str,
    intent: str,
    indexed: LocationResult,
    live: list[tuple[DriveItem, VerifiedLocation]],
    live_status: DriveSearchStatus,
) -> HybridLocationResult:
    merged: dict[str, HybridFile] = {}
    for rank, found in enumerate(indexed.files, 1):
        fid = found.doc.file_id
        merged[fid] = HybridFile(
            file_id=fid, name=found.doc.name, path=found.location,
            file_url=found.file_url, created_time="",
            modified_time=found.doc.mod_time, mime_type="",
            indexed=found, live=None, rrf=found.rrf,
            evidence=list(found.evidence),
        )
    for rank, (item, verified) in enumerate(live, 1):
        if item.trashed or not verified.inside_root:
            continue
        current = merged.get(item.id)
        contribution = 1.0 / (RRF_K + rank)
        if current is None:
            current = HybridFile(
                file_id=item.id, name=item.name, path=verified.path,
                file_url=item.web_view_link, created_time=item.created_time,
                modified_time=item.modified_time, mime_type=item.mime_type,
                indexed=None, live=item, live_path=verified.path,
                rrf=contribution, evidence=[f"Drive 검색 순위 {rank}"],
            )
            merged[item.id] = current
        else:
            current.name = item.name
            current.path = verified.path
            current.file_url = item.web_view_link or current.file_url
            current.created_time = item.created_time
            current.modified_time = item.modified_time
            current.mime_type = item.mime_type
            current.live = item
            current.live_path = verified.path
            current.rrf += contribution
            current.evidence.append(f"Drive 검색 순위 {rank}")

    files = list(merged.values())
    qname = _query_filename(query)
    qstem = _stem(qname) if qname else ""

    def relevance(f: HybridFile):
        exact = int(bool(qname and normalize(f.name) == qname))
        stem = int(bool(qstem and _stem(f.name) == qstem))
        both = int(bool(f.indexed and f.live))
        folder = int(bool(
            _FOLDER.search(query) and f.mime_type == "application/vnd.google-apps.folder"
        ))
        year_name = int(bool(
            (match := _YEAR.search(query)) and match.group(1) in f.name
        ))
        return (exact, stem, folder, year_name, both, f.rrf, normalize(f.name))

    files.sort(key=relevance, reverse=True)
    if _RECENT.search(query) or _CREATED.search(query) or _MODIFIED.search(query):
        files = _apply_date_filter(query, files)
        mode = _timestamp_mode(query)
        query_year = _YEAR.search(query)
        files.sort(
            key=lambda f: (
                _name_matches_year(f.name, query_year.group(1))
                if query_year else False,
                f.rrf if query_year else 0.0,
                f.created_time if mode == "created" else f.modified_time,
                f.modified_time if mode == "created" else f.created_time,
                normalize(f.name),
            ),
            reverse=True,
        )
    return HybridLocationResult(
        query=query, intent=intent, files=files[:5],
        live_status=live_status, note=indexed.note,
    )
