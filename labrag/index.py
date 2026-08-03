"""인덱싱 오케스트레이션: 드라이브 파일 → Qdrant.

파일 하나당 흐름:

    manifest 확인 → (변경됐으면) 다운로드 → 파싱 → 청킹 → 임베딩
      → 기존 청크 삭제 → upsert → manifest 기록

기존 청크를 먼저 지우는 이유: 파일이 수정되면 청크 수가 줄어들 수 있다.
새 청크만 덮어쓰면 예전의 남는 청크가 유령처럼 검색 결과에 계속 나온다.

실패를 조용히 넘기지 않는다. 파싱 실패·OCR 필요 같은 사유를 manifest 에 남겨서
"몇 개가 왜 빠졌는지"를 항상 셀 수 있게 한다. 이게 없으면 RAG 가 답을 못 찾을 때
"자료가 없는 건지 인덱싱이 안 된 건지" 구분할 수 없다.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import manifest
from .chunk import chunk_file
from .drive import DriveFile
from .fetch import fetch
from .filetypes import NEEDS_CONVERT, NEEDS_MODEL, TEXT_CATEGORIES
from .models import Models
from .parse import ParseError, SkipFile, parse, pdf_text_ratio
from .store import Store

# 텍스트 레이어가 이 비율 미만인 PDF 는 스캔본으로 보고 건너뛴다 (OCR 대상)
MIN_PDF_TEXT_RATIO = 0.2

# 다운로드 동시 실행 수.
# 실측: 파일당 평균 9.7초였고 대부분이 rclone 다운로드 대기였다 (파싱·임베딩은 빠름).
# 다운로드는 I/O 라 스레드로 겹칠 수 있다. 파싱과 임베딩은 순차로 둔다 —
# 임베딩은 GPU 를 쓰므로 병렬로 밀어넣어도 이득이 없고 VRAM 만 위험해진다.
# 6 정도면 Drive API 한도(사용자당 100초에 약 1000요청)에 한참 못 미친다.
DOWNLOAD_WORKERS = 6


@dataclass
class Result:
    indexed: int = 0
    chunks: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    # 이전 실행에서 실패해 이번엔 재시도하지 않은 파일. '변경없음'과 절대 합치지 않는다 —
    # 합치면 실패가 정상으로 둔갑해서 "왜 이 문서를 못 찾지?"를 추적할 수 없게 된다.
    still_failed: int = 0
    deleted_files: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def _skip(self, reason: str) -> None:
        self.skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def summary(self) -> str:
        lines = [
            f"인덱싱 {self.indexed}개 파일 / 청크 {self.chunks:,}개",
            f"변경없음 {self.unchanged}개 · 건너뜀 {self.skipped}개 · 실패 {self.failed}개",
        ]
        if self.still_failed:
            lines.append(
                f"⚠️ 이전 실패 상태로 남아있음: {self.still_failed}개 "
                f"(재시도하려면 retry_failed=True)"
            )
        if self.deleted_files:
            lines.append(f"드라이브에서 삭제되어 인덱스에서 제거: {self.deleted_files}개")
        if self.skip_reasons:
            lines.append("건너뛴 사유:")
            for reason, n in sorted(self.skip_reasons.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {n:5,}  {reason}")
        if self.errors:
            lines.append(f"실패 상세 (최대 10건):")
            for name, err in self.errors[:10]:
                lines.append(f"  - {name}: {err}")
        return "\n".join(lines)


def _skip_reason(f: DriveFile) -> str | None:
    """인덱싱 대상이 아니면 사유를, 대상이면 None."""
    if f.name.startswith("~$"):
        # MS Office가 파일을 열어둔 동안 만드는 임시/잠금 마커. 내용이 아니라
        # 잠금 정보 몇 바이트뿐이라 zip으로도 못 열린다 (실측: BadZipFile).
        # 다운로드·파싱을 시도할 필요가 없으니 여기서 먼저 걸러 실패 집계에서 뺀다.
        return "Office 임시/잠금 파일 (~$)"
    if f.category in TEXT_CATEGORIES:
        return None
    if f.category in NEEDS_MODEL:
        return f"OCR/ASR 필요 ({f.category})"
    if f.category in NEEDS_CONVERT:
        return f"레거시 포맷 변환 필요 ({f.category})"
    return f"대상 아님 ({f.category})"


def index_files(
    files: list[DriveFile],
    models: Models,
    store: Store,
    conn: sqlite3.Connection,
    force: bool = False,
    retry_failed: bool = False,
    progress: bool = True,
    workers: int = DOWNLOAD_WORKERS,
) -> Result:
    """파일 목록을 인덱싱한다 (증분).

    다운로드는 스레드로 병렬 실행하고, 파싱·임베딩·저장은 순차로 처리한다.
    """
    res = Result()

    if not store.exists():
        store.create(models.dim)

    # ── 1단계: 무엇을 실제로 처리할지 먼저 가른다 (네트워크 접근 없음) ──
    todo: list[DriveFile] = []
    for f in files:
        if (reason := _skip_reason(f)) is not None:
            res._skip(reason)
            manifest.record(conn, f, 0, status="skipped", error=reason)
            continue

        if not force and not manifest.needs_index(conn, f, retry_failed=retry_failed):
            # 왜 건너뛰는지 구분한다. 이전 실패를 '변경없음'으로 세면 문제가 숨는다.
            rec = manifest.get(conn, f.id)
            if rec is not None and rec.status == "failed":
                res.still_failed += 1
                res.errors.append((f.name, f"(이전 실패 유지) {rec.error or ''}"[:200]))
            elif rec is not None and rec.status == "skipped":
                res._skip(rec.error or "이전에 건너뜀")
            else:
                res.unchanged += 1
            continue
        todo.append(f)
    conn.commit()

    if not todo:
        return res

    # ── 2단계: 다운로드를 미리 병렬로 돌리고, 완료된 것부터 처리 ──
    total = len(todo)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fetch")
    futures = {f.id: pool.submit(fetch, f) for f in todo}

    try:
        for i, f in enumerate(todo, start=1):
            if progress and (i % 25 == 0 or i == total):
                print(f"  [{i}/{total}] 인덱싱 {res.indexed} · 청크 {res.chunks:,} · "
                      f"실패 {res.failed}", flush=True)
            _index_one(f, futures[f.id].result, models, store, conn, res)
            conn.commit()
    finally:
        # 남은 다운로드는 취소한다 (중단 시 백그라운드 스레드가 계속 돌지 않게)
        for fut in futures.values():
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    return res


def _index_one(f: DriveFile, get_path, models: Models, store: Store,
               conn: sqlite3.Connection, res: Result) -> None:
    """파일 하나를 파싱·임베딩·저장한다. get_path() 는 다운로드 결과를 준다."""
    try:
        path: Path = get_path()

        # 드라이브 원본이 0바이트인 경우가 실제로 있다 (실측: test.pptx 등) —
        # 파서마다 다른 낯선 예외(PackageNotFoundError 등)로 실패하는 대신
        # 여기서 먼저 걸러 사유를 명확히 남긴다.
        if path.stat().st_size == 0:
            reason = "빈 파일 (0바이트) — 드라이브 원본이 비어있음"
            res._skip(reason)
            manifest.record(conn, f, 0, status="skipped", error=reason)
            return

        # 스캔본 PDF 는 텍스트가 안 나온다 — 파싱 전에 걸러 사유를 남긴다
        if f.category == "pdf" and pdf_text_ratio(path) < MIN_PDF_TEXT_RATIO:
            reason = "스캔본 PDF (텍스트 레이어 없음) — OCR 필요"
            res._skip(reason)
            manifest.record(conn, f, 0, status="skipped", error=reason)
            return

        blocks = parse(path, f.category)
        chunks = chunk_file(f, blocks)
        if not chunks:
            reason = "추출된 텍스트 없음"
            res._skip(reason)
            manifest.record(conn, f, 0, status="skipped", error=reason)
            return

        vectors = models.embed([c.embed_text for c in chunks])
        store.delete_file([f.id])      # 예전 청크 제거 후 새로 넣는다
        store.upsert(chunks, vectors)
        manifest.record(conn, f, len(chunks), status="indexed")
        res.indexed += 1
        res.chunks += len(chunks)

    except SkipFile as e:
        # 정책적 제외 — '실패'와 섞지 않는다 (모듈 SkipFile 설명 참고)
        res._skip(str(e)[:200])
        manifest.record(conn, f, 0, status="skipped", error=str(e)[:500])
    except ParseError as e:
        res.failed += 1
        res.errors.append((f.name, str(e)[:200]))
        manifest.record(conn, f, 0, status="failed", error=str(e)[:500])
    except Exception as e:
        res.failed += 1
        res.errors.append((f.name, f"{type(e).__name__}: {str(e)[:180]}"))
        manifest.record(conn, f, 0, status="failed",
                        error=f"{type(e).__name__}: {str(e)[:480]}")


def prune_deleted(files: list[DriveFile], store: Store,
                  conn: sqlite3.Connection, res: Result | None = None) -> int:
    """드라이브에서 사라진 파일의 청크를 인덱스에서 제거한다.

    ⚠️ files 는 '현재 범위 전체'여야 한다. 일부 폴더만 스캔한 목록을 넘기면
    나머지 폴더의 인덱스를 전부 지워버린다. 그래서 기본 동작이 아니라 별도 함수다.
    """
    live = {f.id for f in files}
    gone = sorted(manifest.all_ids(conn) - live)
    if gone:
        store.delete_file(gone)
        manifest.forget(conn, gone)
        if res is not None:
            res.deleted_files = len(gone)
    return len(gone)
