"""증분 인덱싱 상태 추적 (SQLite).

드라이브는 계속 바뀐다. 매번 전체를 재인덱싱하면 시간과 GPU가 낭비되므로
"이 파일을 어느 시점 버전으로, 청크 몇 개로 인덱싱했는지"를 기록해둔다.

파일 ID를 키로 쓰는 이유: 드라이브에서 파일을 다른 폴더로 옮기거나 이름을
바꿔도 ID는 유지된다. 경로를 키로 쓰면 폴더 정리 한 번에 전체가 재인덱싱된다.

상태 판정:
  - manifest에 없음                     → 신규 (인덱싱)
  - mod_time 다름 또는 size 다름         → 변경 (재인덱싱)
  - manifest에 있는데 드라이브에 없음    → 삭제 (청크 제거)
  - status='failed'                     → 파싱 실패. 재시도 여부를 따로 판단
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    root        TEXT NOT NULL,
    category    TEXT NOT NULL,
    size        INTEGER NOT NULL,
    mod_time    TEXT NOT NULL,
    n_chunks    INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL,            -- indexed | failed | skipped
    error       TEXT,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_root   ON files(root);

-- 위치 인덱싱(mode=listing) 상태.
-- 데이터셋 폴더는 목록 조회만 수십 분이 걸린다 (IP102 는 이미지 7.5만 장).
-- 매 실행마다 다시 훑으면 증분 동기화를 자동으로 돌릴 수 없으므로,
-- 한 번 만들어두고 --refresh-listing 을 줄 때만 갱신한다.
CREATE TABLE IF NOT EXISTS listings (
    root        TEXT PRIMARY KEY,
    folder_id   TEXT NOT NULL,
    n_files     INTEGER NOT NULL,
    n_chunks    INTEGER NOT NULL,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass
class Record:
    file_id: str
    name: str
    path: str
    root: str
    category: str
    size: int
    mod_time: str
    n_chunks: int
    status: str
    error: str | None


@contextmanager
def connect(db: Path | None = None) -> Iterator[sqlite3.Connection]:
    path = db or settings.manifest_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def get(conn: sqlite3.Connection, file_id: str) -> Record | None:
    row = conn.execute(
        "SELECT file_id,name,path,root,category,size,mod_time,n_chunks,status,error "
        "FROM files WHERE file_id=?", (file_id,)
    ).fetchone()
    return Record(*row) if row else None


def all_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT file_id FROM files")}


def needs_index(conn: sqlite3.Connection, f, retry_failed: bool = False) -> bool:
    """이 파일을 (재)인덱싱해야 하는가.

    Google 네이티브 문서는 size가 항상 -1이라 크기 비교가 무의미하다.
    그래서 mod_time을 1차 신호로 쓰고, size는 보조로만 본다.
    """
    rec = get(conn, f.id)
    if rec is None:
        return True
    if rec.status == "failed":
        return retry_failed
    if rec.status == "skipped":
        return False
    if rec.mod_time != f.mod_time:
        return True
    if f.size >= 0 and rec.size != f.size:
        return True
    return False


def record(conn: sqlite3.Connection, f, n_chunks: int,
           status: str = "indexed", error: str | None = None) -> None:
    conn.execute(
        "INSERT INTO files (file_id,name,path,root,category,size,mod_time,"
        "n_chunks,status,error,indexed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now')) "
        "ON CONFLICT(file_id) DO UPDATE SET "
        "name=excluded.name, path=excluded.path, root=excluded.root, "
        "category=excluded.category, size=excluded.size, mod_time=excluded.mod_time, "
        "n_chunks=excluded.n_chunks, status=excluded.status, error=excluded.error, "
        "indexed_at=excluded.indexed_at",
        (f.id, f.name, f.path, f.root, f.category, f.size, f.mod_time,
         n_chunks, status, error),
    )


def forget(conn: sqlite3.Connection, file_ids: list[str]) -> None:
    conn.executemany("DELETE FROM files WHERE file_id=?", [(i,) for i in file_ids])


def has_listing(conn: sqlite3.Connection, root: str) -> bool:
    return conn.execute("SELECT 1 FROM listings WHERE root=?", (root,)).fetchone() is not None


def record_listing(conn: sqlite3.Connection, root: str, folder_id: str,
                   n_files: int, n_chunks: int, total_bytes: int) -> None:
    conn.execute(
        "INSERT INTO listings (root,folder_id,n_files,n_chunks,total_bytes,indexed_at) "
        "VALUES (?,?,?,?,?,datetime('now')) "
        "ON CONFLICT(root) DO UPDATE SET folder_id=excluded.folder_id, "
        "n_files=excluded.n_files, n_chunks=excluded.n_chunks, "
        "total_bytes=excluded.total_bytes, indexed_at=excluded.indexed_at",
        (root, folder_id, n_files, n_chunks, total_bytes),
    )


def listing_stats(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT root,n_files,n_chunks,total_bytes,indexed_at FROM listings "
        "ORDER BY n_files DESC")]


def forget_listing(conn: sqlite3.Connection, roots: list[str]) -> None:
    conn.executemany("DELETE FROM listings WHERE root=?", [(r,) for r in roots])


def stats(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT status, COUNT(*) n, COALESCE(SUM(n_chunks),0) chunks "
        "FROM files GROUP BY status"
    ).fetchall()
    return {r["status"]: {"files": r["n"], "chunks": r["chunks"]} for r in rows}
