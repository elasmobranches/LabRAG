"""Drive 후보가 허용된 루트의 후손인지 검증하고 경로를 캐시한다."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .drive_live import DriveItem, DriveLiveClient

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drive_folders (
    folder_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_ids_json TEXT NOT NULL,
    root_id TEXT NOT NULL,
    inside_root INTEGER NOT NULL,
    verified_path TEXT NOT NULL,
    verified_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drive_folders_root
ON drive_folders(root_id, verified_at);
"""


@dataclass(frozen=True)
class VerifiedLocation:
    inside_root: bool
    path: str
    folder_ids: tuple[str, ...]


class AncestryVerifier:
    def __init__(self, client: DriveLiveClient, *, root_id: str,
                 db_path: Path, root_name: str = "ResearchWorkspace",
                 ttl_hours: float = 24.0):
        self.client = client
        self.root_id = root_id
        self.root_name = root_name
        self.db_path = db_path
        self.ttl = timedelta(hours=ttl_hours)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _cached(self, folder_id: str) -> VerifiedLocation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT inside_root,verified_path,verified_at FROM drive_folders "
                "WHERE folder_id=? AND root_id=?", (folder_id, self.root_id)
            ).fetchone()
        if not row:
            return None
        try:
            checked = datetime.fromisoformat(row["verified_at"])
            if checked.tzinfo is None:
                checked = checked.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        if datetime.now(timezone.utc) - checked > self.ttl:
            return None
        return VerifiedLocation(bool(row["inside_root"]), row["verified_path"],
                                (folder_id,))

    def _store(self, folder: DriveItem, result: VerifiedLocation) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO drive_folders "
                "(folder_id,name,parent_ids_json,root_id,inside_root,verified_path,verified_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(folder_id) DO UPDATE SET "
                "name=excluded.name,parent_ids_json=excluded.parent_ids_json,"
                "root_id=excluded.root_id,inside_root=excluded.inside_root,"
                "verified_path=excluded.verified_path,verified_at=excluded.verified_at",
                (folder.id, folder.name, json.dumps(folder.parents), self.root_id,
                 int(result.inside_root), result.path,
                 datetime.now(timezone.utc).isoformat()),
            )

    async def _verify_folder(self, folder_id: str, *, deadline: float,
                             visited: frozenset[str]) -> VerifiedLocation:
        if folder_id == self.root_id:
            return VerifiedLocation(True, self.root_name, (self.root_id,))
        if folder_id in visited:
            return VerifiedLocation(False, "", ())
        cached = self._cached(folder_id)
        if cached is not None:
            return cached
        try:
            folder = await self.client.get_item(folder_id, deadline=deadline)
        except Exception:
            return VerifiedLocation(False, "", ())
        if folder.trashed or not folder.parents:
            result = VerifiedLocation(False, "", ())
            self._store(folder, result)
            return result
        next_visited = visited | {folder_id}
        for parent_id in folder.parents:
            parent = await self._verify_folder(
                parent_id, deadline=deadline, visited=next_visited
            )
            if parent.inside_root:
                result = VerifiedLocation(
                    True, f"{parent.path}/{folder.name}",
                    parent.folder_ids + (folder.id,),
                )
                self._store(folder, result)
                return result
        result = VerifiedLocation(False, "", ())
        self._store(folder, result)
        return result

    async def verify(self, item: DriveItem, *, deadline: float) -> VerifiedLocation:
        if item.trashed:
            return VerifiedLocation(False, "", ())
        if item.id == self.root_id:
            return VerifiedLocation(True, self.root_name, (self.root_id,))
        for parent_id in item.parents:
            parent = await self._verify_folder(parent_id, deadline=deadline,
                                               visited=frozenset({item.id}))
            if parent.inside_root:
                return VerifiedLocation(
                    True, f"{parent.path}/{item.name}", parent.folder_ids
                )
        return VerifiedLocation(False, "", ())
