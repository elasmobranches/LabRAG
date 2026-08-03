"""인덱싱 범위 정책 (config/scope.json).

"드라이브 전체"를 인덱싱하지 않는 이유가 두 가지 있다.

1. 성능 — 학습용 데이터셋 폴더(이미지 수만 장)를 훑으면 스캔만 수십 분 걸리고,
   정작 문서 검색에는 쓸모가 없다.
2. 프라이버시 — 개인 문서(증명서, 지원서류)가 같은 계정에 섞여 있다.
   연구실 공용 RAG에 들어가면 의도치 않게 공유된다.

그래서 "무엇을 검색 가능하게 만들지"를 명시적인 파일로 관리한다.
폴더는 ID로 지정한다 — 이름이 바뀌어도 정책이 깨지지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .drive import Root

SCOPE_PATH = settings.root / "config" / "scope.json"


@dataclass
class Scope:
    include: list[dict]
    exclude: list[dict]
    my_drive_mode: str
    my_drive_folders: list[dict]

    @property
    def excluded_ids(self) -> set[str]:
        return {e["id"] for e in self.exclude}

    def roots(self) -> list[Root]:
        """스캔·인덱싱할 루트 목록.

        폴더 ID를 --drive-root-folder-id 로 지정하면 그 폴더가 루트가 된다.
        My Drive든 '나와 공유됨'이든 구분이 필요 없어진다 — ID는 전역이다.
        """
        out = [
            Root(label=e["name"], spec=f"{settings.remote}:",
                 flags=("--drive-root-folder-id", e["id"]),
                 depth=e.get("depth"),
                 mode=e.get("mode", "content"),
                 folder_id=e["id"],
                 split=bool(e.get("split")))
            for e in self.include
        ]
        if self.my_drive_mode == "all":
            out.append(Root("My Drive", f"{settings.remote}:"))
        elif self.my_drive_mode == "folders":
            out += [
                Root(label=f"My Drive/{e['name']}", spec=f"{settings.remote}:",
                     flags=("--drive-root-folder-id", e["id"]))
                for e in self.my_drive_folders
            ]
        return out


def load(path: Path | None = None) -> Scope:
    data = json.loads((path or SCOPE_PATH).read_text(encoding="utf-8"))
    md = data.get("my_drive", {})
    scope = Scope(
        include=[e for e in data.get("include", []) if not e.get("disabled")],
        exclude=data.get("exclude", []),
        my_drive_mode=md.get("mode", "off"),
        my_drive_folders=md.get("folders", []),
    )
    overlap = {e["id"] for e in scope.include} & scope.excluded_ids
    if overlap:
        raise ValueError(f"include와 exclude에 같은 폴더가 있음: {overlap}")
    return scope
