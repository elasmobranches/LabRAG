"""드라이브에서 파일을 로컬 캐시로 가져온다.

경로 대신 **파일 ID**로 받는다 (`rclone backend copyid`):
  - 파일 이름에 특수문자(전각 슬래시 등)가 있어도 경로 이스케이프 문제가 없다.
  - "나와 공유됨"/My Drive/공유 드라이브 구분이 필요 없다 — ID는 전역이다.
  - 파일이 다른 폴더로 옮겨져도 ID는 유지되므로 증분 인덱싱이 깨지지 않는다.

Google 네이티브 문서는 rclone이 export 포맷으로 변환해서 내려준다.
"""
from __future__ import annotations

import subprocess
import unicodedata
from pathlib import Path

from .config import settings
from .drive import DriveFile, RcloneError, _clean_stderr
from .parse import SkipFile

# 목록(lsjson)과 내려받기(copyid)가 같은 변환 규칙을 쓰도록 명시한다.
# rclone 기본값과 동일하지만, 설정이 바뀌어도 파일명과 실제 내용이 어긋나지 않게 고정.
EXPORT_FORMATS = "docx,xlsx,pptx"

# 리눅스 NAME_MAX(255바이트)에 여유를 둔 상한. rclone이 돌려주는 한글 파일명이 NFD
# (자모 분해)인 경우가 있는데, 같은 글자라도 NFD는 UTF-8로 NFC보다 훨씬 길다
# (완성형 한 글자 = NFC 3바이트, NFD는 자모마다 나뉘어 6~9바이트). 실측: 90자 안팎
# 한글 파일명이 NFD로는 400바이트를 넘겨 os.stat()부터 ENAMETOOLONG으로 실패했다
# ("(식량정책관-농업기반과) 8월 14일자 매일경제 보도 ...pdf" 등 2건).
_MAX_NAME_BYTES = 200


_MAX_EXT_BYTES = 20  # 확장자 자체가 비정상적으로 긴 경우에도 상한을 보장한다


def _safe_local_name(name: str) -> str:
    """로컬 캐시 파일명으로 안전한 형태를 만든다 (드라이브상의 실제 이름은 그대로
    manifest/Qdrant의 name 필드에 남으므로 인용 표시에는 영향이 없다 — 여기서
    바뀌는 건 파일시스템에 실제로 쓰는 파일명뿐이다).

    1. NFC로 정규화해 NFD보다 짧게 만든다.
    2. 경로 구분자·`.`/`..`를 캐시 경로 밖으로 못 새어나가게 막는다 — 정규 Drive UI로는
       만들 수 없는 이름이지만(전각 슬래시 ／ 는 되지만 진짜 ASCII / 는 안 된다),
       API 응답 이상값까지 방어한다.
    3. 그래도 길면 확장자·본체 각각에 상한을 두고 자른다.
    """
    nfc = unicodedata.normalize("NFC", name)
    nfc = nfc.replace("/", "／").replace("\\", "＼")
    if nfc in (".", ".."):
        nfc = "_" + nfc
    if not nfc:
        nfc = "_"
    if len(nfc.encode("utf-8")) <= _MAX_NAME_BYTES:
        return nfc
    stem, dot, ext = nfc.rpartition(".")
    if not dot:
        stem, ext = nfc, ""
    while ext and len(ext.encode("utf-8")) > _MAX_EXT_BYTES:
        ext = ext[:-1]
    budget = _MAX_NAME_BYTES - len((dot + ext).encode("utf-8"))
    while stem and len(stem.encode("utf-8")) > budget:
        stem = stem[:-1]
    result = stem + dot + ext
    return result if result else "_"


def cache_path(f: DriveFile) -> Path:
    """이 파일의 로컬 캐시 경로.

    ID로 디렉터리를 나눠 이름 충돌을 없애고, 원래 파일명은 그대로 보존해
    파서가 확장자를 보고 판단할 수 있게 한다.
    (Google 네이티브도 rclone이 이미 확장자를 붙여서 주므로 그대로 쓴다.)
    """
    # ID 앞 2글자로 샤딩 — 한 디렉터리에 수만 개가 쌓이는 것을 막는다
    return settings.raw_dir / f.id[:2] / f.id / _safe_local_name(f.name)


def candidate_ids(file_id: str) -> list[str]:
    """드라이브 바로가기(shortcut)는 ID 가 두 개 붙어서 온다.

    rclone lsjson 이 바로가기 항목의 ID 를 탭으로 이어붙여 준다:
        '15zYlnW8yN6E1RvIBf7ceeSjeV5oCIGbJNJB2VqkOjvU\\t1fkrF37chrvVaLRsCE28DagPtY190a_OG'
    이걸 그대로 copyid 에 넘기면 실패한다. 실측에서는 앞쪽이 실제 대상이었지만,
    순서를 가정하지 않고 차례로 시도한다 (바로가기에서만 최대 2회).
    """
    parts = [p for p in file_id.split() if p]
    return parts or [file_id]


def fetch(f: DriveFile, force: bool = False, timeout: int = 600) -> Path:
    """파일 하나를 캐시로 내려받고 로컬 경로를 돌려준다."""
    dest = cache_path(f)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    for fid in candidate_ids(f.id):
        proc = subprocess.run(
            [str(settings.rclone), "backend", "copyid",
             "--drive-export-formats", EXPORT_FORMATS,
             f"{settings.remote}:", fid, str(dest)],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode == 0:
            break
        err = _clean_stderr(proc.stderr, keep=300)
        # 대상이 삭제된 바로가기 — 받을 것이 없다. 고칠 문제가 아니므로 실패가 아니라
        # 건너뜀으로 분류한다 (그러지 않으면 '실패' 집계가 실제 문제를 가린다).
        if "Dangling shortcut" in proc.stderr:
            raise SkipFile(f"끊어진 바로가기 (대상이 삭제됨): {f.name}")
        errors.append(f"[{fid[:12]}…] {err}")
    else:
        raise RcloneError(
            f"파일 받기 실패 (name={f.name}): " + " | ".join(errors)
        )

    if dest.exists():
        return dest

    # rclone이 export 확장자를 스스로 붙였을 수 있다 — 같은 디렉터리에서 찾는다.
    candidates = [p for p in dest.parent.iterdir() if p.is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # 이름이 가장 비슷한 것을 고른다
        return max(candidates, key=lambda p: len(set(p.stem) & set(dest.stem)))
    raise RcloneError(f"받았다고 나왔는데 파일이 없음 (id={f.id}, name={f.name})")
