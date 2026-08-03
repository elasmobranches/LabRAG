"""rclone를 통한 Google Drive 접근.

rclone를 쓰는 이유:
  - GCP 프로젝트/서비스 계정 없이 내장 OAuth 클라이언트로 바로 붙는다.
  - Google 네이티브 문서(Docs/Sheets/Slides) export를 대신 처리해준다.
  - 공유 드라이브(Team Drive)를 연결 문자열로 투명하게 다룬다.
  - lsjson이 파일 ID를 같이 주므로, 나중에 Drive API로 갈아타도 인덱스를 버릴 필요가 없다.

권한은 drive.readonly로 고정 — 이 코드에는 쓰기 경로가 없다.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Iterator

from .config import settings
from .filetypes import GOOGLE_EXPORT_MIME, categorize


class RcloneError(RuntimeError):
    pass


class ListTimeout(RcloneError):
    """폴더 목록 조회가 제한 시간을 넘겼다.

    호출자가 이 폴더를 하위 폴더로 쪼개 재시도할 수 있게 별도 예외로 둔다
    (index.py 의 적응적 분할). 실제로 학회 폴더 깊숙이 EnergyPlus 시뮬레이션
    출력 폴더가 있어서 목록 조회만 15분이 넘게 걸렸다.
    """


def _clean_stderr(stderr: str, keep: int = 800) -> str:
    """rclone stderr 에서 실제 오류만 남긴다.

    rclone 은 매 실행마다 NOTICE 를 뱉는다 (공용 client_id 폐기 예고 등).
    이게 stderr 첫 줄이라, 오류 메시지를 잘라서 보고하면 정작 원인은 잘려나가고
    NOTICE 만 보인다. 실제로 이 때문에 목록 조회 실패 원인을 못 볼 뻔했다.
    """
    lines = [ln for ln in stderr.splitlines()
             if ln.strip() and "NOTICE" not in ln and "INFO  " not in ln]
    out = "\n".join(lines).strip() or stderr.strip()
    return out[:keep]


def _run(args: list[str], timeout: int = 1800) -> str:
    cmd = [str(settings.rclone), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RcloneError(
            f"rclone 실패 (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"{_clean_stderr(proc.stderr)}"
        )
    return proc.stdout


def check_remote() -> bool:
    """리모트가 설정돼 있는지 확인."""
    out = _run(["listremotes"])
    return f"{settings.remote}:" in out.split()


@dataclass(frozen=True)
class SharedDrive:
    id: str
    name: str

    @property
    def spec(self) -> str:
        """이 공유 드라이브를 가리키는 rclone 경로."""
        return f"{settings.remote},team_drive={self.id}:"


def list_shared_drives() -> list[SharedDrive]:
    """계정이 볼 수 있는 공유 드라이브 목록."""
    out = _run(["backend", "drives", f"{settings.remote}:"])
    return [SharedDrive(id=d["id"], name=d["name"]) for d in json.loads(out or "[]")]


@dataclass
class DriveFile:
    """드라이브의 파일 하나. 스캔·인덱싱 양쪽에서 쓰는 공통 표현."""

    id: str
    path: str          # 루트 기준 상대 경로
    name: str
    size: int          # Google 네이티브는 -1 (용량 미정)
    mime_type: str
    mod_time: str      # RFC3339
    root: str          # 어느 드라이브에서 왔는지 (예: "My Drive", "연구실공유")
    category: str = field(init=False)

    def __post_init__(self) -> None:
        self.category = categorize(self.name, self.mime_type)

    @property
    def is_google_native(self) -> bool:
        """Google Docs/Sheets/Slides 원본인지.

        rclone은 이미 Office 포맷으로 변환해 보고하므로 MIME만으로는 구별할 수 없다.
        export 전이라 용량을 모르는 것(Size == -1)이 확실한 신호다.
        """
        return self.size < 0 and self.mime_type in GOOGLE_EXPORT_MIME

    @property
    def web_url(self) -> str:
        """인용에 쓸 드라이브 웹 링크.

        Google 네이티브는 docs.google.com 편집 링크가 맞다.
        drive.google.com/file/d/... 은 네이티브 문서에서 동작하지 않는다.

        바로가기(shortcut)는 self.id 가 탭으로 이어붙은 두 ID 문자열이다
        (fetch.py의 candidate_ids 참고 — 다운로드는 이미 처리돼 있었는데, 인용
        링크는 원본 id를 그대로 써서 URL 안에 탭 문자가 섞여 깨진 링크가 나갔다).
        링크에는 첫 번째(실측상 실제 대상) ID만 쓴다.
        """
        primary_id = self.id.split()[0] if self.id.split() else self.id
        if self.is_google_native:
            kind = GOOGLE_EXPORT_MIME[self.mime_type]
            return f"https://docs.google.com/{kind}/d/{primary_id}/edit"
        return f"https://drive.google.com/file/d/{primary_id}/view"

    @property
    def top_folder(self) -> str:
        """최상위 폴더 이름 (인덱싱 범위를 폴더 단위로 고를 때 쓴다)."""
        head, sep, _ = self.path.partition("/")
        return head if sep else "(루트 직속 파일)"

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        return d


def _lsjson_stream(spec: str, subpath: str = "", flags: list[str] | None = None,
                   depth: int | None = None) -> Iterator[dict]:
    """lsjson을 스트리밍으로 파싱.

    lsjson은 전체를 하나의 JSON 배열로 뱉는다. 파일이 수만 개면 커지므로
    한 줄씩 읽어 항목 단위로 넘긴다 (rclone은 항목당 한 줄로 출력한다).
    """
    target = spec + subpath if subpath else spec
    cmd = [
        str(settings.rclone), "lsjson",
        "-R",                    # 재귀
        "--files-only",
        "--no-mimetype=false",   # MIME 타입 포함 (기본이지만 명시)
        # --fast-list 없으면 디렉터리마다 API 호출을 하나씩 해서 극단적으로 느려진다.
        # 실측 ([Workspace]/논문, 284개 파일): 36.7초 → 3.5초 (10.4배). 결과는 동일.
        # 메모리를 더 쓰지만 (전체 목록을 한 번에 들고 트리를 재구성) 문제될 규모가 아니다.
        "--fast-list",
        *(["--max-depth", str(depth)] if depth else []),
        *(flags or []),
        target,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]"):
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read() if proc.stderr else ""
        raise RcloneError(
            f"rclone lsjson 실패 (exit {proc.returncode}): {_clean_stderr(stderr)}")


@dataclass(frozen=True)
class Root:
    """스캔·인덱싱의 단위가 되는 드라이브 루트.

    "나와 공유됨"은 별도 드라이브가 아니라 같은 리모트에 플래그로 접근한다.
    공유 드라이브(Team Drive)가 없는 연구실에서는 자료가 대부분 여기에 있다.
    """

    label: str
    spec: str
    flags: tuple[str, ...] = ()
    # 탐색 깊이 제한. depth=1 이면 그 폴더에 직접 놓인 파일만 본다.
    # 무거운 하위 폴더가 있는 폴더의 '루트 직속 파일'만 주워올 때 쓴다 —
    # 하위 폴더 단위로 범위를 쪼개면 루트에 바로 있는 파일이 누락되기 때문이다.
    depth: int | None = None
    # "content": 본문을 파싱해 인덱싱 (기본)
    # "listing": 내용은 넣지 않고 "어디에 무엇이 있다"만 인덱싱 (labrag/listing.py)
    mode: str = "content"
    folder_id: str = ""       # listing 모드에서 드라이브 링크를 만들 때 쓴다
    # True 면 하위 폴더 각각을 독립 루트로 펼친다 (expand_roots 참고).
    # 큰 폴더를 손으로 쪼개는 대신 쓴다 — 새 구성원/과제 폴더가 생겨도 자동으로 잡힌다.
    split: bool = False


def default_roots() -> list[Root]:
    """계정이 접근 가능한 모든 루트: My Drive + 나와 공유됨 + 공유 드라이브들."""
    roots = [
        Root("My Drive", f"{settings.remote}:"),
        Root("Shared with me", f"{settings.remote}:", ("--drive-shared-with-me",)),
    ]
    try:
        roots += [Root(d.name, d.spec) for d in list_shared_drives()]
    except RcloneError:
        pass
    return roots


def normalize_id(raw: str) -> str:
    """드라이브 바로가기는 ID 가 탭으로 두 개 붙어서 온다 — 첫 번째가 대상이다.

    파일만의 문제가 아니다. 폴더 바로가기도 같은 형태로 오고, 그대로
    --drive-root-folder-id 에 넘기면 동작이 불확실해진다.
    (파일 쪽 대응은 fetch.candidate_ids 참고 — 거기서는 순서를 가정하지 않고
    차례로 시도하지만, 폴더는 재시도 비용이 크므로 첫 번째를 쓴다.)
    """
    parts = raw.split()
    return parts[0] if parts else raw


def list_subdirs(root: Root) -> list[tuple[str, str]]:
    """이 루트의 바로 아래 하위 폴더들 [(id, name)]. API 호출 한 번."""
    out = _run([
        "lsjson", "--dirs-only", "--max-depth", "1", *root.flags, root.spec,
    ])
    items = json.loads(out or "[]")
    return [(normalize_id(d["ID"]), d["Name"]) for d in items if d.get("ID")]


def expand_roots(roots: list[Root]) -> list[Root]:
    """split=True 인 루트를 하위 폴더 단위로 펼친다.

    왜 필요한가: 연구실 드라이브에는 구성원별 미팅 폴더나 과제별 폴더처럼
    하위 폴더가 많고 각각이 큰 곳이 있다. 한 루트로 목록을 조회하면
    (실측) 15분에 14,751개까지 나열하고도 끝나지 않는다. 그러면 그 폴더 하나가
    나머지 전부를 막는다.

    손으로 하위 폴더 ID를 나열하는 것보다 이 방식이 나은 이유:
      - 새 구성원이 들어오거나 새 과제 폴더가 생겨도 자동으로 잡힌다.
      - depth-1 조회는 API 호출 한 번이라 사실상 공짜다.
      - 폴더별로 실패가 격리되고 진행 상황이 보인다.
      - `--only 이연구` 처럼 개별 실행이 가능해진다.

    펼칠 때 **부모 자신도 depth=1 로 넣는다** — 그러지 않으면 부모에 직접 놓인
    파일이 누락된다 (실제로 eval 셋이 이 누락을 잡아냈다).
    """
    out: list[Root] = []
    for r in roots:
        if not r.split:
            out.append(r)
            continue
        # 부모 직속 파일
        out.append(Root(label=f"{r.label} (직속 파일)", spec=r.spec, flags=r.flags,
                        depth=1, mode=r.mode, folder_id=r.folder_id))
        try:
            subs = list_subdirs(r)
        except RcloneError as e:
            print(f"[!] {r.label} 하위 폴더 조회 실패 — 통째로 처리한다: {e}")
            out.append(Root(label=r.label, spec=r.spec, flags=r.flags,
                            mode=r.mode, folder_id=r.folder_id))
            continue
        for fid, name in sorted(subs, key=lambda x: x[1]):
            out.append(Root(label=f"{r.label}/{name}", spec=r.spec,
                            flags=("--drive-root-folder-id", fid),
                            mode=r.mode, folder_id=fid))
    return out


# 폴더 하나의 목록 조회 제한 시간(초). 넘기면 하위 폴더로 쪼개 재시도한다.
# --fast-list 는 트리를 다 모은 뒤에야 출력하므로, 중간 진행을 볼 수 없다.
# 그래서 스트리밍 대신 통째로 받으며 타임아웃을 건다.
#
# 90초로 잡은 근거: 정상적인 문서 폴더는 실측 0.7~9초에 끝난다 (수십~수백 개 파일).
# 90초를 넘긴다는 것은 하위 트리가 비정상적으로 크다는 신호이므로, 빨리 포기하고
# 쪼개는 편이 총 소요 시간이 짧다. 180초로 뒀을 때는 탐색 자체가 오래 걸렸다.
LIST_TIMEOUT = 90


def list_files(root: Root, timeout: int = LIST_TIMEOUT) -> list[DriveFile]:
    """루트 하나의 파일 목록. 제한 시간을 넘기면 ListTimeout 을 던진다."""
    cmd = [
        str(settings.rclone), "lsjson", "-R", "--files-only",
        "--no-mimetype=false", "--fast-list",
        *(["--max-depth", str(root.depth)] if root.depth else []),
        *root.flags, root.spec,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ListTimeout(
            f"목록 조회 {timeout}초 초과 ({root.label}) — 하위 폴더가 매우 많은 것으로 보임"
        ) from None
    if proc.returncode != 0:
        raise RcloneError(
            f"목록 조회 실패 (exit {proc.returncode}, {root.label}): "
            f"{_clean_stderr(proc.stderr)}")
    return [
        DriveFile(
            id=item.get("ID", ""), path=item.get("Path", ""), name=item.get("Name", ""),
            size=int(item.get("Size", -1)), mime_type=item.get("MimeType", ""),
            mod_time=item.get("ModTime", ""), root=root.label,
        )
        for item in json.loads(proc.stdout or "[]")
    ]


def walk(spec: str, root_label: str, subpath: str = "",
         flags: tuple[str, ...] | list[str] = (),
         depth: int | None = None) -> Iterator[DriveFile]:
    """드라이브 하나를 재귀 순회 (스트리밍). 타임아웃이 필요하면 list_files 를 쓸 것."""
    for item in _lsjson_stream(spec, subpath, list(flags), depth):
        yield DriveFile(
            id=item.get("ID", ""),
            path=item.get("Path", ""),
            name=item.get("Name", ""),
            size=int(item.get("Size", -1)),
            mime_type=item.get("MimeType", ""),
            mod_time=item.get("ModTime", ""),
            root=root_label,
        )
