"""위치 인덱싱 — 내용은 넣지 않고 "어디에 무엇이 있다"만 검색 가능하게 만든다.

## 왜 별도 모드가 필요한가

데이터셋·사진 폴더(이미지 수만 장, 체크포인트, 라벨링 파일)는 본문을 인덱싱해도
얻을 게 없다. 이미지에는 텍스트가 없고, 있어도 문서 검색 결과를 흐린다.
그런데 "소 어노테이션 데이터 어디 있었지?", "IP102는 어느 폴더야?" 같은 질문은
실제로 자주 나온다. 필요한 것은 **내용이 아니라 위치**다.

## 파일마다 청크를 만들지 않는 이유

IP102 는 이미지가 약 7.5만 장이다. 파일당 청크 하나면 7.5만 개가 되어
임베딩 시간과 인덱스를 데이터셋이 지배한다. 그래서 **디렉터리 단위로 요약 청크
하나**를 만든다. 파일 수·용량·확장자 분포·예시 파일명이 들어가므로
"어디에 뭐가 몇 개 있다"는 답이 가능하면서 청크 수는 수십~수백 개로 유지된다.
"""
from __future__ import annotations

import uuid
from collections import Counter, defaultdict

from .chunk import _POINT_NAMESPACE, Chunk
from .drive import DriveFile

# 요약에 넣을 예시 파일명 개수. 너무 많으면 임베딩이 파일명 노이즈에 지배된다.
SAMPLE_NAMES = 8


def _human(n: int) -> str:
    if n <= 0:
        return "0B"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}TB"


def _dir_of(path: str) -> str:
    head, sep, _ = path.rpartition("/")
    return head if sep else ""


def build_listing_chunks(files: list[DriveFile], root_label: str,
                         folder_url: str) -> list[Chunk]:
    """파일 목록 → 디렉터리별 요약 청크.

    folder_url: 이 범위의 최상위 폴더 드라이브 링크. 디렉터리마다 ID를 따로
    조회하지 않는다 (--files-only 목록에는 디렉터리 ID가 없고, 굳이 한 번 더
    긁을 값어치가 없다). 대신 청크 본문에 상대 경로를 적어 찾아갈 수 있게 한다.
    """
    by_dir: dict[str, list[DriveFile]] = defaultdict(list)
    for f in files:
        by_dir[_dir_of(f.path)].append(f)

    chunks: list[Chunk] = []

    # ── 폴더 전체 요약 (한 개) ──
    total_size = sum(f.size for f in files if f.size > 0)
    ext_all = Counter()
    for f in files:
        name = f.name.lower()
        dot = name.rfind(".")
        ext_all[name[dot:] if dot > 0 else "(확장자 없음)"] += 1
    top_dirs = sorted(by_dir.items(), key=lambda kv: -len(kv[1]))[:15]

    overview = [
        f"[드라이브 위치 요약] {root_label}",
        f"이 폴더에는 파일 {len(files):,}개, 총 {_human(total_size)}가 있다. "
        f"하위 디렉터리 {len(by_dir):,}개.",
        "파일 종류: " + ", ".join(f"{e} {c:,}개" for e, c in ext_all.most_common(10)),
        "",
        "파일이 많은 하위 디렉터리:",
    ]
    for d, fs in top_dirs:
        loc = f"{root_label}/{d}" if d else root_label
        overview.append(f"  {loc} — {len(fs):,}개, "
                        f"{_human(sum(x.size for x in fs if x.size > 0))}")
    overview.append("")
    overview.append("※ 이 폴더는 내용(본문)을 인덱싱하지 않았다. "
                    "데이터셋·사진 등이라 위치 정보만 검색 가능하다.")

    chunks.append(_make(root_label, "", "\n".join(overview), folder_url, 0))

    # ── 디렉터리별 요약 ──
    for i, (d, fs) in enumerate(
            sorted(by_dir.items(), key=lambda kv: -len(kv[1])), start=1):
        loc = f"{root_label}/{d}" if d else root_label
        ext = Counter()
        for f in fs:
            name = f.name.lower()
            dot = name.rfind(".")
            ext[name[dot:] if dot > 0 else "(확장자 없음)"] += 1
        size = sum(f.size for f in fs if f.size > 0)
        names = [f.name for f in fs[:SAMPLE_NAMES]]
        rest = len(fs) - len(names)

        body = [
            f"[드라이브 위치] {loc}",
            f"파일 {len(fs):,}개 · {_human(size)}",
            "확장자: " + ", ".join(f"{e} {c:,}개" for e, c in ext.most_common(6)),
            "예시 파일: " + ", ".join(names) + (f" (외 {rest:,}개)" if rest > 0 else ""),
        ]
        chunks.append(_make(root_label, d, "\n".join(body), folder_url, i))

    return chunks


def _make(root_label: str, dirpath: str, text: str, url: str, index: int) -> Chunk:
    """위치 요약 청크 하나.

    file_id 를 'listing:<root>:<dir>' 로 두어 ID 가 결정론적이다 —
    다시 만들어도 같은 포인트를 덮어쓰므로 중복이 생기지 않는다.
    """
    loc = f"{root_label}/{dirpath}" if dirpath else root_label
    c = Chunk(
        text=text,
        file_id=f"listing:{root_label}:{dirpath}",
        file_name=loc,
        file_url=url,
        root=root_label,
        path=dirpath,
        mod_time="",
        index=index,
        label="위치 정보",
    )
    # Chunk.__post_init__ 이 만든 ID 를 덮어쓴다 (index 가 정렬 순서에 따라
    # 흔들릴 수 있으므로 경로만으로 결정론적이게 한다)
    c.id = str(uuid.uuid5(_POINT_NAMESPACE, f"listing:{root_label}:{dirpath}"))
    return c
