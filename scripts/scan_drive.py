#!/usr/bin/env python
"""연구실 드라이브 코퍼스 스캔.

파일을 내려받지 않고 목록만 훑어서 "무엇이 얼마나 있는지"를 파악한다.
인덱싱 범위·청킹 전략·OCR 필요 여부를 여기 숫자로 결정한다.

    python scripts/scan_drive.py                 # My Drive + 모든 공유 드라이브
    python scripts/scan_drive.py --list-drives   # 공유 드라이브 목록만
    python scripts/scan_drive.py --only "My Drive" --subpath "논문"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labrag.config import settings
from labrag.drive import DriveFile, check_remote, default_roots, list_shared_drives, walk
from labrag.filetypes import NEEDS_CONVERT, NEEDS_MODEL, TEXT_CATEGORIES


def human(n: int) -> str:
    if n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"


def table(rows: list[tuple], headers: tuple[str, ...]) -> str:
    all_rows = [headers, *[tuple(str(c) for c in r) for r in rows]]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
    out = [
        "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for r in all_rows[1:]:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def report(files: list[DriveFile]) -> None:
    total_size = sum(f.size for f in files if f.size > 0)
    print(f"\n{'=' * 72}")
    print(f"총 {len(files):,}개 파일 · {human(total_size)}")
    print("=" * 72)

    # --- 드라이브별 ---
    by_root: dict[str, list[DriveFile]] = defaultdict(list)
    for f in files:
        by_root[f.root].append(f)
    print("\n[드라이브별]")
    print(table(
        [(r, f"{len(fs):,}", human(sum(x.size for x in fs if x.size > 0)))
         for r, fs in sorted(by_root.items(), key=lambda kv: -len(kv[1]))],
        ("드라이브", "파일수", "용량"),
    ))

    # --- 카테고리별 ---
    by_cat: dict[str, list[DriveFile]] = defaultdict(list)
    for f in files:
        by_cat[f.category].append(f)

    def bucket(cat: str) -> str:
        if cat in TEXT_CATEGORIES:
            return "① 바로 인덱싱"
        if cat in NEEDS_CONVERT:
            return "② 변환 필요"
        if cat in NEEDS_MODEL:
            return "③ OCR/ASR 필요"
        return "④ 대상 아님"

    print("\n[카테고리별]")
    print(table(
        [(bucket(c), c, f"{len(fs):,}", human(sum(x.size for x in fs if x.size > 0)))
         for c, fs in sorted(by_cat.items(), key=lambda kv: (bucket(kv[0]), -len(kv[1])))],
        ("처리 단계", "카테고리", "파일수", "용량"),
    ))

    # --- 인덱싱 대상 요약 ---
    tier1 = [f for f in files if f.category in TEXT_CATEGORIES]
    tier2 = [f for f in files if f.category in NEEDS_CONVERT]
    tier3 = [f for f in files if f.category in NEEDS_MODEL]
    print(f"\n① 바로 인덱싱 가능: {len(tier1):,}개 ({human(sum(f.size for f in tier1 if f.size > 0))})")
    print(f"② 포맷 변환 후 가능: {len(tier2):,}개")
    print(f"③ OCR/ASR 필요:     {len(tier3):,}개")

    # --- 최상위 폴더별 (인덱싱 범위 고르기용) ---
    folder_stats: dict[tuple[str, str], list[DriveFile]] = defaultdict(list)
    for f in tier1:
        folder_stats[(f.root, f.top_folder)].append(f)
    print("\n[① 대상이 많은 최상위 폴더 상위 25개]")
    print(table(
        [(r, fol, f"{len(fs):,}", human(sum(x.size for x in fs if x.size > 0)))
         for (r, fol), fs in sorted(folder_stats.items(), key=lambda kv: -len(kv[1]))[:25]],
        ("드라이브", "최상위 폴더", "①파일수", "용량"),
    ))

    # --- 확장자 상위 ---
    exts = Counter()
    for f in files:
        suffix = Path(f.name).suffix.lower()
        exts[suffix or "(확장자 없음)"] += 1
    print("\n[확장자 상위 25개]")
    print(table([(e, f"{c:,}") for e, c in exts.most_common(25)], ("확장자", "개수")))

    # --- 인덱싱 비용 추정 ---
    # PDF는 페이지당 ~2.5청크, 문서류는 파일당 ~15청크로 거칠게 잡는다.
    pdf_bytes = sum(f.size for f in tier1 if f.category == "pdf" and f.size > 0)
    est_pdf_pages = pdf_bytes / (120 * 1024)          # 논문 PDF 페이지 평균 ~120KB
    est_chunks = int(est_pdf_pages * 2.5 + (len(tier1) - sum(1 for f in tier1 if f.category == "pdf")) * 15)
    print(f"\n[거친 추정] 청크 약 {est_chunks:,}개")
    print(f"  임베딩 소요: 약 {est_chunks / 250 / 60:.0f}분 (TEI 250청크/s 가정)")
    print(f"  Qdrant 저장: 약 {human(int(est_chunks * (2560 * 4 + 1200)))} (2560차원 float32 + 본문)")
    print("  ※ 실제 값은 첫 폴더 인덱싱 후 실측으로 대체할 것\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-drives", action="store_true", help="공유 드라이브 목록만 출력")
    ap.add_argument("--only", action="append", default=None,
                    help="이 드라이브만 스캔 (여러 번 지정 가능). 'My Drive' 또는 공유 드라이브 이름")
    ap.add_argument("--subpath", default="", help="드라이브 내 하위 경로로 제한")
    ap.add_argument("--out", default=None, help="파일 목록 JSONL 저장 경로")
    args = ap.parse_args()

    if not check_remote():
        print(f"[!] rclone 리모트 '{settings.remote}'가 없어. 먼저 인증해:", file=sys.stderr)
        print(f"    {settings.rclone} config create {settings.remote} drive scope=drive.readonly",
              file=sys.stderr)
        return 1

    shared = list_shared_drives()
    if args.list_drives:
        print(f"공유 드라이브 {len(shared)}개:")
        for d in shared:
            print(f"  - {d.name}  (id={d.id})")
        print("\n그리고 개인 My Drive 하나.")
        return 0

    roots = default_roots()
    if args.only:
        wanted = set(args.only)
        roots = [r for r in roots if r.label in wanted]
        if not roots:
            print(f"[!] --only 로 준 이름과 맞는 루트가 없어. 사용 가능: "
                  f"{[r.label for r in default_roots()]}", file=sys.stderr)
            return 1

    files: list[DriveFile] = []
    for root in roots:
        print(f"[*] 스캔 중: {root.label} ...", end="", flush=True)
        try:
            got = list(walk(root.spec, root.label, args.subpath, root.flags))
        except Exception as e:  # 권한 없는 공유 드라이브 등은 건너뛴다
            print(f" 실패 ({type(e).__name__}: {str(e)[:120]})")
            continue
        files.extend(got)
        print(f" {len(got):,}개")

    if not files:
        print("[!] 파일이 하나도 안 잡혔어. 경로나 권한을 확인해.", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else settings.scan_dir / "files.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for f in files:
            fh.write(json.dumps(f.to_dict(), ensure_ascii=False) + "\n")

    report(files)
    print(f"전체 목록 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
