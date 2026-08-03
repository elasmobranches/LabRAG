#!/usr/bin/env python
"""config/scope.json 의 include 폴더만 스캔한다.

드라이브 전체 재귀 스캔은 데이터셋 폴더 때문에 수십 분이 걸리고 쓸모도 없다.
여기서는 실제 인덱싱할 폴더만 훑어서 바로 쓸 수 있는 숫자를 낸다.

    python scripts/scan_scope.py              # 전부
    python scripts/scan_scope.py --only 회의록  # 폴더 하나만
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labrag import scope as scope_mod
from labrag.config import settings
from labrag.drive import DriveFile, check_remote, walk
from labrag.filetypes import NEEDS_CONVERT, NEEDS_MODEL, TEXT_CATEGORIES


def human(n: int) -> str:
    if n < 0:
        return "?"
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}TB"


def table(rows: list[tuple], headers: tuple[str, ...]) -> str:
    def w(s) -> int:  # 한글 폭 보정
        return sum(2 if ord(c) > 0x1100 else 1 for c in str(s))

    all_rows = [headers, *[tuple(str(c) for c in r) for r in rows]]
    widths = [max(w(r[i]) for r in all_rows) for i in range(len(headers))]
    def pad(s, i):
        return str(s) + " " * max(0, widths[i] - w(s))
    out = ["  ".join(pad(h, i) for i, h in enumerate(headers)),
           "  ".join("-" * widths[i] for i in range(len(headers)))]
    for r in all_rows[1:]:
        out.append("  ".join(pad(c, i) for i, c in enumerate(r)))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="이 폴더만 (이름 일부 일치)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not check_remote():
        print(f"[!] rclone 리모트 '{settings.remote}' 없음", file=sys.stderr)
        return 1

    sc = scope_mod.load()
    roots = sc.roots()
    if args.only:
        roots = [r for r in roots if any(o in r.label for o in args.only)]
        if not roots:
            print("[!] --only 와 맞는 폴더 없음", file=sys.stderr)
            return 1

    print(f"[*] 대상 폴더 {len(roots)}개 (제외 {len(sc.exclude)}개, "
          f"My Drive={sc.my_drive_mode})\n")

    files: list[DriveFile] = []
    per_root: dict[str, int] = {}
    failed: list[tuple[str, str]] = []
    for r in roots:
        t0 = time.time()
        print(f"  {r.label[:48]:50}", end="", flush=True)
        try:
            got = list(walk(r.spec, r.label, flags=r.flags))
        except Exception as e:
            print(f" 실패: {type(e).__name__}")
            failed.append((r.label, str(e)[:200]))
            continue
        files.extend(got)
        per_root[r.label] = len(got)
        print(f" {len(got):6,}개  {time.time() - t0:5.1f}s")

    if failed:
        print("\n[!] 접근 실패한 폴더:")
        for label, err in failed:
            print(f"  - {label}: {err}")

    if not files:
        print("\n[!] 파일이 없음", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else settings.scan_dir / "scope_files.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for f in files:
            fh.write(json.dumps(f.to_dict(), ensure_ascii=False) + "\n")

    # ---------------- 리포트 ----------------
    total_size = sum(f.size for f in files if f.size > 0)
    n_gnative = sum(1 for f in files if f.is_google_native)
    print(f"\n{'=' * 74}")
    print(f"총 {len(files):,}개 파일 · {human(total_size)} "
          f"(Google 네이티브 {n_gnative:,}개는 용량 미포함)")
    print("=" * 74)

    by_cat: dict[str, list[DriveFile]] = defaultdict(list)
    for f in files:
        by_cat[f.category].append(f)

    def bucket(cat: str) -> str:
        if cat in TEXT_CATEGORIES:
            return "① 바로 인덱싱"
        if cat in NEEDS_CONVERT:
            return "② 변환 필요"
        if cat in NEEDS_MODEL:
            return "③ OCR/ASR"
        return "④ 대상 아님"

    print("\n[카테고리별]")
    print(table(
        [(bucket(c), c, f"{len(fs):,}", human(sum(x.size for x in fs if x.size > 0)))
         for c, fs in sorted(by_cat.items(), key=lambda kv: (bucket(kv[0]), -len(kv[1])))],
        ("처리 단계", "카테고리", "파일수", "용량")))

    tier1 = [f for f in files if f.category in TEXT_CATEGORIES]
    tier2 = [f for f in files if f.category in NEEDS_CONVERT]
    tier3 = [f for f in files if f.category in NEEDS_MODEL]
    print(f"\n① 바로 인덱싱 {len(tier1):,}개  ② 변환 필요 {len(tier2):,}개  "
          f"③ OCR/ASR {len(tier3):,}개  ④ 대상 아님 {len(files) - len(tier1) - len(tier2) - len(tier3):,}개")

    print("\n[폴더별 ① 대상]")
    fold: dict[str, list[DriveFile]] = defaultdict(list)
    for f in tier1:
        fold[f.root].append(f)
    print(table(
        [(r, f"{len(fold.get(r, [])):,}", f"{per_root.get(r, 0):,}")
         for r in sorted(per_root, key=lambda k: -len(fold.get(k, [])))],
        ("폴더", "①대상", "전체")))

    exts = Counter(Path(f.name).suffix.lower() or "(없음)" for f in files)
    print("\n[확장자 상위 20]")
    print(table([(e, f"{c:,}") for e, c in exts.most_common(20)], ("확장자", "개수")))

    # 구형 .hwp 는 파서가 없으니 개수를 따로 보고한다
    n_old_hwp = sum(1 for f in files if f.name.lower().endswith(".hwp"))
    n_hwpx = sum(1 for f in files if f.name.lower().endswith(".hwpx"))
    if n_old_hwp or n_hwpx:
        print(f"\n[한글] .hwp(구형, 미지원) {n_old_hwp}개 · .hwpx(지원) {n_hwpx}개")

    print(f"\n목록 저장: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
