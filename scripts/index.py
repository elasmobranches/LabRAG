#!/usr/bin/env python
"""드라이브 → Qdrant 인덱싱 (폴더 단위 스트리밍).

## '전체 스캔 후 인덱싱'이 아니라 '폴더 하나씩 훑고 바로 인덱싱'인 이유

연구실 드라이브에는 구성원별 미팅 폴더처럼 하위 폴더가 수백 개인 곳이 있다.
Google Drive API 는 디렉터리마다 호출이 필요해서 그런 폴더 하나가 수십 분을 잡아먹는다.
전체 스캔을 먼저 끝내려 하면 그 한 폴더 때문에 아무것도 검색할 수 없는 상태로
계속 기다리게 된다. 폴더 단위로 처리하면 진행한 만큼 즉시 검색 가능해지고,
느린 폴더가 나머지를 막지 않는다.

    python scripts/index.py                     # scope.json 의 include 전부
    python scripts/index.py --only 논문          # 이름에 '논문' 포함된 폴더만
    python scripts/index.py --retry-failed      # 이전 실패 건 재시도
    python scripts/index.py --stats             # 현재 인덱스 상태만 보기
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labrag import manifest
from labrag import scope as scope_mod
from labrag.config import settings
from labrag.drive import (LIST_TIMEOUT, ListTimeout, Root, check_remote,
                          expand_roots, list_files, list_subdirs)
from labrag.index import Result, index_files
from labrag.listing import build_listing_chunks
from labrag.models import Models
from labrag.store import Store


def show_stats() -> int:
    store = Store()
    print("[Qdrant]", store.stats())
    with manifest.connect() as conn:
        st = manifest.stats(conn)
        if not st:
            print("[manifest] 아직 비어있음")
            return 0
        print("[manifest]")
        for status, v in sorted(st.items()):
            print(f"  {status:10} 파일 {v['files']:6,}  청크 {v['chunks']:8,}")
        rows = conn.execute(
            "SELECT root, COUNT(*) n, SUM(n_chunks) c FROM files "
            "WHERE status='indexed' GROUP BY root ORDER BY c DESC"
        ).fetchall()
        if rows:
            print("[폴더별 인덱싱됨 — 본문]")
            for r in rows:
                print(f"  {r['root'][:44]:46} 파일 {r['n']:5,}  청크 {r['c'] or 0:7,}")

        lst = manifest.listing_stats(conn)
        if lst:
            print("[폴더별 위치 인덱싱 — 본문 없음]")
            for r in lst:
                gb = r["total_bytes"] / 1073741824
                print(f"  {r['root'][:44]:46} 파일 {r['n_files']:6,}  "
                      f"위치청크 {r['n_chunks']:4,}  {gb:6.1f}GB  {r['indexed_at'][:10]}")
        # 건너뜀/실패 사유 집계 — "왜 이 문서가 검색이 안 되지"의 답이 여기 있다.
        # SHOW_REASONS_LIMIT 은 조용히 자르지 않기 위한 안전장치일 뿐이다 — 사유
        # 종류가 이보다 적으면(보통 그렇다) 전부 보여준다. 예전엔 무조건 상위 12개만
        # 보여줘서 failed 27건 중 7건(사유 7가지)이 화면에 아예 안 나온 적이 있었다.
        SHOW_REASONS_LIMIT = 30
        for status in ("skipped", "failed"):
            rows = conn.execute(
                "SELECT error, COUNT(*) n FROM files WHERE status=? "
                "GROUP BY error ORDER BY n DESC", (status,)
            ).fetchall()
            if rows:
                print(f"[{status} 사유]")
                for r in rows[:SHOW_REASONS_LIMIT]:
                    print(f"  {r['n']:6,}  {(r['error'] or '(사유 없음)')[:90]}")
                hidden = rows[SHOW_REASONS_LIMIT:]
                if hidden:
                    print(f"  ... 그 외 {sum(r['n'] for r in hidden):,}건 "
                          f"({len(hidden)}가지 사유, 상위 {SHOW_REASONS_LIMIT}개만 표시)")
    return 0


# 적응적 분할의 최대 깊이. 이보다 깊어지면 포기하고 사유를 남긴다.
MAX_SPLIT_DEPTH = 4


def _direct_only(r: Root) -> Root:
    """이 폴더에 직접 놓인 파일만 보는 루트 (하위 폴더 미포함 → 항상 빠르다)."""
    return Root(label=r.label, spec=r.spec, flags=r.flags, depth=1,
                mode=r.mode, folder_id=r.folder_id)


def resolve_files(root: Root, excluded: set[str] | None = None,
                  indent: str = "│") -> list[tuple[Root, list]]:
    """루트의 파일 목록을 얻는다. 시간이 오래 걸리면 하위 폴더로 쪼개 재시도.

    왜 필요한가: 깊은 곳에 거대한 폴더가 숨어 있을 수 있다. 실제로
    `[Workspace]/학회/2024_LightSym/EnergyPlus modeling/DesignBuilder/EnergyPlus modeling`
    이 그랬다 — EnergyPlus 는 시뮬레이션 출력 파일을 수천 개 만들고, 폴더 이름이
    재귀적으로 반복될 정도로 중첩돼 있었다. 학회 폴더 안에 있어서 예상하기 어려웠고,
    한 단계 분할(`split=true`)로는 잡히지 않았다.

    손으로 큰 폴더를 계속 찾아다니는 것은 두더지잡기다. 목록 조회가 느리면 자동으로
    한 단계 더 쪼개 들어가고, 최대 깊이에 닿으면 **포기하지 않고 직속 파일만이라도
    건진다** — 그 아래가 시뮬레이션 출력이어도 그 층의 문서는 살려야 한다.

    excluded: 이 ID 를 가진 폴더는 들어가지 않는다 (scope.json 의 exclude).
    """
    ex = excluded or set()

    def go(r: Root, depth: int) -> list[tuple[Root, list]]:
        try:
            return [(r, list_files(r))]
        except ListTimeout:
            pass  # 아래에서 쪼갠다

        out: list[tuple[Root, list]] = []
        # 이 층의 직속 파일은 어떤 경우에도 먼저 건진다
        try:
            out.append((_direct_only(r), list_files(_direct_only(r), timeout=60)))
        except Exception as e:
            print(f"{indent}   직속 파일 조회 실패 ({r.label}): {type(e).__name__}")

        if depth >= MAX_SPLIT_DEPTH:
            print(f"{indent} ⚠ {r.label}: 분할 깊이 {MAX_SPLIT_DEPTH} 도달 — "
                  f"이 아래는 인덱싱하지 않음 (직속 파일만 처리)")
            return out

        print(f"{indent} 목록 조회 {LIST_TIMEOUT}초 초과 → 하위 폴더로 분할: {r.label}")
        try:
            subs = list_subdirs(r)
        except Exception as e:
            print(f"{indent}   하위 폴더 조회 실패: {type(e).__name__} — 직속 파일만")
            return out

        for fid, name in sorted(subs, key=lambda x: x[1]):
            if fid in ex:
                print(f"{indent}   제외됨(scope.json): {r.label}/{name}")
                continue
            child = Root(label=f"{r.label}/{name}", spec=r.spec,
                         flags=("--drive-root-folder-id", fid),
                         mode=r.mode, folder_id=fid)
            out.extend(go(child, depth + 1))
        return out

    return go(root, 0)


def do_listing(root: Root, models: Models, store: Store, conn,
               excluded: set[str] | None = None) -> int:
    """위치 인덱싱: 내용 없이 "어디에 무엇이 있다"만 넣는다. 청크 수를 돌려준다."""
    files = [f for _, fs in resolve_files(root, excluded) for f in fs]
    if not files:
        print("│ 파일 없음")
        return 0
    url = f"https://drive.google.com/drive/folders/{root.folder_id}"
    chunks = build_listing_chunks(files, root.label, url)
    vectors = models.embed([c.embed_text for c in chunks])
    # 이전 위치 청크를 지우고 새로 넣는다 (디렉터리가 사라졌을 수 있으므로)
    store.delete_root([root.label])
    store.upsert(chunks, vectors)
    manifest.record_listing(conn, root.label, root.folder_id, len(files),
                            len(chunks), sum(f.size for f in files if f.size > 0))
    conn.commit()
    print(f"│ 파일 {len(files):,}개 → 위치 청크 {len(chunks):,}개 (내용은 인덱싱하지 않음)")
    return len(chunks)


def drop_roots(roots: list[str], assume_yes: bool = False) -> int:
    """폴더 단위로 인덱스를 제거한다 (드라이브 원본은 건드리지 않는다).

    청크와 manifest 기록을 함께 지운다. manifest 를 남기면 다음 실행에서
    '변경없음'으로 판단해 다시 인덱싱되지 않는 유령 상태가 된다.
    """
    store = Store()
    if not store.exists():
        print("[!] 컬렉션이 없음", file=sys.stderr)
        return 1

    with manifest.connect() as conn:
        known = sorted({r["root"] for r in conn.execute("SELECT DISTINCT root FROM files")}
                       | {r["root"] for r in manifest.listing_stats(conn)})
        targets = [r for r in roots if r in known]
        unknown = [r for r in roots if r not in known]
        if unknown:
            print(f"[!] manifest 에 없는 폴더명: {unknown}", file=sys.stderr)
            print(f"    사용 가능: {known}", file=sys.stderr)
            if not targets:
                return 1

        print("제거 대상:")
        for r in targets:
            row = conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(n_chunks),0) c FROM files WHERE root=?",
                (r,)).fetchone()
            kind = " [위치 인덱싱]" if manifest.has_listing(conn, r) else ""
            print(f"  {r}{kind}  — 파일 {row['n']:,}개 · 청크 {row['c']:,}개 "
                  f"(Qdrant 실측 {store.count_root(r):,})")

        if not assume_yes:
            ans = input("\n제거할까? 드라이브 원본은 그대로 남는다 [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("취소")
                return 0

        store.delete_root(targets)
        for r in targets:
            conn.execute("DELETE FROM files WHERE root=?", (r,))
        manifest.forget_listing(conn, targets)
        conn.commit()

    print(f"\n제거 완료. 남은 포인트 {store.count():,}개")
    print("※ 다시 넣으려면 config/scope.json 에서 disabled 를 풀고 "
          "python scripts/index.py 를 실행하면 된다.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="이름 일부가 일치하는 폴더만")
    ap.add_argument("--force", action="store_true", help="변경 여부 무시하고 전부 재인덱싱")
    ap.add_argument("--retry-failed", action="store_true", help="이전 실패 건 재시도")
    ap.add_argument("--stats", action="store_true", help="인덱스 상태만 출력하고 종료")
    ap.add_argument("--recreate", action="store_true",
                    help="컬렉션을 지우고 새로 만든다 (임베딩 모델을 바꿨을 때)")
    ap.add_argument("--drop-root", action="append",
                    help="이 폴더의 인덱스를 제거한다 (--stats 의 폴더명과 동일하게). "
                         "여러 번 지정 가능. '일단 넣고 나중에 빼기'용")
    ap.add_argument("--yes", action="store_true", help="--drop-root 확인 건너뛰기")
    ap.add_argument("--refresh-listing", action="store_true",
                    help="mode=listing 폴더의 위치 정보를 다시 훑는다 "
                         "(기본은 한 번 만들면 건너뜀 — 데이터셋 폴더는 목록 조회가 느림)")
    args = ap.parse_args()

    if args.stats:
        return show_stats()

    if args.drop_root:
        return drop_roots(args.drop_root, assume_yes=args.yes)

    if not check_remote():
        print(f"[!] rclone 리모트 '{settings.remote}' 없음", file=sys.stderr)
        return 1

    sc = scope_mod.load()
    # split=True 인 폴더를 하위 폴더 단위로 펼친다. --only 필터는 펼친 뒤에 적용해야
    # '--only 이연구' 처럼 하위 폴더 이름으로 고를 수 있다.
    roots = expand_roots(sc.roots())
    excluded = sc.excluded_ids
    if args.only:
        roots = [r for r in roots if any(o in r.label for o in args.only)]
    if not roots:
        print("[!] 대상 폴더가 없음. config/scope.json 확인", file=sys.stderr)
        return 1

    store = Store()
    total = Result()

    with Models() as models:
        health = models.health()
        if any(v != "ok" for v in health.values()):
            print(f"[!] 모델 서버 상태 이상: {health}", file=sys.stderr)
            print("    docker compose up -d embed rerank", file=sys.stderr)
            return 1
        print(f"[*] 임베딩 차원 {models.dim} · 대상 폴더 {len(roots)}개\n")

        if args.recreate or not store.exists():
            store.create(models.dim, recreate=args.recreate)

        with manifest.connect() as conn:
            for root in roots:
                print(f"┌ {root.label}"
                      + ("  [위치 인덱싱]" if root.mode == "listing" else ""))
                t0 = time.time()

                if root.mode == "listing":
                    # 이미 만들어뒀으면 건너뛴다. 데이터셋 폴더는 목록 조회만
                    # 수십 분이라 매 실행마다 다시 훑으면 증분 동기화를 못 쓴다.
                    if manifest.has_listing(conn, root.label) and not args.refresh_listing:
                        print("│ 이미 인덱싱됨 — 갱신하려면 --refresh-listing")
                        print(f"└ 건너뜀\n")
                        continue
                    try:
                        n = do_listing(root, models, store, conn, excluded)
                        total.chunks += n
                    except Exception as e:
                        print(f"└ 실패: {type(e).__name__}: {e}\n")
                        continue
                    print(f"└ {time.time() - t0:.1f}s · 누적 포인트 {store.count():,}\n")
                    continue

                try:
                    parts = resolve_files(root, excluded)
                except Exception as e:
                    print(f"└ 목록 실패: {type(e).__name__}: {e}\n")
                    continue
                n_files = sum(len(fs) for _, fs in parts)
                extra = f", {len(parts)}개로 분할" if len(parts) > 1 else ""
                print(f"│ 파일 {n_files:,}개 (목록 {time.time() - t0:.1f}s{extra})")

                for sub, files in parts:
                    if not files:
                        continue
                    if len(parts) > 1:
                        print(f"│  ├ {sub.label} — {len(files):,}개")
                    res = index_files(files, models, store, conn,
                                      force=args.force, retry_failed=args.retry_failed)
                    # 폴더 단위로 합산
                    total.indexed += res.indexed
                    total.chunks += res.chunks
                    total.unchanged += res.unchanged
                    total.skipped += res.skipped
                    total.failed += res.failed
                    total.still_failed += res.still_failed
                    total.errors.extend(res.errors)
                    for k, v in res.skip_reasons.items():
                        total.skip_reasons[k] = total.skip_reasons.get(k, 0) + v
                    print(f"│  인덱싱 {res.indexed} · 청크 {res.chunks:,} · "
                          f"변경없음 {res.unchanged} · 건너뜀 {res.skipped} · 실패 {res.failed}")

                print(f"└ {time.time() - t0:.1f}s · 누적 포인트 {store.count():,}\n")

    print("=" * 72)
    print(total.summary())
    print(f"\nQdrant: {store.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
