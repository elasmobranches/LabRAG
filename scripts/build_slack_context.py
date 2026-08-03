#!/usr/bin/env python3
"""Slack 스레드별 contextual context를 생성한다.

스레드마다 1회 호출하고 JSONL에 즉시 append한다. 이미 완료된 스레드는 건너뛰므로
중단 후 재실행할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from labrag.config import settings

PROMPT = """다음은 연구실 Slack의 한 대화 스레드다.

채널: #{channel}
대화:
{text}

이 스레드의 검색용 맥락을 한국어 1~2문장으로 작성해라.
규칙:
1. 대화에 실제로 나온 주제·프로젝트·문서·결정·문제만 사용한다.
2. 없는 내용을 추측하지 않는다.
3. 사람 이름·날짜·채널 참여·상태 변경을 요약의 핵심으로 삼지 않는다.
4. 주제나 실질적 내용이 없고 단순 알림·상태 변경이면 SKIP만 출력한다.
5. 설명만 출력한다. 따옴표나 '이 스레드는' 같은 서두는 쓰지 않는다."""


def load_threads(path: Path) -> dict[tuple[str, str], dict]:
    grouped = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        key = (m["channel_id"], m["thread_ts"])
        grouped.setdefault(key, {"channel_id": m["channel_id"], "channel": m["channel_name"],
                                 "thread_ts": m["thread_ts"], "messages": []})["messages"].append(m)
    return grouped


def generate(row: dict) -> str | None:
    messages = sorted(row["messages"], key=lambda x: x["ts"])
    text = "\n".join(f"{m['user_name']}: {m['text']}" for m in messages)[:6000]
    if len(text) < 100 or all(x in text for x in ("상태 변경됨",)):
        return None
    try:
        with httpx.Client(timeout=180) as client:
            r = client.post(f"{settings.vllm_url}/chat/completions", json={
                "model": settings.vllm_model,
                "messages": [{"role": "user", "content": PROMPT.format(channel=row["channel"], text=text)}],
                "temperature": 0.1, "max_tokens": 120,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"].strip()
            if not out or len(out) < 8 or out.upper().startswith("SKIP"):
                return None
            if any(term in out for term in ("참여했다", "참여했습니다", "채널에 들어", "상태 변경")):
                return None
            return out[:500]
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/slack/messages.jsonl")
    ap.add_argument("--out", default="data/slack/thread_context.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rows = load_threads(Path(args.raw))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line:
                x = json.loads(line); done[(x["channel_id"], x["thread_ts"])] = x
    todo = [r for k, r in rows.items() if k not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"스레드 전체 {len(rows):,} · 완료 {len(done):,} · 이번 생성 {len(todo):,}", flush=True)
    with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]; context = fut.result()
            if context:
                f.write(json.dumps({"channel_id": r["channel_id"], "thread_ts": r["thread_ts"], "context": context}, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"진행 {i:,}/{len(todo):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
