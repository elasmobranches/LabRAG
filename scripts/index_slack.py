#!/usr/bin/env python3
"""Slack 채널을 수집하고 Slack 전용 dense 컬렉션에 인덱싱한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from pathlib import Path

from qdrant_client import models as qm

from labrag.models import Models
from labrag.slack import SlackAPIError, SlackClient, SlackMessage, collect, collect_after, contextual_chunks, write_jsonl
from labrag.store import DENSE, Store

COLLECTION = "lab_slack"
NAMESPACE = uuid.UUID("7a9c7f0e-2e6a-4c57-8fd5-3ca88c6b7f0d")


def point_id(c: dict) -> str:
    key = f"{c['channel_id']}:{c['thread_ts']}:{c['index']}"
    return str(uuid.uuid5(NAMESPACE, key))


def payload(c: dict) -> dict:
    digest = hashlib.sha256(c["text"].encode()).hexdigest()
    return {k: v for k, v in {**c, "source": "slack", "content_hash": digest,
            "citation": f"Slack · #{c['channel_name']} · thread {c['thread_ts']}"}
            .items() if k != "embed_text"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/slack/messages.jsonl")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--limit-channels", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--contexts", default=None, help="thread context JSONL")
    ap.add_argument("--incremental", action="store_true", help="기존 raw의 채널별 마지막 ts 이후만 수집")
    args = ap.parse_args()

    with SlackClient() as slack:
        auth = slack.auth_test()
        print(f"Slack 인증 성공 · {auth.get('team')} · user={auth.get('user_id')}")
        # 현재 collector는 전체 채널을 대상으로 하므로 권한 문제를 초기에 드러낸다.
        channels = slack.conversations()
        print(f"접근 가능한 채널 {len(channels)}개 · 비공개 {sum(c.get('is_private', False) for c in channels)}개")
        if args.limit_channels:
            # 테스트용: API client의 채널 목록을 제한하는 대신 수집 결과를 제한한다.
            channels = channels[:args.limit_channels]
        if args.incremental and Path(args.raw).exists() and Path(args.raw).stat().st_size > 0:
            existing = [json.loads(x) for x in Path(args.raw).read_text(encoding="utf-8").splitlines() if x]
            latest = {}
            for m in existing:
                latest[m["channel_id"]] = max(latest.get(m["channel_id"], "0"), m["ts"])
            new = collect_after(slack, channels, latest)
            existing_keys={(m["channel_id"],m["ts"]) for m in existing}
            additions=[m.__dict__ for m in new if (m.channel_id,m.ts) not in existing_keys]
            messages=[SlackMessage(**m) for m in existing] + [SlackMessage(**m) for m in additions]
            print(f"증분 수집 {len(new):,}개 · 전체 raw {len(messages):,}개", flush=True)
        else:
            messages = collect(slack, channels=channels, cache_dir=Path("data/slack/channels"))

    write_jsonl(messages, Path(args.raw))
    chunks = contextual_chunks(messages)
    if args.contexts:
        contexts = {}
        for line in Path(args.contexts).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            contexts[(row["channel_id"], row["thread_ts"])] = row["context"]
        chunks = contextual_chunks(messages, thread_contexts=contexts)
    print(f"메시지 {len(messages):,}개 · contextual chunk {len(chunks):,}개 · raw={args.raw}")
    if args.dry_run:
        return 0

    with Models() as models:
        store = Store(collection=args.collection)
        store.create(models.dim)
        vectors = models.embed([c["embed_text"] for c in chunks])
        for i in range(0, len(chunks), 128):
            cs, vs = chunks[i:i + 128], vectors[i:i + 128]
            store.client.upsert(
                collection_name=args.collection,
                points=[qm.PointStruct(id=point_id(c), vector={DENSE: v}, payload=payload(c))
                        for c, v in zip(cs, vs)], wait=True,
            )
    print(f"Qdrant 컬렉션 '{args.collection}'에 {len(chunks):,}개 저장 완료")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SlackAPIError as e:
        raise SystemExit(f"[Slack 오류] {e}")
