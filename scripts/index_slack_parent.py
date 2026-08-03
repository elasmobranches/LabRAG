#!/usr/bin/env python3
"""Slack 스레드 전체를 parent vector로 인덱싱한다."""
from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
from qdrant_client import models as qm
from labrag.models import Models
from labrag.store import DENSE, Store

COLLECTION = "lab_slack_parent"
NS = uuid.UUID("b8c91365-8e41-44b2-9f9a-5df5dd4d61a4")

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--raw",default="data/slack/messages.jsonl")
    ap.add_argument("--contexts",default="data/slack/thread_context_combined.jsonl")
    ap.add_argument("--collection",default=COLLECTION); args=ap.parse_args()
    grouped={}
    for line in Path(args.raw).read_text(encoding="utf-8").splitlines():
        m=json.loads(line); grouped.setdefault((m["channel_id"],m["thread_ts"]),[]).append(m)
    contexts={}; cp=Path(args.contexts)
    if cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            x=json.loads(line); contexts[(x["channel_id"],x["thread_ts"])] = x["context"]
    parents=[]
    for (channel_id,thread_ts),ms in grouped.items():
        ms.sort(key=lambda x:x["ts"]); body="\n".join(f"{m['user_name']}: {m['text']}" for m in ms)[:7000]
        context=contexts.get((channel_id,thread_ts)); text=f"채널: #{ms[0]['channel_name']}\n스레드 전체 대화:\n{body}"
        embed=f"[Slack 스레드 전체 맥락]\n{context}\n\n{text}" if context else text
        parents.append({"channel_id":channel_id,"channel_name":ms[0]["channel_name"],"thread_ts":thread_ts,"text":text,"embed_text":embed,"file_id":f"slack-thread:{channel_id}:{thread_ts}","citation":f"Slack · #{ms[0]['channel_name']} · thread {thread_ts}","source":"slack_parent"})
    with Models() as models:
        store=Store(collection=args.collection); store.create(models.dim); vectors=models.embed([p["embed_text"] for p in parents])
        for i in range(0,len(parents),128):
            ps,vs=parents[i:i+128],vectors[i:i+128]
            store.client.upsert(collection_name=args.collection,points=[qm.PointStruct(id=str(uuid.uuid5(NS,p["file_id"])),vector={DENSE:v},payload={k:v for k,v in p.items() if k!="embed_text"}) for p,v in zip(ps,vs)],wait=True)
    print(f"Slack parent threads {len(parents):,} · collection={args.collection}"); return 0

if __name__ == "__main__": raise SystemExit(main())
