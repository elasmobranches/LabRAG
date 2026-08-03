"""Slack Web API 수집과 Slack 대화용 contextual chunk 생성."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

from langchain_text_splitters import RecursiveCharacterTextSplitter


class SlackAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlackMessage:
    channel_id: str
    channel_name: str
    ts: str
    thread_ts: str
    user_id: str
    user_name: str
    text: str
    reply_count: int = 0
    permalink: str | None = None


class SlackClient:
    def __init__(self, token: str | None = None, timeout: float = 60.0) -> None:
        self.token = token or os.environ.get("SLACK_TOKEN")
        if not self.token:
            raise SlackAPIError("SLACK_TOKEN 환경변수가 없습니다")
        self.http = httpx.Client(
            base_url="https://slack.com/api",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=timeout,
        )

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> "SlackClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        for attempt in range(6):
            r = self.http.get(f"/{method}", params=params)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", "5"))
                time.sleep(min(wait, 120))
                continue
            r.raise_for_status()
            data = r.json()
            if data.get("ok"):
                return data
            if data.get("error") == "ratelimited":
                wait = int(r.headers.get("Retry-After", "5"))
                time.sleep(min(wait, 120))
                continue
            raise SlackAPIError(f"{method}: {data.get('error', 'unknown_error')}")
        raise SlackAPIError(f"{method}: rate limit 재시도 초과")

    def auth_test(self) -> dict[str, Any]:
        return self.call("auth.test")

    def conversations(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cursor = ""
        while True:
            args: dict[str, Any] = {
                "types": "public_channel,private_channel",
                "exclude_archived": "true",
                "limit": 200,
            }
            if cursor:
                args["cursor"] = cursor
            try:
                data = self.call("conversations.list", **args)
            except SlackAPIError as e:
                raise SlackAPIError(
                    f"채널 목록 조회 실패: {e}. 공개/비공개 채널용 channels:read, "
                    "groups:read 권한과 비공개 채널 멤버십을 확인하세요."
                ) from e
            out.extend(data.get("channels", []))
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                return out

    def _paged(self, method: str, **params: Any) -> Iterator[dict[str, Any]]:
        cursor = ""
        while True:
            args = dict(params, limit=200)
            if cursor:
                args["cursor"] = cursor
            data = self.call(method, **args)
            yield from data.get("messages", [])
            cursor = data.get("response_metadata", {}).get("next_cursor", "")
            if not cursor:
                return

    def messages(self, channel: dict[str, Any], oldest: str | None = None) -> list[dict[str, Any]]:
        channel_id = channel["id"]
        args: dict[str, Any] = {"channel": channel_id}
        if oldest:
            args["oldest"] = oldest
        roots = list(self._paged("conversations.history", **args))
        out: list[dict[str, Any]] = []
        for root in roots:
            thread_ts = root.get("thread_ts", root.get("ts", ""))
            if root.get("reply_count", 0):
                replies = list(self._paged(
                    "conversations.replies", channel=channel_id, ts=thread_ts
                ))
                out.extend(replies)
            else:
                out.append(root)
        return out


def load_users(client: SlackClient) -> dict[str, str]:
    users: dict[str, str] = {}
    cursor = ""
    while True:
        args: dict[str, Any] = {"limit": 200}
        if cursor:
            args["cursor"] = cursor
        data = client.call("users.list", **args)
        for user in data.get("members", []):
            profile = user.get("profile", {})
            users[user["id"]] = (
                profile.get("display_name") or profile.get("real_name")
                or user.get("name") or user["id"]
            )
        cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not cursor:
            return users


def collect(client: SlackClient, channels: list[dict[str, Any]] | None = None,
            cache_dir: Path | None = None) -> list[SlackMessage]:
    users = load_users(client)
    out: list[SlackMessage] = []
    for channel in channels if channels is not None else client.conversations():
        cache_file = cache_dir / f"{channel['id']}.jsonl" if cache_dir else None
        if cache_file and cache_file.exists():
            rows = [json.loads(line) for line in cache_file.read_text(encoding="utf-8").splitlines() if line]
            out.extend(SlackMessage(**row) for row in rows)
            print(f"[Slack] {channel.get('name', channel['id'])}: 캐시 {len(rows):,}개", flush=True)
            continue
        rows: list[dict[str, Any]] = []
        for msg in client.messages(channel):
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            ts = msg.get("ts", "")
            rows.append(SlackMessage(
                channel_id=channel["id"], channel_name=channel.get("name", channel["id"]),
                ts=ts, thread_ts=msg.get("thread_ts", ts),
                user_id=msg.get("user", "unknown"),
                user_name=users.get(msg.get("user", ""), msg.get("user", "unknown")),
                text=text, reply_count=int(msg.get("reply_count", 0) or 0),
            ).__dict__)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".tmp")
            tmp.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            tmp.replace(cache_file)
        out.extend(SlackMessage(**row) for row in rows)
        print(f"[Slack] {channel.get('name', channel['id'])}: {len(rows):,}개", flush=True)
    out.sort(key=lambda m: (m.channel_name, m.thread_ts, m.ts))
    return out


def collect_after(client: SlackClient, channels: list[dict[str, Any]],
                  latest: dict[str, str]) -> list[SlackMessage]:
    """채널별 마지막 ts 이후의 새 root/thread를 수집한다."""
    users = load_users(client)
    out: list[SlackMessage] = []
    for channel in channels:
        rows = []
        for msg in client.messages(channel, oldest=latest.get(channel["id"])):
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            ts = msg.get("ts", "")
            rows.append(SlackMessage(
                channel_id=channel["id"], channel_name=channel.get("name", channel["id"]),
                ts=ts, thread_ts=msg.get("thread_ts", ts),
                user_id=msg.get("user", "unknown"),
                user_name=users.get(msg.get("user", ""), msg.get("user", "unknown")),
                text=text, reply_count=int(msg.get("reply_count", 0) or 0),
            ))
        out.extend(rows)
        if rows:
            print(f"[Slack incremental] {channel.get('name', channel['id'])}: {len(rows):,}개", flush=True)
    return out


def write_jsonl(messages: list[SlackMessage], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for m in messages:
            row = m.__dict__ if hasattr(m, "__dict__") else m
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def contextual_chunks(messages: list[SlackMessage], size: int = 1200,
                      overlap: int = 200,
                      thread_contexts: dict[tuple[str, str], str] | None = None) -> list[dict[str, Any]]:
    """스레드 경계를 보존하고 긴 스레드만 재귀적으로 나눈다.

    LLM이 생성한 contextual retrieval 문맥은 별도 단계에서 추가할 수 있도록
    채널·작성자·스레드 첫 메시지를 항상 임베딩 텍스트에 포함한다.
    """
    by_thread: dict[tuple[str, str], list[SlackMessage]] = {}
    for m in messages:
        by_thread.setdefault((m.channel_id, m.thread_ts), []).append(m)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "다. ", "요. ", " ", ""],
        keep_separator=True,
    )
    chunks: list[dict[str, Any]] = []
    for (channel_id, thread_ts), thread in by_thread.items():
        thread.sort(key=lambda m: m.ts)
        header = f"채널: #{thread[0].channel_name}\n스레드 시작: {thread[0].text}\n"
        body = "\n".join(f"{m.user_name}: {m.text}" for m in thread)
        pieces = splitter.split_text(header + body) if len(header + body) > size else [header + body]
        for index, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunks.append({
                "channel_id": channel_id, "channel_name": thread[0].channel_name,
                "thread_ts": thread_ts, "index": index, "text": piece,
                "source_ts": thread[0].ts,
                "file_id": f"slack:{channel_id}:{thread_ts}:{index}",
            })
            context = (thread_contexts or {}).get((channel_id, thread_ts))
            if context:
                chunks[-1]["context"] = context
                chunks[-1]["embed_text"] = f"[Slack 대화 맥락]\n{context}\n\n{piece}"
            else:
                chunks[-1]["embed_text"] = piece
    return chunks
