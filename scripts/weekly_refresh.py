#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labrag.config import settings
from labrag.drive import check_remote
from labrag.models import Models
from labrag.refresh import generation_health, project_root, run_refresh
from labrag.slack import SlackClient
from labrag.store import Store


def internal_check() -> int:
    checks: dict[str, object] = {}
    checks["drive_remote"] = "ok" if check_remote() else "error"
    try:
        with Models() as models:
            checks["models"] = models.health()
    except Exception as exc:
        checks["models"] = f"{type(exc).__name__}: {exc}"
    try:
        with SlackClient() as slack:
            auth = slack.auth_test()
            checks["slack"] = {
                "status": "ok",
                "team": auth.get("team"),
                "user_id": auth.get("user_id"),
            }
    except Exception as exc:
        checks["slack"] = f"{type(exc).__name__}: {exc}"
    checks["generation"] = generation_health(settings.vllm_url)
    success = (
        checks["drive_remote"] == "ok"
        and isinstance(checks["models"], dict)
        and all(value == "ok" for value in checks["models"].values())
        and isinstance(checks["slack"], dict)
        and checks["generation"].get("status") == "ok"
    )
    print(json.dumps(checks, ensure_ascii=False))
    return 0 if success else 1


def internal_verify() -> int:
    collections = {}
    for name in (settings.collection, "lab_slack", "lab_slack_parent"):
        try:
            store = Store(collection=name)
            collections[name] = store.stats()
        except Exception as exc:
            collections[name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps({"collections": collections}, ensure_ascii=False, default=str))
    return int(any(isinstance(value, str) for value in collections.values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--internal-check", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-verify", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_check:
        return internal_check()
    if args.internal_verify:
        return internal_verify()
    root = project_root()
    report = run_refresh(root=root, check_only=args.check)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
