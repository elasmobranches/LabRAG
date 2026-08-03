from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import httpx

Status = Literal["success", "failed", "skipped"]
MAX_DETAIL = 2000


@dataclass(frozen=True)
class RefreshStep:
    name: str
    command: list[str]
    mutating: bool = True


@dataclass(frozen=True)
class StepResult:
    name: str
    status: Status
    returncode: int
    duration_seconds: float
    detail: str


@dataclass(frozen=True)
class RefreshReport:
    results: list[StepResult]

    def by_name(self, name: str) -> StepResult:
        return next(result for result in self.results if result.name == name)

    @property
    def exit_code(self) -> int:
        return int(any(result.status == "failed" for result in self.results))

    def as_dict(self) -> dict:
        return {
            "status": "success" if self.exit_code == 0 else "partial_failure",
            "exit_code": self.exit_code,
            "results": [asdict(result) for result in self.results],
        }


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def child_process_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{root}{os.pathsep}{current}" if current else str(root)
    )
    return env


def generation_health(
    base_url: str,
    *,
    getter: Callable[..., object] = httpx.get,
) -> dict[str, str]:
    try:
        response = getter(f"{base_url.rstrip('/')}/models", timeout=10)
        raise_for_status = getattr(response, "raise_for_status")
        raise_for_status()
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}


def build_steps(root: Path) -> list[RefreshStep]:
    del root
    python = sys.executable
    contexts = "data/slack/thread_context_combined.jsonl"
    return [
        RefreshStep(
            "preflight",
            [python, "scripts/weekly_refresh.py", "--internal-check"],
            mutating=False,
        ),
        RefreshStep("drive", [python, "scripts/index.py"]),
        RefreshStep(
            "slack_collect",
            [
                python,
                "scripts/index_slack.py",
                "--incremental",
                "--contexts",
                contexts,
                "--dry-run",
            ],
        ),
        RefreshStep(
            "slack_context",
            [
                python,
                "scripts/build_slack_context.py",
                "--out",
                contexts,
            ],
        ),
        RefreshStep(
            "slack_index",
            [
                python,
                "scripts/index_slack.py",
                "--incremental",
                "--contexts",
                contexts,
            ],
        ),
        RefreshStep(
            "slack_parent",
            [
                python,
                "scripts/index_slack_parent.py",
                "--contexts",
                contexts,
            ],
        ),
        RefreshStep(
            "verify",
            [python, "scripts/weekly_refresh.py", "--internal-verify"],
            mutating=False,
        ),
    ]


def _detail(completed: subprocess.CompletedProcess[str]) -> str:
    combined = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return combined[-MAX_DETAIL:]


def run_refresh(
    *,
    root: Path,
    check_only: bool,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> RefreshReport:
    results: list[StepResult] = []
    env = child_process_env(root)
    for step in build_steps(root):
        if check_only and step.mutating:
            results.append(StepResult(step.name, "skipped", 0, 0.0, "check mode"))
            continue
        started = time.monotonic()
        try:
            completed = runner(
                step.command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            duration = time.monotonic() - started
            status: Status = "success" if completed.returncode == 0 else "failed"
            detail = _detail(completed)
            print(
                json.dumps(
                    {
                        "step": step.name,
                        "status": status,
                        "returncode": completed.returncode,
                        "detail": detail,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            results.append(
                StepResult(
                    step.name,
                    status,
                    completed.returncode,
                    round(duration, 3),
                    detail,
                )
            )
        except Exception as exc:
            duration = time.monotonic() - started
            detail = f"{type(exc).__name__}: {exc}"[-MAX_DETAIL:]
            print(
                json.dumps(
                    {"step": step.name, "status": "failed", "detail": detail},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            results.append(
                StepResult(step.name, "failed", 1, round(duration, 3), detail)
            )
    return RefreshReport(results)
