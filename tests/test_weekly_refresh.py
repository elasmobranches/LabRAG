from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from labrag.refresh import (
    build_steps,
    generation_health,
    project_root,
    run_refresh,
)


class WeeklyRefreshTests(unittest.TestCase):
    def test_child_scripts_can_import_project_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "probe_pkg").mkdir()
            (root / "probe_pkg" / "__init__.py").write_text(
                "VALUE = 42\n",
                encoding="utf-8",
            )
            (root / "scripts" / "probe.py").write_text(
                "from probe_pkg import VALUE\n"
                "raise SystemExit(0 if VALUE == 42 else 1)\n",
                encoding="utf-8",
            )

            def runner(command, **kwargs):
                del command
                return subprocess.run(
                    [sys.executable, "scripts/probe.py"],
                    **kwargs,
                )

            report = run_refresh(root=root, check_only=True, runner=runner)

        self.assertEqual(report.by_name("preflight").status, "success")
        self.assertEqual(report.by_name("verify").status, "success")

    def test_generation_health_reports_unavailable_endpoint(self):
        def failing_get(url, timeout):
            del url, timeout
            raise ConnectionError("generation offline")

        result = generation_health("http://localhost:8000/v1", getter=failing_get)
        self.assertEqual(result["status"], "error")
        self.assertIn("generation offline", result["detail"])

    def test_project_root_follows_checked_out_package(self):
        root = project_root()
        self.assertTrue((root / "labrag" / "refresh.py").exists())
        self.assertTrue((root / "scripts" / "weekly_refresh.py").exists())

    def test_drive_failure_does_not_skip_slack_steps(self):
        calls = []

        def runner(command, **kwargs):
            del kwargs
            calls.append(command)
            failed = command[1:] == ["scripts/index.py"]
            return subprocess.CompletedProcess(
                command,
                1 if failed else 0,
                "",
                "drive failed" if failed else "",
            )

        report = run_refresh(
            root=Path("/workspace/labrag"),
            check_only=False,
            runner=runner,
        )

        self.assertEqual(
            [result.name for result in report.results],
            [
                "preflight",
                "drive",
                "slack_collect",
                "slack_context",
                "slack_index",
                "slack_parent",
                "verify",
            ],
        )
        self.assertEqual(report.by_name("drive").status, "failed")
        self.assertEqual(report.by_name("slack_index").status, "success")
        self.assertEqual(report.exit_code, 1)
        self.assertEqual(len(calls), 7)

    def test_weekly_commands_are_incremental_and_nondestructive(self):
        commands = [step.command for step in build_steps(Path("/workspace/labrag"))]
        flat = "\n".join(" ".join(command) for command in commands)
        self.assertIn("scripts/index.py", flat)
        self.assertIn("scripts/index_slack.py --incremental", flat)
        self.assertIn(
            "--contexts data/slack/thread_context_combined.jsonl",
            flat,
        )
        for forbidden in ("--force", "--recreate", "--drop-root"):
            self.assertNotIn(forbidden, flat)

    def test_check_mode_skips_mutating_steps(self):
        calls = []

        def runner(command, **kwargs):
            del kwargs
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "ok", "")

        report = run_refresh(
            root=Path("/workspace/labrag"),
            check_only=True,
            runner=runner,
        )

        self.assertEqual(
            [result.name for result in report.results if result.status == "success"],
            ["preflight", "verify"],
        )
        self.assertTrue(all(
            report.by_name(name).status == "skipped"
            for name in (
                "drive",
                "slack_collect",
                "slack_context",
                "slack_index",
                "slack_parent",
            )
        ))
        self.assertEqual(len(calls), 2)
        self.assertEqual(report.exit_code, 0)

    def test_report_json_does_not_include_unbounded_output(self):
        long_text = "x" * 5000

        def runner(command, **kwargs):
            del kwargs
            return subprocess.CompletedProcess(command, 0, long_text, "")

        report = run_refresh(
            root=Path("/workspace/labrag"),
            check_only=True,
            runner=runner,
        )
        payload = report.as_dict()
        self.assertLessEqual(
            len(payload["results"][0]["detail"]),
            2000,
        )


if __name__ == "__main__":
    unittest.main()
