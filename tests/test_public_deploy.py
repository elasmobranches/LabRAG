import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


class PublicDeployTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return (DEPLOY / name).read_text(encoding="utf-8")

    def test_templates_do_not_pin_a_machine_or_account(self):
        names = (
            "labrag-server.service.example",
            "labrag-weekly-refresh.service.example",
            "labrag-weekly-refresh.timer.example",
        )
        for name in names:
            with self.subTest(name=name):
                text = self.read(name)
                self.assertNotIn("/home/ubuntu", text)
                self.assertIsNone(re.search(r"\b192\.168(?:\.\d{1,3}){2}\b", text))

    def test_timer_runs_monday_at_four_and_catches_missed_runs(self):
        text = self.read("labrag-weekly-refresh.timer.example")
        self.assertIn("OnCalendar=Mon *-*-* 04:00:00", text)
        self.assertIn("Persistent=true", text)

    def test_services_use_home_placeholders_and_environment_file(self):
        for name in (
            "labrag-server.service.example",
            "labrag-weekly-refresh.service.example",
        ):
            with self.subTest(name=name):
                text = self.read(name)
                self.assertIn("%h/LabRAG", text)
                self.assertIn("EnvironmentFile=-%h/LabRAG/.env", text)

    def test_refresh_service_uses_lock_and_incremental_entrypoint(self):
        text = self.read("labrag-weekly-refresh.service.example")
        self.assertIn("flock", text)
        self.assertIn("scripts/weekly_refresh.py", text)
        self.assertNotIn("--rebuild", text)
        self.assertNotIn("--drop", text)

    def test_compose_uses_configurable_web_bind_address(self):
        text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("LABRAG_WEB_BIND_ADDRESS", text)
        self.assertNotIn("/home/ubuntu", text)


if __name__ == "__main__":
    unittest.main()
