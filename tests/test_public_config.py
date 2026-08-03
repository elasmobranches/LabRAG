import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from labrag.config import Settings


ROOT = Path(__file__).resolve().parents[1]


class PublicConfigTests(unittest.TestCase):
    def test_default_paths_do_not_reference_a_developer_home(self):
        settings = Settings(_env_file=None)
        rendered = " ".join(
            str(value)
            for value in (settings.root, settings.data_dir, settings.rclone_config)
        )
        self.assertNotIn("/home/ubuntu", rendered)

    def test_model_name_comes_from_environment(self):
        env = os.environ.copy()
        env["LABRAG_MODEL_ID"] = "DemoRAG"
        result = subprocess.run(
            [sys.executable, "-c", "import labrag.server as s; print(s.MODEL_ID)"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "DemoRAG")

    def test_example_configuration_is_valid_json(self):
        for name in ("scope.example.json", "canonical_records.example.json"):
            with self.subTest(name=name):
                content = (ROOT / "config" / name).read_text(encoding="utf-8")
                self.assertIsInstance(json.loads(content), dict)


if __name__ == "__main__":
    unittest.main()
