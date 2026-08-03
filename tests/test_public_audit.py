import tempfile
import unittest
from pathlib import Path

from tools.public_audit import scan_repository


class PublicAuditTests(unittest.TestCase):
    def scan(self, name: str, content: str, forbidden=()):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / name).write_text(content, encoding="utf-8")
            return scan_repository(root, forbidden)

    def test_flags_oauth_client_secret_assignment(self):
        assignment = "client_" + 'secret = "live-secret-value"\n'
        findings = self.scan("settings.py", assignment)
        self.assertEqual([finding.rule for finding in findings], ["credential-value"])

    def test_flags_private_network_address(self):
        address = "192" + ".168.40.12"
        findings = self.scan("compose.yml", f"API_URL=http://{address}:8100\n")
        self.assertEqual([finding.rule for finding in findings], ["private-ipv4"])

    def test_flags_caller_supplied_private_literal(self):
        findings = self.scan(
            "notes.md",
            "confidential-project-name is deployed here\n",
            forbidden=("confidential-project-name",),
        )
        self.assertEqual([finding.rule for finding in findings], ["forbidden-literal"])

    def test_allows_documented_placeholders_and_loopback(self):
        findings = self.scan(
            ".env.example",
            "SLACK_TOKEN=your-slack-token\nQDRANT_URL=http://127.0.0.1:6333\n",
        )
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
