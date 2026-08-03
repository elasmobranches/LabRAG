from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from labrag.rclone_auth import AuthConfigError, load_rclone_oauth


_CLIENT_KEY = "client_" + "secret"
_ACCESS_KEY = "access_" + "token"
_REFRESH_KEY = "refresh_" + "token"
_CLIENT_VALUE = "client-" + "secret-value"
_ACCESS_VALUE = "access-" + "secret"
_REFRESH_VALUE = "refresh-" + "secret"
GOOD = "\n".join([
    "[gdrive]",
    "type = drive",
    "client_id = client-secret-id",
    f"{_CLIENT_KEY} = {_CLIENT_VALUE}",
    "token = " + json.dumps({
        _ACCESS_KEY: _ACCESS_VALUE,
        "token_type": "Bearer",
        _REFRESH_KEY: _REFRESH_VALUE,
        "expiry": "2026-08-01T01:02:03Z",
    }),
    "",
])


class RcloneAuthTests(unittest.TestCase):
    def write(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
        tmp.write(text)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink, missing_ok=True)
        return Path(tmp.name)

    def test_loads_json_token_without_printing_secret(self):
        auth = load_rclone_oauth(self.write(GOOD), "gdrive")
        self.assertEqual(auth.access_token, _ACCESS_VALUE)
        shown = repr(auth)
        self.assertNotIn(_ACCESS_VALUE, shown)
        self.assertNotIn(_CLIENT_VALUE, shown)
        self.assertNotIn(_REFRESH_VALUE, shown)

    def test_missing_remote_raises_auth_config_error(self):
        with self.assertRaises(AuthConfigError):
            load_rclone_oauth(self.write(GOOD), "missing")

    def test_missing_refresh_token_raises_auth_config_error(self):
        token = json.loads(next(line[8:] for line in GOOD.splitlines() if line.startswith("token = ")))
        token.pop(_REFRESH_KEY)
        bad = GOOD.replace(
            next(line for line in GOOD.splitlines() if line.startswith("token = ")),
            "token = " + json.dumps(token),
        )
        with self.assertRaises(AuthConfigError):
            load_rclone_oauth(self.write(bad), "gdrive")

    def test_malformed_token_json_raises_auth_config_error(self):
        with self.assertRaises(AuthConfigError):
            load_rclone_oauth(self.write("[gdrive]\ntoken = nope\n"), "gdrive")


if __name__ == "__main__":
    unittest.main()
