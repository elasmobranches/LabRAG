"""기존 rclone OAuth 설정을 읽되 비밀값을 로그에 노출하지 않는다."""
from __future__ import annotations

import configparser
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class AuthConfigError(RuntimeError):
    pass


def _parse_expiry(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True, repr=False)
class RcloneOAuth:
    client_id: str
    client_secret: str
    access_token: str
    refresh_token: str
    expiry: datetime | None
    token_type: str

    def __repr__(self) -> str:
        return (
            "RcloneOAuth(client_id=<redacted>, client_secret=<redacted>, "
            f"access_token=<redacted>, refresh_token=<redacted>, expiry={self.expiry!r}, "
            f"token_type={self.token_type!r})"
        )


def load_rclone_oauth(path: Path, remote: str) -> RcloneOAuth:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with path.open("r", encoding="utf-8") as fh:
            parser.read_file(fh)
    except OSError as exc:
        raise AuthConfigError("rclone 설정 파일을 읽을 수 없음") from exc

    if remote not in parser:
        raise AuthConfigError(f"rclone remote가 없음: {remote}")
    section = parser[remote]
    try:
        token = json.loads(section.get("token", ""))
    except json.JSONDecodeError as exc:
        raise AuthConfigError("rclone OAuth token JSON이 잘못됨") from exc

    access_token = str(token.get("access_token") or "")
    refresh_token = str(token.get("refresh_token") or "")
    client_id = section.get("client_id", "").strip()
    client_secret = section.get("client_secret", "").strip()
    if not access_token or not refresh_token:
        raise AuthConfigError("rclone OAuth access/refresh token이 없음")
    if not client_id or not client_secret:
        raise AuthConfigError("rclone OAuth client_id/client_secret이 없음")

    return RcloneOAuth(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
        expiry=_parse_expiry(token.get("expiry")),
        token_type=str(token.get("token_type") or "Bearer"),
    )
