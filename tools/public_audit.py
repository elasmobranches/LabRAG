"""Scan a public export without echoing the sensitive value that triggered it."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(?:client_secret|access_token|refresh_token|password|passwd|api_key)"
    r"\s*[:=]\s*[\"']?"
    r"(?!your[-_]|example[-_]|changeme\b|<[^>]+>)"
    r"[^\s\"'#]{8,}"
)
_PRIVATE_IPV4 = re.compile(
    r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
    r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
)
_SECRET_FILENAME = re.compile(r"(?i)(?:^|/)client_secret_[^/]+\.json$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str


def _text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        yield path


def scan_repository(
    root: Path, forbidden_literals: Sequence[str] = ()
) -> list[Finding]:
    """Return file and rule metadata only; never return the matched content."""
    root = root.resolve()
    findings: list[Finding] = []
    forbidden = tuple(value for value in forbidden_literals if value.strip())

    for path in _text_files(root):
        relative = path.relative_to(root).as_posix()
        if _SECRET_FILENAME.search(relative):
            findings.append(Finding(relative, 0, "oauth-secret-file"))

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if _CREDENTIAL_ASSIGNMENT.search(line):
                findings.append(Finding(relative, line_number, "credential-value"))
            if _PRIVATE_IPV4.search(line):
                findings.append(Finding(relative, line_number, "private-ipv4"))
            if any(value in line for value in forbidden):
                findings.append(Finding(relative, line_number, "forbidden-literal"))

    return findings


def _load_forbidden(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return ()
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--forbidden-file", type=Path)
    args = parser.parse_args()

    findings = scan_repository(
        Path(args.root), _load_forbidden(args.forbidden_file)
    )
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.rule}")
    print(f"public audit: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
