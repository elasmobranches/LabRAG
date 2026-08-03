"""Google Drive v3 읽기 전용 실시간 검색 클라이언트."""
from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .catalog import extract_lexical_terms
from .rclone_auth import RcloneOAuth

DRIVE_API = "https://www.googleapis.com/drive/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"
FIELDS = "files(id,name,mimeType,parents,createdTime,modifiedTime,trashed,webViewLink),nextPageToken"
_EXTS = r"(?:pdf|docx?|xlsx?|pptx?|hwp|hwpx|csv|txt|md|zip)"
_EXT_QUERY = re.compile(
    rf'"([^"]+\.{_EXTS})"|\'([^\']+\.{_EXTS})\'|(\S+\.{_EXTS})', re.I
)


def _escape_q(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def build_drive_query(user_query: str) -> str:
    exact = _EXT_QUERY.search(user_query)
    if exact:
        name = next(g for g in exact.groups() if g).strip()
        return f"trashed = false and name contains '{_escape_q(name)}'"
    terms = [
        t for t in extract_lexical_terms(user_query)
        if t not in {"있어", "있지", "어딨어", "열어"}
    ][:4]
    if not terms:
        return "trashed = false"
    clauses = " or ".join(f"fullText contains '{_escape_q(t)}'" for t in terms)
    return f"trashed = false and ({clauses})"


def build_drive_queries(user_query: str) -> list[str]:
    """정확 이름 검색과 의미 힌트 검색을 작은 Drive 쿼리 여러 개로 나눈다."""
    exact = _EXT_QUERY.search(user_query)
    if exact:
        return [build_drive_query(user_query)]
    journal_query = bool(re.search(r"저널|투고", user_query, re.I))
    terms = [
        t for t in extract_lexical_terms(user_query)
        if t not in {"있어", "있지", "어딨어", "열어", "있는", "들어", "폴더가"}
    ]
    if journal_query:
        terms = ["journal", "list", *terms]
    terms = list(dict.fromkeys(terms))
    if not terms:
        return ["trashed = false"]
    request_words = {
        "최신", "최근", "자료", "문서", "목록", "폴더", "파일",
        "가장", "뒤", "어느", "중", "결과", "정리", "후보", "후보를",
    }
    particle_suffixes = ("에서", "으로", "를", "은", "는", "가", "을", "에", "의", "로")
    name_terms = [
        t for t in terms
        if t not in request_words
        and not re.match(r"^20\d{2}년", t)
        and not t.endswith(("한", "할", "된"))
        and not any(
            t.endswith(suffix) and t[:-len(suffix)] in terms
            for suffix in particle_suffixes
        )
    ]
    name_terms = [
        term for term in name_terms
        if not any(other == term + "이" for other in name_terms)
    ]
    queries = [
        f"trashed = false and name contains '{_escape_q(term)}'"
        for term in name_terms[:3]
    ]
    if len(name_terms) >= 2:
        combined_terms = ["journal", "list"] if journal_query else name_terms[:4]
        queries.insert(
            0,
            "trashed = false and "
            + " and ".join(
                f"name contains '{_escape_q(term)}'" for term in combined_terms
            ),
        )
    queries.append(build_drive_query(user_query))
    return list(dict.fromkeys(queries))


@dataclass(frozen=True)
class DriveItem:
    id: str
    name: str
    mime_type: str
    parents: tuple[str, ...]
    created_time: str
    modified_time: str
    web_view_link: str
    trashed: bool = False

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "DriveItem":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            mime_type=str(data.get("mimeType") or ""),
            parents=tuple(data.get("parents") or ()),
            created_time=str(data.get("createdTime") or ""),
            modified_time=str(data.get("modifiedTime") or ""),
            web_view_link=str(data.get("webViewLink") or ""),
            trashed=bool(data.get("trashed", False)),
        )


@dataclass(frozen=True)
class DriveSearchStatus:
    state: str
    detail: str = ""


class DriveLiveError(RuntimeError):
    def __init__(self, state: str):
        super().__init__(state)
        self.state = state


class DriveLiveClient:
    def __init__(self, oauth: RcloneOAuth, http: httpx.AsyncClient):
        self.oauth = oauth
        self.http = http
        self._access_token = oauth.access_token
        self._expiry = oauth.expiry
        self._refresh_lock = asyncio.Lock()

    def _remaining(self, deadline: float) -> float:
        left = deadline - time.monotonic()
        if left <= 0:
            raise DriveLiveError("timeout")
        return left

    def _expired(self) -> bool:
        if self._expiry is None:
            return False
        expiry = self._expiry
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= datetime.now(timezone.utc) + timedelta(seconds=30)

    async def _refresh(self, deadline: float, force: bool = False) -> None:
        async with self._refresh_lock:
            if not force and not self._expired():
                return
            try:
                response = await self.http.post(
                    TOKEN_URL,
                    data={
                        "client_id": self.oauth.client_id,
                        "client_secret": self.oauth.client_secret,
                        "refresh_token": self.oauth.refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=self._remaining(deadline),
                )
                response.raise_for_status()
                data = response.json()
                self._access_token = data["access_token"]
                self._expiry = datetime.now(timezone.utc) + timedelta(
                    seconds=int(data.get("expires_in", 3600))
                )
            except DriveLiveError:
                raise
            except Exception as exc:
                raise DriveLiveError("auth_error") from exc

    async def _request(self, method: str, url: str, *, deadline: float,
                       params: dict[str, Any] | None = None) -> httpx.Response:
        await self._refresh(deadline)
        refreshed_401 = False
        transient_retry = False
        while True:
            try:
                response = await self.http.request(
                    method, url, params=params,
                    headers={"Authorization": f"Bearer {self._access_token}"},
                    timeout=self._remaining(deadline),
                )
            except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
                raise DriveLiveError("timeout") from exc
            except httpx.HTTPError as exc:
                raise DriveLiveError("error") from exc

            if response.status_code == 401 and not refreshed_401:
                refreshed_401 = True
                await self._refresh(deadline, force=True)
                continue
            if response.status_code in (403, 429) and not transient_retry:
                transient_retry = True
                await asyncio.sleep(min(0.03 + random.random() * 0.02,
                                        self._remaining(deadline)))
                continue
            if response.status_code >= 500 and not transient_retry:
                transient_retry = True
                await asyncio.sleep(min(0.03, self._remaining(deadline)))
                continue
            if response.status_code == 401:
                raise DriveLiveError("auth_error")
            if response.status_code in (403, 429):
                raise DriveLiveError("rate_limited")
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise DriveLiveError("error") from exc
            return response

    async def search(self, query: str, *, deadline: float,
                     limit: int = 20) -> tuple[list[DriveItem], DriveSearchStatus]:
        try:
            async def one(q: str) -> list[DriveItem]:
                response = await self._request(
                    "GET", f"{DRIVE_API}/files", deadline=deadline,
                    params={
                        "q": q, "pageSize": min(limit, 100), "fields": FIELDS,
                        "supportsAllDrives": "true",
                        "includeItemsFromAllDrives": "true",
                    },
                )
                return [DriveItem.from_api(x) for x in response.json().get("files", [])]

            queries = build_drive_queries(query)
            outcomes: list[list[DriveItem] | BaseException] = []
            try:
                priority = await one(queries[0])
                if priority:
                    outcomes.append(priority)
                else:
                    outcomes.append(priority)
                    outcomes.extend(await asyncio.gather(
                        *(one(q) for q in queries[1:]),
                        return_exceptions=True,
                    ))
            except Exception as exc:
                outcomes.append(exc)
                outcomes.extend(await asyncio.gather(
                    *(one(q) for q in queries[1:]),
                    return_exceptions=True,
                ))
            batches = [outcome for outcome in outcomes
                       if not isinstance(outcome, Exception)]
            if not batches:
                drive_errors = [
                    outcome for outcome in outcomes
                    if isinstance(outcome, DriveLiveError)
                ]
                state = drive_errors[0].state if drive_errors else "error"
                return [], DriveSearchStatus(state)
            unique: dict[str, DriveItem] = {}
            for batch in batches:
                for item in batch:
                    if item.id and not item.trashed:
                        unique.setdefault(item.id, item)
            return list(unique.values()), DriveSearchStatus("ok")
        except DriveLiveError as exc:
            return [], DriveSearchStatus(exc.state)
        except Exception:
            return [], DriveSearchStatus("error")

    async def get_item(self, file_id: str, *, deadline: float) -> DriveItem:
        response = await self._request(
            "GET", f"{DRIVE_API}/files/{file_id}", deadline=deadline,
            params={
                "fields": "id,name,mimeType,parents,createdTime,modifiedTime,trashed,webViewLink",
                "supportsAllDrives": "true",
            },
        )
        return DriveItem.from_api(response.json())
