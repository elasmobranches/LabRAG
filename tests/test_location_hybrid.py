from __future__ import annotations

import unittest
import inspect
from unittest.mock import patch

from labrag.ancestry import VerifiedLocation
from labrag.catalog import DocumentRecord
from labrag.drive_live import DriveItem, DriveSearchStatus
from labrag.location import LocatedFile, LocationResult, maybe_locate_hybrid, render_hybrid
from labrag.config import Settings


def indexed_result(query="Conference.pdf 어디 있어?"):
    doc = DocumentRecord("a", "Conference.pdf", "Conference.pdf", "[Workspace]/학회",
                         "문서", "2026-01-01T00:00:00Z", 1)
    return LocationResult(query, [LocatedFile(
        doc, 1 / 61, lexical_rank=1, file_url="https://drive/a"
    )])


class FakeDrive:
    def __init__(self, state="ok", delay=0):
        self.state = state
        self.delay = delay
        self.calls = 0

    async def search(self, query, *, deadline, limit):
        import asyncio
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        item = DriveItem("a", "Conference.pdf", "application/pdf", ("p",),
                         "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z",
                         "https://drive/a")
        return ([item] if self.state == "ok" else []), DriveSearchStatus(self.state)


class FakeAncestry:
    def __init__(self, inside=True):
        self.inside = inside

    async def verify(self, item, *, deadline):
        return VerifiedLocation(self.inside, f"ResearchWorkspace/{item.name}", ("p",))


class SlowAncestry(FakeAncestry):
    async def verify(self, item, *, deadline):
        import asyncio
        await asyncio.sleep(0.04)
        return await super().verify(item, deadline=deadline)


class HybridLocationTests(unittest.IsolatedAsyncioTestCase):
    def test_live_drive_timeout_defaults_to_five_seconds(self):
        self.assertEqual(Settings().live_drive_timeout, 5.0)
        default = inspect.signature(maybe_locate_hybrid).parameters["timeout"].default
        self.assertEqual(default, 5.0)

    async def call(self, drive, ancestry=None, *, query="Conference.pdf 어디 있어?",
                   timeout=0.2):
        with patch("labrag.location.maybe_locate", return_value=indexed_result(query)):
            return await maybe_locate_hybrid(
                query, object(), object(), None, object(),
                drive_client=drive, ancestry=ancestry or FakeAncestry(),
                timeout=timeout,
            )

    async def test_drive_success_returns_hybrid_result(self):
        result = await self.call(FakeDrive())
        self.assertIsNotNone(result.files[0].live)

    async def test_drive_timeout_returns_original_index_result(self):
        result = await self.call(FakeDrive(delay=0.2), timeout=0.01)
        self.assertIsInstance(result, LocationResult)
        self.assertEqual(result.files[0].doc.name, "Conference.pdf")

    async def test_drive_error_keeps_indexed_candidate(self):
        result = await self.call(FakeDrive(state="error"))
        self.assertEqual(result.files[0].name, "Conference.pdf")

    async def test_content_intent_never_calls_drive(self):
        drive = FakeDrive()
        result = await maybe_locate_hybrid(
            "토마토 레이블링 방법이 뭐야?", object(), object(), None, object(),
            drive_client=drive, ancestry=FakeAncestry(),
        )
        self.assertIsNone(result)
        self.assertEqual(drive.calls, 0)

    async def test_forced_location_bypasses_content_guard(self):
        drive = FakeDrive()
        with patch("labrag.location.maybe_locate", return_value=indexed_result()):
            result = await maybe_locate_hybrid(
                "링크드 리스트를 설명해줘", object(), object(), None, object(),
                drive_client=drive, ancestry=FakeAncestry(), force=True,
            )
        self.assertIsNotNone(result)
        self.assertEqual(drive.calls, 1)

    async def test_outside_root_live_result_is_discarded(self):
        result = await self.call(FakeDrive(), FakeAncestry(False))
        self.assertEqual(result.files[0].file_id, "a")
        self.assertIsNone(result.files[0].live)

    async def test_render_shows_both_sources(self):
        result = await self.call(FakeDrive())
        text = render_hybrid(result)
        self.assertIn("문서 인덱스 ✓ · Google Drive 실시간 ✓", text)
        self.assertNotIn("access_token", text)

    async def test_ancestry_candidates_are_verified_concurrently(self):
        import time
        drive = FakeDrive()
        original = drive.search
        async def many(query, *, deadline, limit):
            items, status = await original(query, deadline=deadline, limit=limit)
            return items * 10, status
        drive.search = many
        started = time.monotonic()
        await self.call(drive, SlowAncestry(), timeout=0.15)
        self.assertLess(time.monotonic() - started, 0.12)


if __name__ == "__main__":
    unittest.main()
