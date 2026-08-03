from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from labrag.ancestry import AncestryVerifier
from labrag.drive_live import DriveItem


def item(id, name, parents=(), *, trashed=False):
    return DriveItem(id, name, "application/vnd.google-apps.folder",
                     tuple(parents), "", "", "", trashed)


class FakeDrive:
    def __init__(self, items):
        self.items = {x.id: x for x in items}
        self.calls = []

    async def get_item(self, file_id, *, deadline):
        self.calls.append(file_id)
        if time.monotonic() > deadline or file_id not in self.items:
            raise RuntimeError("missing")
        return self.items[file_id]


class AncestryTests(unittest.IsolatedAsyncioTestCase):
    def make(self, folders, *, ttl=24):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        drive = FakeDrive(folders)
        verifier = AncestryVerifier(
            drive, root_id="root", db_path=Path(tmp.name) / "a.sqlite",
            root_name="ResearchWorkspace", ttl_hours=ttl,
        )
        return verifier, drive

    async def test_accepts_direct_child(self):
        verifier, _ = self.make([])
        result = await verifier.verify(item("f", "Conference.pdf", ["root"]),
                                       deadline=time.monotonic() + 1)
        self.assertTrue(result.inside_root)
        self.assertEqual(result.path, "ResearchWorkspace/Conference.pdf")

    async def test_accepts_deep_descendant_and_builds_path(self):
        verifier, _ = self.make([
            item("conference", "학회", ["root"]),
            item("kroc", "2026_Conference", ["conference"]),
        ])
        result = await verifier.verify(item("f", "Conference.pdf", ["kroc"]),
                                       deadline=time.monotonic() + 1)
        self.assertTrue(result.inside_root)
        self.assertEqual(result.path,
                         "ResearchWorkspace/학회/2026_Conference/Conference.pdf")

    async def test_rejects_same_name_outside_root(self):
        verifier, _ = self.make([item("outside", "ResearchWorkspace", ["other"])])
        result = await verifier.verify(item("f", "Conference.pdf", ["outside"]),
                                       deadline=time.monotonic() + 1)
        self.assertFalse(result.inside_root)

    async def test_rejects_parent_cycle(self):
        verifier, _ = self.make([
            item("a", "a", ["b"]), item("b", "b", ["a"]),
        ])
        result = await verifier.verify(item("f", "x", ["a"]),
                                       deadline=time.monotonic() + 1)
        self.assertFalse(result.inside_root)

    async def test_cache_avoids_duplicate_get_calls(self):
        verifier, drive = self.make([item("p", "연구비", ["root"])])
        target = item("f", "내역.xlsx", ["p"])
        await verifier.verify(target, deadline=time.monotonic() + 1)
        await verifier.verify(target, deadline=time.monotonic() + 1)
        self.assertEqual(drive.calls.count("p"), 1)

    async def test_zero_ttl_revalidates_moved_folder(self):
        verifier, drive = self.make([item("p", "연구비", ["root"])], ttl=0)
        target = item("f", "내역.xlsx", ["p"])
        first = await verifier.verify(target, deadline=time.monotonic() + 1)
        drive.items["p"] = item("p", "옮긴폴더", ["root"])
        second = await verifier.verify(target, deadline=time.monotonic() + 1)
        self.assertNotEqual(first.path, second.path)


if __name__ == "__main__":
    unittest.main()
