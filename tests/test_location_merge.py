from __future__ import annotations

import unittest

from labrag.ancestry import VerifiedLocation
from labrag.catalog import DocumentRecord
from labrag.drive_live import DriveItem, DriveSearchStatus
from labrag.location import LocatedFile, LocationResult
from labrag.location_merge import merge_location_results


def indexed(fid, name, *, mod="2025-01-01T00:00:00Z", rank=1):
    doc = DocumentRecord(fid, name, name, "[Workspace]/학회", "문서", mod, 1)
    found = LocatedFile(doc, 1 / (60 + rank), lexical_rank=rank,
                        file_url=f"https://drive/{fid}")
    return found


def live(fid, name, *, created="2025-01-01T00:00:00Z",
         modified="2025-01-01T00:00:00Z", path=None, trashed=False):
    item = DriveItem(fid, name, "application/pdf", ("p",), created, modified,
                     f"https://drive/{fid}", trashed)
    loc = VerifiedLocation(True, path or f"ResearchWorkspace/{name}", ("p",))
    return item, loc


class LocationMergeTests(unittest.TestCase):
    def merge(self, query, indexed_files=(), live_files=(), intent="location",
              state="ok"):
        return merge_location_results(
            query, intent, LocationResult(query, list(indexed_files)),
            list(live_files), DriveSearchStatus(state),
        )

    def test_deduplicates_by_drive_id_and_marks_both(self):
        result = self.merge("Conference.pdf 어디", [indexed("a", "Conference.pdf")],
                            [live("a", "Conference.pdf")])
        self.assertEqual(len(result.files), 1)
        self.assertIn("문서 인덱스 ✓", result.files[0].provenance)
        self.assertIn("Google Drive 실시간 ✓", result.files[0].provenance)

    def test_exact_filename_outranks_semantic_match(self):
        result = self.merge("'Conference.pdf' 찾아줘",
                            [indexed("b", "Conference 포스터.pdf", rank=1)],
                            [live("a", "Conference.pdf")])
        self.assertEqual(result.files[0].file_id, "a")

    def test_drive_only_item_is_marked_not_indexed(self):
        result = self.merge("새 문서", live_files=[live("a", "새문서.pdf")])
        self.assertIn("미반영", result.files[0].provenance)

    def test_index_only_item_survives_timeout_unchanged(self):
        result = self.merge("문서", [indexed("a", "문서.pdf")], state="timeout")
        self.assertEqual(result.files[0].file_id, "a")
        self.assertIsNone(result.files[0].live)

    def test_live_metadata_wins_for_moved_file(self):
        result = self.merge("문서", [indexed("a", "옛이름.pdf")],
                            [live("a", "새이름.pdf", path="[Workspace]/새폴더/새이름.pdf")])
        self.assertEqual(result.files[0].name, "새이름.pdf")
        self.assertEqual(result.files[0].path, "[Workspace]/새폴더/새이름.pdf")

    def test_same_name_different_ids_remain_separate(self):
        result = self.merge("문서", [indexed("a", "문서.pdf")],
                            [live("b", "문서.pdf")])
        self.assertEqual(len(result.files), 2)

    def test_generic_latest_sorts_modified_time(self):
        result = self.merge("가장 최근 문서", live_files=[
            live("a", "A", modified="2026-01-01T00:00:00Z"),
            live("b", "B", modified="2026-07-01T00:00:00Z"),
        ])
        self.assertEqual(result.files[0].file_id, "b")

    def test_written_wording_sorts_created_time(self):
        result = self.merge("가장 최근 작성된 문서", live_files=[
            live("a", "A", created="2026-07-01T00:00:00Z", modified="2026-07-01T00:00:00Z"),
            live("b", "B", created="2026-08-01T00:00:00Z", modified="2026-01-01T00:00:00Z"),
        ])
        self.assertEqual(result.files[0].file_id, "b")

    def test_year_and_month_filter_before_sort(self):
        result = self.merge("2026년 7월 최신 문서", live_files=[
            live("a", "A", modified="2026-07-01T00:00:00Z"),
            live("b", "B", modified="2026-08-01T00:00:00Z"),
        ])
        self.assertEqual([x.file_id for x in result.files], ["a"])

    def test_year_without_recency_keeps_file_named_for_year(self):
        result = self.merge("2023년 연구비 내역 시트", [
            indexed("a", "2023 연구비 내역.xlsx", mod="2024-03-01T00:00:00Z")
        ])
        self.assertEqual(result.files[0].file_id, "a")

    def test_latest_year_keeps_file_with_full_year_in_name(self):
        result = self.merge("2024년도 연구업무편람 중 가장 최신 파일", live_files=[
            live("expected", "[연구업무편람]2024학년도.pdf",
                 modified="2025-01-01T00:00:00Z"),
            live("other", "일반 문서.pdf", modified="2024-12-01T00:00:00Z"),
        ])
        self.assertEqual(result.files[0].file_id, "expected")

    def test_latest_year_keeps_file_with_abbreviated_year_prefix(self):
        result = self.merge("2026년에 구매한 라즈베리파이 영수증 중 최신 자료", live_files=[
            live("expected", "260407_물품구입_라즈베리파이_구매영수증.pdf",
                 modified="2025-01-01T00:00:00Z"),
            live("other", "2026 일반 영수증.pdf",
                 modified="2026-07-01T00:00:00Z"),
        ])
        self.assertEqual(result.files[0].file_id, "expected")

    def test_folder_question_prioritizes_folder_mime(self):
        base_item, base_loc = live("f", "2026_Conference")
        folder = (DriveItem(
            base_item.id, base_item.name, "application/vnd.google-apps.folder",
            base_item.parents, base_item.created_time, base_item.modified_time,
            base_item.web_view_link,
        ), base_loc)
        result = self.merge("Conference 자료가 들어 있는 폴더 어디야",
                            live_files=[live("a", "Conference.pdf"), folder])
        self.assertEqual(result.files[0].file_id, "f")


if __name__ == "__main__":
    unittest.main()
