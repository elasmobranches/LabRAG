from __future__ import annotations

import json
import time
import unittest
from datetime import datetime, timedelta, timezone

import httpx

from labrag.drive_live import DriveLiveClient, build_drive_queries, build_drive_query
from labrag.rclone_auth import RcloneOAuth


def oauth(*, expired: bool = False) -> RcloneOAuth:
    expiry = datetime.now(timezone.utc) + timedelta(hours=-1 if expired else 1)
    return RcloneOAuth("cid", "secret", "access", "refresh", expiry, "Bearer")


class DriveQueryTests(unittest.TestCase):
    def test_exact_filename_uses_name_contains_terms(self):
        q = build_drive_query("'Conference_2026.pdf' 어디 있어?")
        self.assertIn("name contains 'Conference_2026.pdf'", q)

    def test_business_query_uses_short_fulltext_terms(self):
        q = build_drive_query("토마토 레이블링 기준 문서 찾아줘")
        self.assertIn("fullText contains '토마토'", q)
        self.assertIn("fullText contains '레이블링'", q)

    def test_single_quote_is_drive_escaped(self):
        q = build_drive_query("team's_report.pdf 찾아줘")
        self.assertIn("team\\'s_report.pdf", q)

    def test_empty_terms_produce_trashed_only_query(self):
        self.assertEqual(build_drive_query("파일 어디 있어?"), "trashed = false")

    def test_business_query_adds_name_queries_and_fulltext_query(self):
        queries = build_drive_queries("토마토 병해충 과제 폴더 찾아줘")
        self.assertTrue(any("name contains '토마토'" in q for q in queries))
        self.assertTrue(any("name contains '병해충'" in q for q in queries))
        self.assertTrue(any("fullText contains" in q for q in queries))

    def test_two_artifact_terms_add_combined_name_query(self):
        queries = build_drive_queries("2026년에 구매한 라즈베리파이 영수증 중 최신 자료")
        self.assertTrue(any(
            "name contains '라즈베리파이' and name contains '영수증'" in q
            for q in queries
        ))

    def test_combined_name_query_uses_four_specific_terms(self):
        queries = build_drive_queries("오이 온실 이미지 생성 실험 결과 문서 찾아줘")
        self.assertTrue(any(
            "name contains '오이' and name contains '온실'"
            " and name contains '이미지' and name contains '생성'" in q
            for q in queries
        ))

    def test_journal_query_adds_english_synonyms(self):
        queries = build_drive_queries("투고할 만한 저널 후보 목록")
        self.assertTrue(any("name contains 'journal'" in q for q in queries))

    def test_journal_query_adds_journal_list_combination(self):
        queries = build_drive_queries("투고할 만한 저널 후보 목록")
        self.assertTrue(any(
            "name contains 'journal' and name contains 'list'" in q
            for q in queries
        ))

    def test_latest_data_query_combines_specific_artifact_terms(self):
        queries = build_drive_queries(
            "2026년에 가장 최근 업데이트된 토마토 해충 데이터 수집 문서"
        )
        self.assertTrue(any(
            "name contains '토마토' and name contains '해충'"
            " and name contains '데이터' and name contains '수집'" in q
            for q in queries
        ))

    def test_write_target_query_strips_particles_and_request_words(self):
        queries = build_drive_queries(
            "연구비를 사용한 뒤 사용 내역은 어느 문서에 작성해야 해"
        )
        self.assertTrue(any(
            "name contains '연구비' and name contains '사용'"
            " and name contains '내역'" in q
            for q in queries
        ))

    def test_query_fanout_is_bounded(self):
        queries = build_drive_queries(
            "2026년에 구매한 라즈베리파이 영수증 중 최신 자료"
        )
        self.assertLessEqual(len(queries), 5)


class DriveLiveTests(unittest.IsolatedAsyncioTestCase):
    async def make_client(self, handler, *, expired=False):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http.aclose)
        return DriveLiveClient(oauth(expired=expired), http)

    async def test_refreshes_expired_token_once(self):
        calls = []
        async def handler(request):
            calls.append(str(request.url))
            if str(request.url).startswith("https://oauth2.googleapis.com"):
                return httpx.Response(200, json={"access_token": "new", "expires_in": 3600})
            self.assertEqual(request.headers["Authorization"], "Bearer new")
            return httpx.Response(200, json={"files": []})
        client = await self.make_client(handler, expired=True)
        items, status = await client.search("토마토", deadline=time.monotonic() + 1)
        self.assertEqual(status.state, "ok")
        self.assertEqual(sum("oauth2" in x for x in calls), 1)

    async def test_search_excludes_trashed_items(self):
        async def handler(request):
            self.assertIn("fields", request.url.params)
            return httpx.Response(200, json={"files": [
                {"id": "a", "name": "ok", "parents": ["p"]},
                {"id": "b", "name": "gone", "trashed": True},
            ]})
        client = await self.make_client(handler)
        items, status = await client.search("토마토", deadline=time.monotonic() + 1)
        self.assertEqual([x.id for x in items], ["a"])
        self.assertEqual(status.state, "ok")

    async def test_search_preserves_successful_batches_when_one_query_fails(self):
        async def handler(request):
            query = request.url.params.get("q", "")
            if "fullText" in query:
                return httpx.Response(400, text="bad query")
            return httpx.Response(200, json={"files": [
                {"id": "kept", "name": "토마토 병해충", "parents": ["p"]},
            ]})
        client = await self.make_client(handler)
        items, status = await client.search(
            "토마토 병해충 과제 폴더 찾아줘", deadline=time.monotonic() + 1
        )
        self.assertEqual(status.state, "ok")
        self.assertEqual([x.id for x in items], ["kept"])

    async def test_search_stops_after_nonempty_priority_query(self):
        calls = 0
        async def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"files": [
                {"id": "exact", "name": "오이 온실 이미지 생성 실험",
                 "parents": ["p"]},
            ]})
        client = await self.make_client(handler)
        items, status = await client.search(
            "오이 온실 이미지 생성 실험 결과 문서", deadline=time.monotonic() + 1
        )
        self.assertEqual(status.state, "ok")
        self.assertEqual([x.id for x in items], ["exact"])
        self.assertEqual(calls, 1)

    async def test_429_retries_once_within_deadline(self):
        n = 0
        async def handler(request):
            nonlocal n
            n += 1
            return httpx.Response(429 if n == 1 else 200, json={"files": []})
        client = await self.make_client(handler)
        _, status = await client.search("토마토.pdf 찾아줘", deadline=time.monotonic() + 1)
        self.assertEqual(status.state, "ok")
        self.assertEqual(n, 2)

    async def test_deadline_returns_timeout_status(self):
        client = await self.make_client(lambda request: httpx.Response(200, json={"files": []}))
        _, status = await client.search("토마토", deadline=time.monotonic() - 1)
        self.assertEqual(status.state, "timeout")

    async def test_authorization_header_never_appears_in_error_detail(self):
        client = await self.make_client(lambda request: httpx.Response(400, text="Bearer access"))
        _, status = await client.search("토마토", deadline=time.monotonic() + 1)
        self.assertEqual(status.state, "error")
        self.assertNotIn("access", status.detail)


if __name__ == "__main__":
    unittest.main()
