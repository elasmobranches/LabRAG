from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from labrag import rag as rag_module
from labrag.rag import (
    Retrieved,
    _should_force_most_recent,
    _is_generic_slack_recency,
    _redact_slack_secrets,
    _sanitize_slack_hits,
    retrieve,
)
from labrag.store import Hit


_extract_month_period = getattr(rag_module, "_extract_month_period", None)
_filter_hits_by_period = getattr(rag_module, "_filter_hits_by_period", None)
_normalize_period_query = getattr(rag_module, "_normalize_period_query", None)
_slack_permalink = getattr(rag_module, "_slack_permalink", None)


def drive_hit(text="QA transcript"):
    return Hit(0.8, {
        "text": text,
        "citation": "QA Discussion Meeting #2 (May 28).txt",
        "file_url": "https://drive/qa",
        "file_id": "drive-qa",
        "root": "[Workspace]/OnlineLectureSeminar",
        "mod_time": "2026-01-01T00:00:00Z",
    })


def slack_hit(thread_ts: str, text: str, channel="프로젝트운영", index=0):
    return Hit(0.8, {
        "text": text,
        "citation": f"Slack · #{channel} · thread {thread_ts}",
        "file_url": "",
        "file_id": f"slack:C1:{thread_ts}:{index}",
        "source": "slack",
        "channel_id": "C1",
        "channel_name": channel,
        "thread_ts": thread_ts,
        "index": index,
    })


def _source_name(hit):
    if hit.payload.get("source") in ("slack", "slack_parent"):
        return "slack"
    return "drive"


@dataclass
class Rank:
    index: int
    score: float


class FakeModels:
    def embed_one(self, query):
        return [0.1]

    def rerank(self, query, texts, instruct=None):
        return [Rank(i, 1.0 - i * 0.01) for i in range(len(texts))]


class FakeDriveStore:
    def search_grouped(self, vector, groups, per_file, roots=None):
        return [drive_hit()]

    def search(self, vector, limit, roots=None):
        return [drive_hit()]

    def latest_file_in_root(self, root):
        return {"file_id": "drive-qa"}

    def chunks_of_file(self, file_id, limit=100):
        return [drive_hit(f"QA chunk {i}") for i in range(6)]


class FakeSlackStore:
    def __init__(self):
        self.hits = [
            slack_hit("100.0", "오래된 Slack 내용", channel="다른채널"),
            slack_hit("300.0", "가장 최근 Slack 내용", index=0),
            slack_hit("300.0", "같은 최신 스레드의 두 번째 청크", index=1),
            slack_hit("200.0", "두 번째로 최근 Slack 내용"),
        ]

    def search(self, vector, limit, roots=None):
        return self.hits[:limit]

    def channel_names(self):
        return ("다른채널", "프로젝트운영")

    def latest_slack_threads(self, limit=50):
        return [
            slack_hit("500.0", "<@U123> 님이 채널에 참여함"),
            slack_hit("450.0", "접속 비밀번호는 qwer1234입니다"),
            slack_hit("440.0", "서버 접속 (pw : qwer5678)"),
            slack_hit("400.0", "전체 색인의 실제 최신 업무 내용"),
            *self.hits,
        ][:limit]


class NoisySlackStore(FakeSlackStore):
    def __init__(self):
        self.hits = [
            slack_hit("600.0", "<@U123> 님이 채널에 참여함"),
            slack_hit(
                "590.0",
                "*manager* 님이 이 채널에 *partner* 님을 추가했습니다. "
                "채널 세부정보에서 권한을 검토할 수 있습니다.",
            ),
            slack_hit("580.0", "<@U456> 님이 채널을 떠남"),
            slack_hit(
                "570.0",
                "*Agtech Research*의 초대로 인해 *partner*이(가) "
                "이 채널에 참여했습니다.",
            ),
            slack_hit(
                "560.0",
                "*partner*이(가) 이 채널에서 자신을 제거했습니다.",
            ),
            slack_hit("550.0", "병해충 과제 미팅 회의록을 공유했습니다."),
        ]

    def latest_slack_threads(self, limit=50):
        return self.hits[:limit]


class ManyDriveStore(FakeDriveStore):
    def search_grouped(self, vector, groups, per_file, roots=None):
        hits = []
        for index in range(4):
            hit = drive_hit(f"Drive 토마토 기록 {index}")
            hit.payload["file_id"] = f"drive-{index}"
            hit.payload["citation"] = f"Drive 기록 {index}.docx"
            hits.append(hit)
        return hits


class SlackRecencyTests(unittest.TestCase):
    def retrieve(self, query):
        return retrieve(
            query,
            FakeModels(),
            FakeDriveStore(),
            candidates=50,
            top_k=6,
            grouped=True,
            expand_context=False,
            slack_store=FakeSlackStore(),
        )

    def test_recent_slack_query_keeps_slack_and_sorts_thread_ts(self):
        result = self.retrieve("최근 slack 내용 뭐있는지 알 수있나")
        self.assertTrue(result.hits)
        self.assertTrue(all(h.payload.get("source") == "slack" for h in result.hits))
        self.assertEqual(
            [h.payload["thread_ts"] for h in result.hits],
            ["450.0", "440.0", "400.0", "300.0", "200.0", "100.0"],
        )
        self.assertNotIn("채널에 참여함", "\n".join(h.text for h in result.hits))
        text = "\n".join(h.text for h in result.hits)
        self.assertNotIn("qwer1234", text)
        self.assertNotIn("qwer5678", text)
        self.assertIn("[REDACTED]", text)

    def test_recent_slack_query_returns_each_thread_once(self):
        result = self.retrieve("최근 Slack 내용 알려줘")
        keys = [
            (h.payload["channel_id"], h.payload["thread_ts"])
            for h in result.hits
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_topic_slack_recency_filters_membership_events(self):
        result = retrieve(
            "Slack에서 최근 업무 내용을 채널별로 요약해줘",
            FakeModels(),
            FakeDriveStore(),
            candidates=50,
            top_k=6,
            grouped=True,
            expand_context=False,
            slack_store=NoisySlackStore(),
        )

        self.assertEqual(
            [hit.text for hit in result.hits],
            ["병해충 과제 미팅 회의록을 공유했습니다."],
        )

    def test_relative_periods_share_the_retrieval_date_filter(self):
        last_month = _extract_month_period(
            "지난달 토마토 업무", today=date(2026, 7, 31)
        )
        two_weeks = _extract_month_period(
            "최근 2주 토마토 업무", today=date(2026, 7, 31)
        )
        self.assertEqual(
            (last_month.start, last_month.end),
            (date(2026, 6, 1), date(2026, 7, 1)),
        )
        self.assertEqual(
            (two_weeks.start, two_weeks.end),
            (date(2026, 7, 18), date(2026, 8, 1)),
        )

    def test_relative_month_query_is_normalized_for_semantic_search(self):
        period = _extract_month_period(
            "지난달 토마토 업무", today=date(2026, 7, 31)
        )
        self.assertEqual(
            _normalize_period_query("지난달 토마토 업무", period),
            "2026년 6월 지난달 토마토 업무",
        )

    def test_explicit_slack_query_excludes_drive_without_recency(self):
        result = self.retrieve("Slack에서 어떤 이야기를 했어?")
        self.assertTrue(result.hits)
        self.assertTrue(all(h.payload.get("source") == "slack" for h in result.hits))

    def test_explicit_drive_query_excludes_slack(self):
        """Drive만 지정한 질문에 Slack 근거가 섞이면 출처 계약을 어긴다."""
        result = self.retrieve("구글 드라이브에 있는 토마토 논문 내용을 알려줘")

        self.assertTrue(result.hits)
        self.assertTrue(all(
            hit.payload.get("source") not in ("slack", "slack_parent")
            for hit in result.hits
        ))

    def test_explicit_channel_filters_slack_results(self):
        result = self.retrieve(
            "Slack #프로젝트운영에서 최근 업무를 담당자별로 정리해줘"
        )
        self.assertTrue(result.hits)
        self.assertTrue(all(
            hit.payload.get("channel_name") == "프로젝트운영"
            for hit in result.hits
        ))

    def test_explicit_channel_without_word_slack_excludes_drive(self):
        """`#채널` 자체가 Slack 출처 지시이므로 Drive 문서가 섞이면 안 된다."""
        result = self.retrieve("#프로젝트운영 채널 업무 내용 알려줘")

        self.assertTrue(result.hits)
        self.assertTrue(all(
            hit.payload.get("source") == "slack" for hit in result.hits
        ))

    def test_channel_name_written_with_a_space_still_finds_the_channel(self):
        """실제 이름은 '프로젝트운영'인데 사람은 '#프로젝트 운영'이라고 쓴다.

        예전에는 이 한 칸 때문에 채널 필터가 Slack 결과를 전부 버려 0건이 됐다.
        """
        result = self.retrieve("슬랙의 #프로젝트 운영 채널에서 최근 내용 알려줘")

        self.assertTrue(result.hits)
        self.assertTrue(all(
            hit.payload.get("channel_name") == "프로젝트운영"
            for hit in result.hits
        ))

    def test_unknown_channel_falls_back_instead_of_returning_nothing(self):
        """없는 채널을 대도 조용히 0건으로 끝나면 사용자는 채널이 비었다고 오해한다."""
        result = self.retrieve("슬랙의 #없는채널 에서 최근 내용 알려줘")

        self.assertTrue(result.hits)
        self.assertTrue(any("채널" in note for note in result.notes))

    def test_natural_generic_recent_slack_phrases_use_global_latest(self):
        for query in (
            "최근 슬랙에서 뭐했어?",
            "최근 Slack 대화가 뭐야",
            "최신 슬랙 메시지 보여줘",
        ):
            with self.subTest(query=query):
                self.assertTrue(_is_generic_slack_recency(query))
        self.assertFalse(_is_generic_slack_recency("최근 슬랙 참독 개발 내용"))

    def test_recent_trip_participant_question_keeps_relevance_ranking(self):
        query = "최근에 부산 갔다 온 사람이 있을까? 당시 참여자들의 담당 업무는?"
        self.assertFalse(_should_force_most_recent(query))
        result = self.retrieve(query)
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].text, "QA transcript")

    def test_parent_context_credentials_are_redacted(self):
        hit = slack_hit("300.0", "본문")
        secret = "secret-" + "value"
        hit.payload["_parent_context"] = "서버 pass" + "word: " + secret
        _sanitize_slack_hits([hit])
        self.assertNotIn(secret, hit.payload["_parent_context"])
        self.assertIn("[REDACTED]", hit.payload["_parent_context"])

    def test_common_slack_github_and_openai_tokens_are_redacted(self):
        tokens = (
            "xapp-1-ABCDEF",
            "github_pat_ABCDEF123456",
            "gho_ABCDEF123456",
            "sk-ABCDEFGHIJKLMNOP",
        )
        redacted = _redact_slack_secrets(" ".join(tokens))
        for token in tokens:
            self.assertNotIn(token, redacted)

    def test_common_unseparated_password_labels_are_redacted(self):
        samples = (
            ("password is secret-one", "secret-one"),
            ("비밀번호 secret-two", "secret-two"),
            ("비번은 secret-three", "secret-three"),
        )
        for text, secret in samples:
            with self.subTest(text=text):
                self.assertNotIn(secret, _redact_slack_secrets(text))

    def test_slack_permalink_uses_channel_and_root_thread_timestamp(self):
        self.assertTrue(callable(_slack_permalink))
        self.assertEqual(
            _slack_permalink("C07RXJNFGAC", "1785399154.691819"),
            "https://slack.com/archives/C07RXJNFGAC/p1785399154691819",
        )

    def test_slack_permalink_rejects_invalid_identifiers(self):
        self.assertTrue(callable(_slack_permalink))
        self.assertEqual(_slack_permalink("C1", "1785399154.691819"), "")
        self.assertEqual(_slack_permalink("C07RXJNFGAC", "not-a-ts"), "")

    def test_sanitized_slack_hit_renders_clickable_source(self):
        hit = slack_hit("1785399154.691819", "업무 기록")
        hit.payload["channel_id"] = "C07RXJNFGAC"
        _sanitize_slack_hits([hit])
        expected = "https://slack.com/archives/C07RXJNFGAC/p1785399154691819"
        self.assertEqual(hit.payload["file_url"], expected)
        sources = Retrieved([hit], [1.0], "질문", 1).sources_block()
        self.assertIn(f"]({expected})", sources)

    def test_slack_permalink_preserves_existing_url_and_drive_url(self):
        slack = slack_hit("1785399154.691819", "업무 기록")
        slack.payload["channel_id"] = "C07RXJNFGAC"
        slack.payload["file_url"] = "https://workspace.slack.com/custom"
        drive = drive_hit()
        original_drive_url = drive.payload["file_url"]
        _sanitize_slack_hits([slack, drive])
        self.assertEqual(
            slack.payload["file_url"],
            "https://workspace.slack.com/custom",
        )
        self.assertEqual(drive.payload["file_url"], original_drive_url)

    def test_explicit_slack_skips_drive_canonical_early_return(self):
        canonical = Retrieved([drive_hit()], [1.0], "질문", 1)
        with patch(
            "labrag.rag.detect_canonical_intent",
            return_value=SimpleNamespace(operation="file"),
        ), patch(
            "labrag.rag._retrieve_canonical", return_value=canonical
        ), patch(
            "labrag.rag._lexical_candidates", return_value=[]
        ), patch(
            "labrag.rag._unindexed_notice", return_value=None
        ):
            result = retrieve(
                "Slack에서 연구실 명단 얘기한 내용",
                FakeModels(),
                FakeDriveStore(),
                expand_context=False,
                catalog=object(),
                canonical_config={"record": object()},
                slack_store=FakeSlackStore(),
            )
        self.assertTrue(result.hits)
        self.assertTrue(all(h.payload.get("source") == "slack" for h in result.hits))

    def test_explicit_slack_does_not_append_drive_unindexed_notice(self):
        with patch("labrag.rag._lexical_candidates", return_value=[]), patch(
            "labrag.rag._unindexed_notice", return_value=drive_hit("Drive notice")
        ):
            result = retrieve(
                "Slack에서 어떤 이야기를 했어?",
                FakeModels(),
                FakeDriveStore(),
                expand_context=False,
                catalog=object(),
                slack_store=FakeSlackStore(),
            )
        self.assertTrue(result.hits)
        self.assertTrue(all(h.payload.get("source") == "slack" for h in result.hits))

    def test_explicit_drive_and_slack_request_keeps_both_sources(self):
        result = retrieve(
            "토마토 업무 기록을 구글드라이브와 슬랙에서 전부 찾아줘",
            FakeModels(),
            ManyDriveStore(),
            candidates=50,
            top_k=2,
            grouped=True,
            expand_context=False,
            slack_store=FakeSlackStore(),
        )
        sources = {
            "slack"
            if hit.payload.get("source") in ("slack", "slack_parent")
            else "drive"
            for hit in result.hits
        }
        self.assertEqual(sources, {"drive", "slack"})

    def test_recent_drive_and_slack_request_keeps_both_sources(self):
        result = retrieve(
            "최근 토마토 업무를 구글드라이브와 슬랙에서 모두 찾아줘",
            FakeModels(),
            ManyDriveStore(),
            candidates=50,
            top_k=2,
            grouped=True,
            expand_context=False,
            slack_store=FakeSlackStore(),
        )
        self.assertEqual(
            {_source_name(hit) for hit in result.hits},
            {"drive", "slack"},
        )

    def test_drive_and_slack_request_skips_drive_only_canonical_return(self):
        canonical = Retrieved([drive_hit()], [1.0], "질문", 1)
        with patch(
            "labrag.rag.detect_canonical_intent",
            return_value=SimpleNamespace(operation="file"),
        ), patch(
            "labrag.rag._retrieve_canonical", return_value=canonical
        ), patch(
            "labrag.rag._lexical_candidates", return_value=[]
        ), patch(
            "labrag.rag._unindexed_notice", return_value=None
        ):
            result = retrieve(
                "연구실 명단을 구글드라이브와 슬랙에서 모두 찾아줘",
                FakeModels(),
                ManyDriveStore(),
                top_k=2,
                expand_context=False,
                catalog=object(),
                canonical_config={"record": object()},
                slack_store=FakeSlackStore(),
            )
        self.assertEqual(
            {_source_name(hit) for hit in result.hits},
            {"drive", "slack"},
        )


class MonthPeriodFilterTests(unittest.TestCase):
    def test_period_filter_api_exists(self):
        self.assertTrue(callable(_extract_month_period))
        self.assertTrue(callable(_filter_hits_by_period))
        self.assertTrue(callable(_normalize_period_query))

    def test_adds_current_year_to_month_only_retrieval_query(self):
        period = _extract_month_period(
            "06월 중 토마토 업무", today=date(2026, 7, 31)
        )
        self.assertEqual(
            _normalize_period_query("06월 중 토마토 업무", period),
            "2026년 06월 중 토마토 업무",
        )
        explicit = "2026년 06월 중 토마토 업무"
        self.assertEqual(
            _normalize_period_query(explicit, period),
            explicit,
        )

    def test_extracts_explicit_and_current_year_month_periods(self):
        explicit = _extract_month_period(
            "2026년 06월 토마토 업무", today=date(2026, 7, 31)
        )
        implicit = _extract_month_period(
            "06월 중 토마토 업무", today=date(2026, 7, 31)
        )
        self.assertEqual((explicit.start, explicit.end), (
            date(2026, 6, 1), date(2026, 7, 1)
        ))
        self.assertEqual(implicit, explicit)
        self.assertIsNone(
            _extract_month_period("최근 토마토 업무", today=date(2026, 7, 31))
        )

    def test_filters_slack_threads_by_korean_calendar_month(self):
        kst = ZoneInfo("Asia/Seoul")
        inside = slack_hit(
            str(datetime(2026, 6, 2, 9, tzinfo=kst).timestamp()),
            "2026년 6월 Slack 업무",
        )
        outside = slack_hit(
            str(datetime(2025, 6, 2, 9, tzinfo=kst).timestamp()),
            "2025년 6월 Slack 업무",
        )
        period = _extract_month_period(
            "2026년 6월 업무", today=date(2026, 7, 31)
        )
        self.assertEqual(_filter_hits_by_period([outside, inside], period), [inside])

    def test_drive_filename_date_overrides_modification_time(self):
        old_named = drive_hit("과거 기록")
        old_named.payload.update({
            "citation": "2025-06-02.docx",
            "mod_time": "2026-06-20T10:00:00Z",
        })
        current_named = drive_hit("현재 기록")
        current_named.payload.update({
            "citation": "260616_개인미팅.docx",
            "mod_time": "2026-07-20T10:00:00Z",
        })
        period = _extract_month_period(
            "2026년 6월 업무", today=date(2026, 7, 31)
        )
        self.assertEqual(
            _filter_hits_by_period([old_named, current_named], period),
            [current_named],
        )

    def test_drive_undated_filename_uses_modification_time(self):
        inside = drive_hit("기간 내 기록")
        inside.payload.update({
            "citation": "토마토 업무.docx",
            "mod_time": "2026-06-20T10:00:00Z",
        })
        outside = drive_hit("기간 밖 기록")
        outside.payload.update({
            "citation": "토마토 업무 백업.docx",
            "mod_time": "2026-07-01T00:00:00Z",
        })
        period = _extract_month_period(
            "2026년 6월 업무", today=date(2026, 7, 31)
        )
        self.assertEqual(
            _filter_hits_by_period([outside, inside], period),
            [inside],
        )

    def test_retrieve_applies_period_before_balancing_sources(self):
        kst = ZoneInfo("Asia/Seoul")

        class PeriodDriveStore(ManyDriveStore):
            def search_grouped(self, vector, groups, per_file, roots=None):
                outside = drive_hit("2025년 기록")
                outside.payload.update({
                    "file_id": "drive-outside",
                    "citation": "2025-06-02.docx",
                })
                inside = drive_hit("2026년 기록")
                inside.payload.update({
                    "file_id": "drive-inside",
                    "citation": "260616_개인미팅.docx",
                })
                return [outside, inside]

        class PeriodSlackStore(FakeSlackStore):
            def search(self, vector, limit, roots=None):
                return [
                    slack_hit(
                        str(datetime(2025, 6, 2, 9, tzinfo=kst).timestamp()),
                        "2025년 Slack 기록",
                    )
                ]

            def latest_slack_threads(self, limit=50):
                return [
                    slack_hit(
                        str(datetime(2026, 6, 2, 9, tzinfo=kst).timestamp()),
                        "2026년 Slack 기록",
                    ),
                ]

        result = retrieve(
            "2026년 6월 토마토 업무를 구글드라이브와 슬랙에서 모두 찾아줘",
            FakeModels(),
            PeriodDriveStore(),
            top_k=2,
            grouped=True,
            expand_context=False,
            slack_store=PeriodSlackStore(),
        )
        self.assertEqual({_source_name(hit) for hit in result.hits}, {
            "drive", "slack"
        })
        self.assertNotIn(
            "2025년", "\n".join(hit.text for hit in result.hits)
        )


if __name__ == "__main__":
    unittest.main()
