from __future__ import annotations

import importlib
import importlib.util
import unittest
from datetime import date


def intent_module():
    return importlib.import_module("labrag.intent")


class QueryIntentTests(unittest.TestCase):
    def test_intent_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("labrag.intent"))

    def test_parses_explicit_and_implicit_months(self):
        parse = intent_module().parse_query_intent
        explicit = parse(
            "2026년 6월 토마토 업무", today=date(2026, 7, 31)
        )
        implicit = parse(
            "6월 토마토 업무", today=date(2026, 7, 31)
        )
        expected = (date(2026, 6, 1), date(2026, 7, 1))
        self.assertEqual((explicit.period.start, explicit.period.end), expected)
        self.assertEqual(implicit.period, explicit.period)
        self.assertTrue(implicit.year_inferred)
        self.assertEqual(
            implicit.retrieval_query,
            "2026년 6월 토마토 업무",
        )

    def test_parses_relative_periods_in_korean_time(self):
        parse = intent_module().parse_query_intent
        last_month = parse("지난달 토마토 업무", today=date(2026, 7, 31))
        two_weeks = parse("최근 2주 토마토 업무", today=date(2026, 7, 31))
        ten_days = parse("최근 10일 토마토 업무", today=date(2026, 7, 31))
        self.assertEqual(
            (last_month.period.start, last_month.period.end),
            (date(2026, 6, 1), date(2026, 7, 1)),
        )
        self.assertEqual(
            (two_weeks.period.start, two_weeks.period.end),
            (date(2026, 7, 18), date(2026, 8, 1)),
        )
        self.assertEqual(
            (ten_days.period.start, ten_days.period.end),
            (date(2026, 7, 22), date(2026, 8, 1)),
        )

    def test_parses_sources_topic_person_channel_and_output(self):
        intent = intent_module().parse_query_intent(
            "2026년 6월 홍길동 님의 토마토 업무를 "
            "Google Drive와 Slack #crop-imaging에서 "
            "담당자별 표로 정리해줘",
            today=date(2026, 7, 31),
        )
        self.assertEqual(intent.sources, ("drive", "slack"))
        self.assertIn("토마토", intent.topics)
        self.assertEqual(intent.people, ("홍길동",))
        self.assertEqual(intent.channels, ("crop-imaging",))
        self.assertEqual(intent.output_format, "table_by_person")
        self.assertEqual(intent.task, "content")

    def test_distinguishes_location_and_content_tasks(self):
        parse = intent_module().parse_query_intent
        self.assertEqual(parse("Conference_2026.pdf 어디 있어?").task, "location")
        self.assertEqual(
            parse("토마토 레이블링 가이드 내용을 요약해줘").task,
            "content",
        )

    def test_unknown_fields_fall_back_without_failure(self):
        intent = intent_module().parse_query_intent("지난번 그거 어떻게 됐어?")
        self.assertIsNone(intent.period)
        self.assertEqual(intent.sources, ())
        self.assertEqual(intent.channels, ())
        self.assertEqual(intent.people, ())
        self.assertEqual(intent.raw_query, "지난번 그거 어떻게 됐어?")


if __name__ == "__main__":
    unittest.main()
