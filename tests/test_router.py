from __future__ import annotations

import unittest
from unittest.mock import patch

from labrag.config import Settings
from labrag.router import (
    classify_query,
    split_hybrid_queries,
    probe_internal_relevance,
    resolve_probe,
)


class FakeModels:
    def embed_one(self, query):
        return [0.1, 0.2]


class FakeStore:
    def __init__(self, scores=(), error=False):
        self.scores = scores
        self.error = error
        self.roots_seen = []

    def search(self, vector, limit, roots=None):
        if self.error:
            raise RuntimeError("offline")
        self.roots_seen.append(roots)
        return [type("Hit", (), {"score": score})() for score in self.scores[:limit]]


class RouterRuleTests(unittest.TestCase):
    def test_forced_mode_wins(self):
        decision = classify_query("Conference.pdf 어디 있어?", "general")
        self.assertEqual(decision.mode, "general")
        self.assertEqual(decision.reason, "forced")

    def test_forced_web_modes_are_accepted(self):
        self.assertEqual(classify_query("질문", "web").mode, "web")
        self.assertEqual(classify_query("질문", "rag_web").mode, "rag_web")

    def test_filename_location_question_uses_location(self):
        decision = classify_query("Conference_2026.pdf 어디 있어?")
        self.assertEqual(decision.mode, "location")

    def test_internal_content_question_uses_rag(self):
        self.assertEqual(
            classify_query("토마토 레이블링 가이드 내용 알려줘").mode, "rag"
        )
        self.assertEqual(
            classify_query("Slack에서 결정한 내용이 뭐야?").mode, "rag"
        )

    def test_standalone_lab_paper_questions_use_rag(self):
        for query in (
            "우리 논문 데이터셋 지적",
            "Conference 논문 요약",
        ):
            decision = classify_query(query)
            self.assertEqual(decision.mode, "rag")

    def test_trip_and_participant_questions_use_internal_rag(self):
        for query in (
            "최근에 부산 갔다 온 사람이 있을까?",
            "2월 24일 부산 출장 참여자와 담당 업무는?",
            "지난 방문에서 누가 어떤 실험을 맡았어?",
        ):
            decision = classify_query(query)
            self.assertEqual(decision.mode, "rag")
            self.assertEqual(decision.reason, "internal_hint")

    def test_general_travel_recommendation_stays_general(self):
        self.assertEqual(
            classify_query("부산 여행지를 추천해줘").mode,
            "general",
        )

    def test_general_knowledge_uses_general(self):
        self.assertEqual(
            classify_query("파이썬 데코레이터를 설명해줘").mode, "general"
        )
        self.assertEqual(
            classify_query("이 문장을 영어로 번역해줘").mode, "general"
        )

    def test_greeting_and_weather_use_general(self):
        self.assertEqual(classify_query("안녕").mode, "general")
        self.assertEqual(classify_query("오늘 날씨 어때?").mode, "general")

    def test_probe_threshold_defaults_to_calibrated_value(self):
        """업무 질문 21개·잡담 24개로 실측해 고른 값. 근거는 config.py 주석 참고."""
        self.assertEqual(Settings().route_probe_threshold, 0.50)

    def test_location_wins_over_internal_hint(self):
        decision = classify_query("우리 연구실 토마토 과제 문서가 어느 폴더야?")
        self.assertEqual(decision.mode, "location")

    def test_openwebui_task_prompt_skips_retrieval(self):
        """Open WebUI 가 대화 턴마다 보내는 제목·태그 생성 요청.

        본문에 대화 전체가 들어 있어 "연구실"·"정책" 같은 단어가 섞이고, Open WebUI 는
        외부 API 로 보낼 때 metadata.task 표시를 떼므로(payload.pop) 우리 서버에는
        표시 없는 긴 글만 온다. 그래서 진짜 질문으로 오인돼 드라이브 검색까지 돌았다.
        """
        title_prompt = (
            "### Task:\n"
            "Generate a concise title summarizing the chat history.\n"
            "### Guidelines:\n"
            "- Keep it short: 2-4 words is best.\n"
            "### Output:\n"
            'JSON format: { "title": "your concise title here" }\n'
            "### Chat History:\n"
            "<chat_history>\n"
            "USER: 우리 연구실 토마토 과제와 농업용 로봇 정책 알려줘\n"
            "ASSISTANT: 연구실 내부 자료에 따르면 ...\n"
            "</chat_history>"
        )

        decision = classify_query(title_prompt)

        self.assertEqual(decision.mode, "general")
        self.assertEqual(decision.reason, "openwebui_task")

    def test_openwebui_task_detection_survives_case_and_missing_heading(self):
        """템플릿 머리말의 대소문자나 생략 때문에 내부 검색이 다시 켜지면 안 된다."""
        prompts = (
            "### TASK:\nGenerate a concise title summarizing the chat history.\n"
            "USER: 우리 연구실 토마토 과제",
            "Generate a concise title summarizing the chat history:\n"
            "우리 연구실 토마토 과제",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt.splitlines()[0]):
                decision = classify_query(prompt)
                self.assertEqual(
                    (decision.mode, decision.reason),
                    ("general", "openwebui_task"),
                )

    def test_web_named_as_the_source_beats_find_wording_location_hint(self):
        """출처를 웹이라고 대놓고 말했으면 '찾아 줘'를 위치 질문으로 읽지 않는다.

        예전에는 location 으로 갔다가 파일을 못 찾고 rag_web 으로 넘어와서, 사용자가
        웹이라고 말했는데도 위치 검색과 내부 검색을 둘 다 헛돌았다.
        """
        for query in (
            "웹에서 토마토 재배법 찾아줘",
            "인터넷에서 스마트팜 사례 찾아줘",
            "웹 검색으로 농기계 가격 찾아줘",
            "스마트팜 사례를 웹에서 함께 찾아줘",
            "스마트팜 사례를 웹을 함께 검색해서 찾아줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(classify_query(query).mode, "web")

    def test_web_as_a_topic_does_not_hijack_a_location_question(self):
        """'웹 크롤러 코드'처럼 웹이 주제일 뿐이면 위치 질문 그대로 둔다."""
        self.assertEqual(
            classify_query("웹 크롤러 코드 찾아줘").mode, "location"
        )

    def test_web_as_a_topic_does_not_turn_internal_content_into_hybrid_search(self):
        """'웹'이 자료의 주제일 뿐인데 Tavily까지 호출하는 회귀를 막는다."""
        for query in (
            "연구실 웹 크롤러 코드 설명해줘",
            "슬랙에 공유된 웹 세미나 내용 알려줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(classify_query(query).mode, "rag")

    def test_slack_and_web_request_beats_find_wording_location_hint(self):
        decision = classify_query("Slack과 인터넷에서 최근 부산 관련 내용을 찾아줘")
        self.assertEqual(
            (decision.mode, decision.reason),
            ("rag_web", "explicit_internal_web"),
        )

    def test_real_location_question_survives_internal_web_rule(self):
        for query in (
            "연구실 웹 크롤러 코드 어디 있어?",
            "슬랙에 공유된 웹 세미나 자료 경로 알려줘",
        ):
            self.assertEqual(classify_query(query).mode, "location")

    def test_explicit_drive_and_slack_request_uses_rag_over_find_wording(self):
        decision = classify_query(
            "2026년 06월 토마토 업무 기록을 찾아줘 구글드라이브와 슬랙 전부"
        )
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "explicit_multi_source")

    def test_external_current_question_uses_web(self):
        decision = classify_query("이번 주 스마트팜 최신 뉴스 알려줘")
        self.assertEqual(decision.mode, "web")
        self.assertEqual(decision.reason, "web_hint")

    def test_current_product_policy_and_product_recommendation_use_web(self):
        for query in (
            "최신 스마트팜 제품 추천해줘",
            "최신 농업 정책 알려줘",
            "농업용 로봇 정책이 어떻게 바뀌었어?",
            "스마트팜 제품 장단점은?",
        ):
            self.assertEqual(classify_query(query).mode, "web")

    def test_internal_product_question_stays_rag(self):
        self.assertEqual(
            classify_query("우리 연구실 제품 장단점 정리해줘").mode,
            "rag",
        )

    def test_explicit_internal_and_web_question_uses_both(self):
        decision = classify_query("연구실 토마토 과제와 웹 최신 동향을 함께 정리해줘")
        self.assertEqual(
            (decision.mode, decision.reason),
            ("rag_web", "explicit_internal_web"),
        )

    def test_web_object_particle_can_explicitly_request_both_sources(self):
        """사용자는 '웹 최신'뿐 아니라 '웹을 함께 검색'이라고도 말한다."""
        for query in (
            "연구실 토마토 자료와 웹을 함께 검색해줘",
            "슬랙 내용과 인터넷도 같이 찾아줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(classify_query(query).mode, "rag_web")

    def test_latest_without_web_request_stays_internal(self):
        decision = classify_query("연구실 토마토 과제 최신 내용 알려줘")
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "internal_hint")

    def test_explicit_drive_or_slack_stays_internal_despite_web_topics(self):
        for query in (
            "구글 드라이브에 있는 토마토 논문 내용 알려줘",
            "슬랙에서 공유한 토마토 논문 내용 알려줘",
        ):
            decision = classify_query(query)
            self.assertEqual(decision.mode, "rag")

    def test_internal_trend_question_without_explicit_web_stays_rag(self):
        decision = classify_query("연구실 토마토 과제 최신 동향을 알려줘")
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "internal_hint")

    def test_explicit_drive_slack_and_web_request_uses_both(self):
        decision = classify_query("구글 드라이브와 슬랙의 토마토 논문을 웹에서도 찾아줘")
        self.assertEqual(
            (decision.mode, decision.reason),
            ("rag_web", "explicit_internal_web"),
        )

    def test_ambiguous_query_requires_probe(self):
        decision = classify_query("지난번에 이야기한 결과가 어떻게 됐어?")
        self.assertTrue(decision.probe_required)
        self.assertEqual(decision.mode, "auto")

    def test_everyday_recommendations_are_answered_without_internal_search(self):
        """내부 말뭉치에 우연히 비슷한 문장이 있어도 생활 질문은 일반 답변이어야 한다."""
        for query in (
            "밥 뭐 먹을까?",
            "운동 루틴 추천해줘",
            "재미있는 영화 추천해줘",
            "친구 생일 선물 추천해줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(classify_query(query).mode, "general")

    def test_internal_context_still_beats_everyday_recommendation_wording(self):
        for query in (
            "슬랙에서 오늘 점심 추천 내용을 알려줘",
            "연구실 영화 추천 데이터셋을 설명해줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(classify_query(query).mode, "rag")

    def test_invalid_forced_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            classify_query("질문", "other")

    def test_structured_parser_failure_preserves_existing_routing(self):
        with patch(
            "labrag.router.parse_query_intent",
            side_effect=RuntimeError("parser unavailable"),
        ):
            decision = classify_query(
                "2026년 6월 토마토 업무를 구글 드라이브와 슬랙에서 찾아줘"
            )
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "explicit_multi_source")


class RouterProbeTests(unittest.TestCase):
    def test_low_internal_relevance_on_web_capable_intent_uses_web(self):
        decision = classify_query("최신 농업 정보")
        self.assertTrue(decision.probe_required)
        self.assertEqual(decision.reason, "ambiguous_web")
        resolved = resolve_probe(decision, 0.2, 0.65)
        self.assertEqual(resolved.mode, "web")
        self.assertEqual(resolved.reason, "probe_web")


class HybridQuerySplitTests(unittest.TestCase):
    def test_separates_internal_and_web_source_wording(self):
        internal, web = split_hybrid_queries(
            "연구실 토마토 과제와 웹 최신 동향을 함께 정리해줘"
        )

        self.assertEqual(internal, "토마토 과제 회의록 연구 내용")
        self.assertEqual(web, "토마토 웹 최신 동향을 함께 정리해줘")

    def test_preserves_shared_topic_when_sources_lead_the_question(self):
        internal, web = split_hybrid_queries(
            "Slack과 인터넷에서 최근 부산 내용을 찾아줘"
        )

        self.assertEqual(internal, "Slack 최근 부산 내용을 찾아줘")
        self.assertEqual(web, "인터넷에서 최근 부산 내용을 찾아줘")

    def test_handles_object_particle_in_explicit_web_request(self):
        internal, web = split_hybrid_queries(
            "연구실 토마토 자료와 웹을 함께 검색해줘"
        )

        self.assertEqual(internal, "연구실 토마토 자료")
        self.assertEqual(web, "토마토 자료와 웹을 함께 검색해줘")

    def test_separates_sources_when_web_clause_comes_first(self):
        internal, web = split_hybrid_queries(
            "웹 최신 토마토 동향과 연구실 토마토 과제를 함께 정리해줘"
        )

        self.assertEqual(internal, "연구실 토마토 과제를 함께 정리해줘")
        self.assertEqual(web, "웹 최신 토마토 동향")

    def test_drive_score_above_threshold_uses_rag(self):
        drive = FakeStore([0.8])
        score, error = probe_internal_relevance(
            "질문", FakeModels(), drive, threshold=0.45
        )
        decision = resolve_probe(classify_query("지난번 결과"), score, 0.45, error)
        self.assertEqual(decision.mode, "rag")

    def test_slack_score_is_considered(self):
        drive = FakeStore([0.1])
        slack = FakeStore([0.7])
        score, error = probe_internal_relevance(
            "질문", FakeModels(), drive, slack_store=slack, threshold=0.45
        )
        self.assertEqual(resolve_probe(
            classify_query("지난번 결과"), score, 0.45, error
        ).mode, "rag")

    def test_low_scores_use_general(self):
        score, error = probe_internal_relevance(
            "질문", FakeModels(), FakeStore([0.2]),
            slack_store=FakeStore([0.3]), threshold=0.45
        )
        self.assertEqual(resolve_probe(
            classify_query("지난번 결과"), score, 0.45, error
        ).mode, "general")

    def test_source_failures_are_isolated(self):
        score, error = probe_internal_relevance(
            "질문", FakeModels(), FakeStore(error=True),
            slack_store=FakeStore([0.7]), threshold=0.45
        )
        self.assertFalse(error)
        self.assertEqual(score, 0.7)

    def test_all_source_failures_fall_back_to_general(self):
        score, error = probe_internal_relevance(
            "질문", FakeModels(), FakeStore(error=True),
            slack_store=FakeStore(error=True), threshold=0.45
        )
        decision = resolve_probe(
            classify_query("지난번 결과"), score, 0.45, error
        )
        self.assertEqual(decision.mode, "general")
        self.assertEqual(decision.reason, "probe_error")

    def test_roots_only_apply_to_drive(self):
        drive = FakeStore([0.2])
        slack = FakeStore([0.3])
        probe_internal_relevance(
            "질문", FakeModels(), drive, slack_store=slack,
            roots=["ResearchWorkspace"], threshold=0.45,
        )
        self.assertEqual(drive.roots_seen, [["ResearchWorkspace"]])
        self.assertEqual(slack.roots_seen, [None])


if __name__ == "__main__":
    unittest.main()
