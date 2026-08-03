from __future__ import annotations

import asyncio
import unittest
import logging
import threading
from unittest.mock import AsyncMock, patch

import httpx
from pydantic import ValidationError

from labrag import server
from labrag.server import (
    ChatRequest,
    MODEL_ID,
    _after_location_attempt,
    _decide_route,
    _location_fallback,
    _log_route_event,
    _requires_store,
    search_only,
)
from labrag.router import RouteDecision
from labrag.rag import Retrieved
from labrag.store import Hit
from labrag.config import Settings
from labrag.web_search import (
    WEB_CONTENT_PREVIEW_CHARS,
    WebHit,
    reset_web_search_metrics,
    search_web,
)


class FakeModels:
    def embed_one(self, query):
        return [0.1]


class FakeStore:
    def __init__(self, score=0.0):
        self.score = score

    def search(self, vector, limit, roots=None):
        if not self.score:
            return []
        return [type("Hit", (), {"score": self.score})()]

    def exists(self):
        return True


class RecoverableStore:
    def __init__(self, collection, *, exists=True, count=1):
        self.collection = collection
        self._exists = exists
        self._count = count

    def exists(self):
        return self._exists

    def count(self):
        return self._count


class FakeRetrieved:
    n_candidates = 1
    hits = []
    scores = []


def drive_retrieved():
    return Retrieved([
        Hit(0.9, {
            "text": "토마토 과제 내부 진행 내용",
            "citation": "토마토 과제 회의록.docx",
            "file_url": "https://drive.google.com/file/1",
            "file_id": "drive-1",
        })
    ], [0.91], "질문", 4)


def informal_web_hit():
    return WebHit(
        title="스마트팜 사용 후기",
        url="https://example.blog/smartfarm",
        content="농가 사용 경험",
        score=0.72,
    )


class FakeGenerationResponse:
    status_code = 200
    text = ""

    def __init__(self, answer="생성된 답변"):
        self.answer = answer

    def json(self):
        return {
            "choices": [{
                "message": {"content": self.answer},
                "finish_reason": "stop",
            }],
            "usage": {},
        }


class ServerRoutingTests(unittest.TestCase):
    def request(self, **kwargs):
        return ChatRequest(
            messages=[{"role": "user", "content": "질문"}], **kwargs
        )

    def test_mode_defaults_to_auto(self):
        self.assertEqual(self.request().mode, "auto")

    def test_public_model_id_is_labrag(self):
        self.assertEqual(MODEL_ID, "LabRAG")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValidationError):
            self.request(mode="other")

    def test_general_mode_does_not_require_store(self):
        self.assertFalse(_requires_store("general"))
        self.assertFalse(_requires_store("web"))
        self.assertTrue(_requires_store("rag"))
        self.assertTrue(_requires_store("rag_web"))
        self.assertTrue(_requires_store("location"))

    def test_forced_web_modes_are_accepted(self):
        self.assertEqual(self.request(mode="web").mode, "web")
        self.assertEqual(self.request(mode="rag_web").mode, "rag_web")

    def test_general_rule_skips_probe(self):
        decision = _decide_route(
            "파이썬 데코레이터를 설명해줘", "auto",
            FakeModels(), FakeStore(0.9), None, None,
        )
        self.assertEqual(decision.mode, "general")
        self.assertIsNone(decision.probe_score)

    def test_ambiguous_query_uses_probe(self):
        decision = _decide_route(
            "지난번 결과가 어땠어?", "auto",
            FakeModels(), FakeStore(0.8), None, None,
        )
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "probe_relevant")

    def test_forced_rag_skips_location_rule(self):
        decision = _decide_route(
            "Conference.pdf 어디 있어?", "rag",
            FakeModels(), FakeStore(), None, None,
        )
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "forced")

    def test_auto_location_without_result_falls_through_to_rag(self):
        decision = _after_location_attempt(
            RouteDecision("location", "location_hint"), "auto", None,
            "그 파일 어디 있어?",
        )
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "location_no_result")

    def test_auto_location_without_result_falls_through_to_rag_web_with_web_hint(self):
        decision = _after_location_attempt(
            RouteDecision("location", "location_hint"), "auto", None,
            "Slack과 인터넷에서 최근 부산 관련 내용을 찾아줘",
        )
        self.assertEqual(decision.mode, "rag_web")
        self.assertEqual(decision.reason, "location_no_result_web")

    def test_web_as_a_topic_does_not_add_web_search_on_location_fallback(self):
        """'웹 크롤러'의 웹은 주제일 뿐, 웹에서 찾아달라는 뜻이 아니다."""
        decision = _after_location_attempt(
            RouteDecision("location", "location_hint"), "auto", None,
            "웹 크롤러 코드 찾아줘",
        )
        self.assertEqual(decision.mode, "rag")
        self.assertEqual(decision.reason, "location_no_result")

    def test_forced_location_without_result_stays_location(self):
        original = RouteDecision("location", "forced")
        self.assertEqual(
            _after_location_attempt(original, "location", None, "질문"), original
        )

    def test_missing_live_drive_is_reported_as_fallback(self):
        fallback, warning = _location_fallback(
            True, None, None, object()
        )
        self.assertTrue(fallback)
        self.assertIn("로컬 색인", warning)

    def test_route_log_contains_aggregates_but_not_query(self):
        decision = RouteDecision("rag", "internal_hint", probe_score=0.7)
        with self.assertLogs("uvicorn.error", level="INFO") as captured:
            _log_route_event(
                "req-1", decision, fallback=False,
                n_candidates=10, n_sources=3, elapsed_ms=12,
            )
        log = captured.output[0]
        self.assertIn('"mode": "rag"', log)
        self.assertIn('"n_sources": 3', log)
        self.assertNotIn("토마토 질문 원문", log)

    def test_route_logger_emits_info_in_production(self):
        self.assertEqual(server.route_logger.name, "uvicorn.error")
        self.assertLessEqual(
            server.route_logger.getEffectiveLevel(), logging.INFO
        )

    def test_followup_routing_query_inherits_recent_user_topic(self):
        history = [
            {"role": "user", "content": "최근에 부산 갔다 온 사람이 있을까?"},
            {"role": "assistant", "content": "untrusted assistant answer"},
            {"role": "user", "content": "당시 참여자들의 담당 업무는?"},
        ]
        self.assertEqual(
            server._routing_query(history),
            "최근에 부산 갔다 온 사람이 있을까?\n당시 참여자들의 담당 업무는?",
        )

    def test_new_topic_does_not_append_previous_user_question(self):
        history = [
            {"role": "user", "content": "부산 출장 참여자는?"},
            {"role": "user", "content": "파이썬 데코레이터를 설명해줘"},
        ]
        self.assertEqual(
            server._routing_query(history),
            "파이썬 데코레이터를 설명해줘",
        )

    def test_recovers_missing_slack_stores_when_qdrant_becomes_ready(self):
        old_state = dict(server._state)
        server._state.clear()
        server._state.update({
            "slack_store": None,
            "slack_parent_store": None,
        })
        try:
            with patch.object(
                server,
                "Store",
                side_effect=lambda collection: RecoverableStore(collection),
            ):
                recovered = server._ensure_slack_stores()
            self.assertTrue(recovered)
            self.assertEqual(
                server._state["slack_store"].collection,
                server.SLACK_COLLECTION,
            )
            self.assertEqual(
                server._state["slack_parent_store"].collection,
                server.SLACK_PARENT_COLLECTION,
            )
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_keeps_existing_slack_store_without_recreating_it(self):
        old_state = dict(server._state)
        current = RecoverableStore(server.SLACK_COLLECTION)
        parent = RecoverableStore(server.SLACK_PARENT_COLLECTION)
        server._state.clear()
        server._state.update({
            "slack_store": current,
            "slack_parent_store": parent,
        })
        try:
            with patch.object(server, "Store") as store_type:
                recovered = server._ensure_slack_stores()
            self.assertTrue(recovered)
            self.assertIs(server._state["slack_store"], current)
            self.assertIs(server._state["slack_parent_store"], parent)
            store_type.assert_not_called()
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_parent_failure_does_not_disable_main_slack_store(self):
        old_state = dict(server._state)
        server._state.clear()
        server._state.update({
            "slack_store": None,
            "slack_parent_store": None,
        })

        def build_store(collection):
            if collection == server.SLACK_PARENT_COLLECTION:
                raise ConnectionError("parent unavailable")
            return RecoverableStore(collection)

        try:
            with patch.object(server, "Store", side_effect=build_store):
                recovered = server._ensure_slack_stores()
            self.assertTrue(recovered)
            self.assertIsNotNone(server._state["slack_store"])
            self.assertIsNone(server._state["slack_parent_store"])
        finally:
            server._state.clear()
            server._state.update(old_state)


class SearchRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_general_search_returns_classification_without_retrieval(self):
        old_state = dict(server._state)
        server._state.clear()
        server._state.update({
            "models": FakeModels(),
            "store": FakeStore(0.9),
            "slack_store": None,
            "catalog": None,
            "docsyn_store": None,
            "drive_live": None,
            "ancestry": None,
            "canonical_config": None,
            "slack_parent_store": None,
        })
        req = ChatRequest(
            mode="general",
            messages=[{"role": "user", "content": "일반 질문"}],
        )
        try:
            with patch.object(
                server, "maybe_locate_hybrid", new=AsyncMock(return_value=None)
            ) as locate, patch.object(server, "retrieve") as retrieve:
                result = await search_only(req)
            self.assertEqual(result["mode"], "general")
            self.assertEqual(result["results"], [])
            locate.assert_not_awaited()
            retrieve.assert_not_called()
        finally:
            server._state.clear()
            server._state.update(old_state)


class WebChatRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_state = dict(server._state)
        self.http = AsyncMock()
        self.http.post.return_value = FakeGenerationResponse()
        server._state.clear()
        server._state.update({
            "models": FakeModels(),
            "store": FakeStore(0.9),
            "http": self.http,
            "slack_store": RecoverableStore(server.SLACK_COLLECTION),
            "slack_parent_store": None,
            "catalog": None,
            "canonical_config": None,
            "docsyn_store": None,
            "drive_live": None,
            "ancestry": None,
        })

    async def asyncTearDown(self):
        server._state.clear()
        server._state.update(self.old_state)

    async def test_web_chat_returns_tiered_sources_without_internal_retrieval(self):
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "이번 주 스마트팜 최신 뉴스 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ), patch.object(
            server, "retrieve", side_effect=AssertionError("웹 전용은 내부 검색 금지")
        ):
            response = await server.chat(req)

        answer = response["choices"][0]["message"]["content"]
        self.assertEqual(response["lab_rag"]["mode"], "web")
        self.assertIn("**웹 출처**", answer)
        self.assertIn("https://example.blog/smartfarm", answer)
        self.assertEqual(
            response["lab_rag"]["web_sources"][0]["url"],
            "https://example.blog/smartfarm",
        )

    async def test_forced_web_mode_does_not_short_circuit_slack_only_query(self):
        server._state["slack_store"] = None
        req = ChatRequest(
            mode="web",
            messages=[{"role": "user", "content": "Slack 최신 내용 가져와"}],
        )
        with patch.object(
            server, "_ensure_slack_stores", return_value=False
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ), patch.object(
            server, "retrieve", side_effect=AssertionError("강제 웹은 RAG 금지")
        ):
            response = await server.chat(req)

        self.assertEqual(response["lab_rag"]["mode"], "web")
        self.assertIn("**웹 출처**", response["choices"][0]["message"]["content"])

    async def test_forced_general_mode_does_not_short_circuit_slack_only_query(self):
        server._state["slack_store"] = None
        self.http.post.return_value = FakeGenerationResponse("일반 답변")
        req = ChatRequest(
            mode="general",
            messages=[{"role": "user", "content": "Slack 최신 내용 가져와"}],
        )
        with patch.object(
            server, "_ensure_slack_stores", return_value=False
        ), patch.object(
            server, "retrieve", side_effect=AssertionError("강제 일반은 RAG 금지")
        ):
            response = await server.chat(req)

        self.assertEqual(response["lab_rag"]["mode"], "general")
        self.assertEqual(
            response["choices"][0]["message"]["content"], "일반 답변"
        )

    async def test_rag_web_starts_internal_and_web_searches_concurrently(self):
        web_started = threading.Event()
        concurrency_observed = []

        def retrieve_after_web_starts(*args, **kwargs):
            concurrency_observed.append(web_started.wait(0.5))
            return drive_retrieved()

        async def start_web(*args, **kwargs):
            web_started.set()
            await asyncio.sleep(0)
            return [informal_web_hit()]

        req = ChatRequest(messages=[{
            "role": "user",
            "content": "연구실 토마토 과제와 웹 최신 동향을 함께 정리해줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "retrieve", side_effect=retrieve_after_web_starts
        ), patch.object(server, "search_web", side_effect=start_web):
            response = await server.chat(req)

        self.assertEqual(concurrency_observed, [True])
        self.assertEqual(response["lab_rag"]["mode"], "rag_web")
        self.assertEqual(len(response["lab_rag"]["sources"]), 1)
        self.assertEqual(len(response["lab_rag"]["web_sources"]), 1)
        answer = response["choices"][0]["message"]["content"]
        self.assertIn("**Google Drive 출처**", answer)
        self.assertIn("**웹 출처**", answer)

    async def test_rag_web_uses_source_specific_queries_in_chat(self):
        """한 문장을 그대로 양쪽에 보내 내부·웹 결과가 모두 오염되는 회귀를 막는다."""
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "연구실 토마토 과제와 웹 최신 동향을 함께 정리해줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "retrieve", return_value=drive_retrieved()
        ) as retrieve, patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ) as web:
            await server.chat(req)

        self.assertEqual(
            retrieve.call_args.args[0],
            "토마토 과제 회의록 연구 내용",
        )
        web.assert_awaited_once_with(
            "토마토 웹 최신 동향을 함께 정리해줘",
            server._state["http"],
            server.settings,
        )

    async def test_rag_web_uses_source_specific_queries_in_search_endpoint(self):
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "Slack과 인터넷에서 최근 부산 내용을 찾아줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "retrieve", return_value=drive_retrieved()
        ) as retrieve, patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ) as web:
            await server.search_only(req)

        self.assertEqual(
            retrieve.call_args.args[0], "Slack 최근 부산 내용을 찾아줘"
        )
        web.assert_awaited_once_with(
            "인터넷에서 최근 부산 내용을 찾아줘",
            server._state["http"],
            server.settings,
        )

    async def test_rag_web_keeps_hybrid_prompt_when_internal_side_is_empty(self):
        empty = Retrieved([], [], "질문", 0)
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "연구실 토마토 과제와 웹 최신 동향을 함께 정리해줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "retrieve", return_value=empty
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ):
            await server.chat(req)

        messages = self.http.post.call_args.kwargs["json"]["messages"]
        prompt = messages[-1]["content"]
        self.assertIn("### 연구실 내부 자료", prompt)
        self.assertIn("### 웹에서 확인한 내용", prompt)
        self.assertIn("### 종합", prompt)
        self.assertIn("내부 검색 결과가 없음", prompt)

    async def test_rag_web_without_any_evidence_reports_both_sources_empty(self):
        self.http.post.return_value = FakeGenerationResponse("일반 답변")
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "연구실 토마토 과제와 웹 최신 동향을 함께 정리해줘",
        }])
        empty = Retrieved([], [], "질문", 0)
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "retrieve", return_value=empty
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[])
        ):
            response = await server.chat(req)

        answer = response["choices"][0]["message"]["content"]
        self.assertIn("연구실 내부 자료와 웹 검색 모두", answer)
        self.assertIn("근거를 찾지 못해", answer)
        self.assertNotIn("연구실 내부 자료를 우선", answer)
        self.assertTrue(response["lab_rag"]["fallback"])

    async def test_web_failure_falls_back_to_general_generation(self):
        self.http.post.return_value = FakeGenerationResponse("일반 지식 답변")
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "이번 주 스마트팜 최신 뉴스 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[])
        ), patch.object(
            server, "retrieve", side_effect=AssertionError("웹 폴백은 내부 검색 금지")
        ):
            response = await server.chat(req)

        answer = response["choices"][0]["message"]["content"]
        self.assertIn("일반 지식 답변", answer)
        self.assertIn("웹 검색 결과를 가져오지 못해", answer)
        self.assertIn("일반 지식으로 답변", answer)
        self.assertEqual(response["lab_rag"]["mode"], "web")
        self.assertTrue(response["lab_rag"]["fallback"])
        self.assertEqual(response["lab_rag"]["web_sources"], [])

    async def test_search_endpoint_returns_web_results_for_web_route(self):
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "이번 주 스마트팜 최신 뉴스 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ), patch.object(
            server, "retrieve", side_effect=AssertionError("웹 전용은 내부 검색 금지")
        ):
            response = await server.search_only(req)

        self.assertEqual(response["mode"], "web")
        self.assertEqual(response["results"], [])
        self.assertEqual(
            response["web_results"][0]["url"], "https://example.blog/smartfarm"
        )
        self.assertEqual(
            response["web_results"][0]["content"], "농가 사용 경험"
        )

    async def test_search_endpoint_caps_web_content_length(self):
        """/search 는 프롬프트와 같은 상한까지만 웹 본문을 돌려준다."""
        long_hit = WebHit(
            title="긴 본문",
            url="https://example.blog/long",
            content="가" * (WEB_CONTENT_PREVIEW_CHARS + 500),
            score=0.7,
        )
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "이번 주 스마트팜 최신 뉴스 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[long_hit])
        ):
            response = await server.search_only(req)

        self.assertEqual(
            len(response["web_results"][0]["content"]),
            WEB_CONTENT_PREVIEW_CHARS,
        )

    async def test_chat_says_when_internal_search_was_skipped(self):
        """내부 근거를 못 찾아 일반 지식으로 답했으면 밝혀야 한다.

        말없이 일반 답변을 하면 사용자는 연구실에 자료가 없다고 오해한다 — 실측에서
        "오로라 배치체계"가 인덱스의 정의 대신 틀린 일반 지식으로 답해졌다.
        """
        req = ChatRequest(messages=[{
            "role": "user", "content": "오로라 배치체계가 뭐야?",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "_decide_route",
            return_value=RouteDecision("general", "probe_low", probe_score=0.42),
        ), patch.object(
            server, "retrieve", side_effect=AssertionError("일반 답변은 검색 금지"),
        ):
            response = await server.chat(req)

        answer = response["choices"][0]["message"]["content"]
        self.assertIn("일반 지식", answer)

    async def test_chat_tells_the_user_which_channel_was_assumed(self):
        """이름 일부만 듣고 채널을 좁혔으면 밝혀야 사용자가 틀린 걸 알아챈다."""
        retrieved = drive_retrieved()
        retrieved.notes = ["#프로젝트운영 채널로 이해했어."]
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "슬랙에서 운영 내용 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(
            server, "retrieve", return_value=retrieved
        ):
            response = await server.chat(req)

        answer = response["choices"][0]["message"]["content"]
        self.assertIn("#프로젝트운영 채널로 이해했어.", answer)

    async def test_search_endpoint_exposes_retrieval_notes(self):
        """평가 도구와 UI도 채널 추정 경고를 숨기지 않아야 한다."""
        retrieved = drive_retrieved()
        retrieved.notes = ["#연구과제 채널로 이해했어."]
        req = ChatRequest(messages=[{
            "role": "user", "content": "슬랙에서 연구과제 진행 상황 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=True
        ), patch.object(server, "retrieve", return_value=retrieved):
            response = await server.search_only(req)

        self.assertEqual(response["notes"], ["#연구과제 채널로 이해했어."])

    async def test_slack_and_web_chat_keeps_web_result_when_slack_is_unavailable(self):
        server._state["slack_store"] = None
        empty = Retrieved([], [], "질문", 0)
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "Slack 최신 내용과 인터넷 검색 결과를 함께 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=False
        ), patch.object(
            server, "retrieve", return_value=empty
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ):
            response = await server.chat(req)

        answer = response["choices"][0]["message"]["content"]
        self.assertEqual(response["lab_rag"]["mode"], "rag_web")
        self.assertEqual(response["lab_rag"]["slack_status"], "unavailable")
        self.assertTrue(response["lab_rag"]["fallback"])
        self.assertIn("Slack 검색 저장소", answer)
        self.assertIn("https://example.blog/smartfarm", answer)
        self.assertEqual(len(response["lab_rag"]["web_sources"]), 1)

    async def test_slack_and_web_search_returns_web_result_with_warning(self):
        server._state["slack_store"] = None
        empty = Retrieved([], [], "질문", 0)
        req = ChatRequest(messages=[{
            "role": "user",
            "content": "Slack 최신 내용과 인터넷 검색 결과를 함께 알려줘",
        }])
        with patch.object(
            server, "_ensure_slack_stores", return_value=False
        ), patch.object(
            server, "retrieve", return_value=empty
        ), patch.object(
            server, "search_web", new=AsyncMock(return_value=[informal_web_hit()])
        ):
            response = await server.search_only(req)

        self.assertEqual(response["mode"], "rag_web")
        self.assertEqual(response["slack_status"], "unavailable")
        self.assertTrue(response["fallback"])
        self.assertIn("Slack 검색 저장소", response["warnings"][0])
        self.assertEqual(
            response["web_results"][0]["content"], "농가 사용 경험"
        )

    async def test_search_reports_unavailable_explicit_slack_without_retrieval(self):
        old_state = dict(server._state)
        server._state.clear()
        server._state.update({
            "models": FakeModels(),
            "store": FakeStore(0.9),
            "slack_store": None,
            "slack_parent_store": None,
            "catalog": None,
            "canonical_config": None,
            "docsyn_store": None,
        })
        req = ChatRequest(
            messages=[{
                "role": "user",
                "content": "Slack에서 최신 내용 가져와",
            }],
        )
        try:
            with patch.object(
                server, "_ensure_slack_stores", return_value=False
            ), patch.object(
                server,
                "retrieve",
                side_effect=AssertionError("Slack 장애 시 검색하면 안 됨"),
            ) as retrieve:
                result = await search_only(req)
            self.assertEqual(result["status"], "unavailable")
            self.assertIn("Slack 검색 저장소", result["error"])
            self.assertEqual(result["results"], [])
            retrieve.assert_not_called()
        finally:
            server._state.clear()
            server._state.update(old_state)

    async def test_chat_reports_unavailable_explicit_slack_without_generation(self):
        old_state = dict(server._state)
        http = AsyncMock()
        server._state.clear()
        server._state.update({
            "models": FakeModels(),
            "store": FakeStore(0.9),
            "http": http,
            "slack_store": None,
            "slack_parent_store": None,
            "catalog": None,
            "canonical_config": None,
            "docsyn_store": None,
        })
        req = ChatRequest(
            messages=[{
                "role": "user",
                "content": "Slack에서 최신 내용 가져와",
            }],
            stream=False,
        )
        try:
            with patch.object(
                server, "_ensure_slack_stores", return_value=False
            ), patch.object(
                server,
                "retrieve",
                side_effect=AssertionError("Slack 장애 시 검색하면 안 됨"),
            ) as retrieve:
                result = await server.chat(req)
            answer = result["choices"][0]["message"]["content"]
            self.assertIn("Slack 검색 저장소", answer)
            self.assertEqual(result["lab_rag"]["status"], "unavailable")
            retrieve.assert_not_called()
            http.post.assert_not_awaited()
        finally:
            server._state.clear()
            server._state.update(old_state)

    async def test_search_uses_followup_context_for_routing_and_retrieval(self):
        old_state = dict(server._state)
        server._state.clear()
        server._state.update({
            "models": FakeModels(),
            "store": FakeStore(0.9),
            "slack_store": RecoverableStore(server.SLACK_COLLECTION),
            "slack_parent_store": None,
            "catalog": None,
            "canonical_config": None,
            "docsyn_store": None,
        })
        req = ChatRequest(messages=[
            {"role": "user", "content": "최근에 부산 갔다 온 사람이 있을까?"},
            {"role": "user", "content": "당시 참여자들의 담당 업무는?"},
        ])
        try:
            with patch.object(
                server, "_ensure_slack_stores", return_value=True
            ), patch.object(
                server, "retrieve", return_value=FakeRetrieved()
            ) as retrieve:
                result = await search_only(req)
            self.assertEqual(
                retrieve.call_args.args[0],
                "최근에 부산 갔다 온 사람이 있을까?\n당시 참여자들의 담당 업무는?",
            )
            self.assertEqual(result["query"], "당시 참여자들의 담당 업무는?")
        finally:
            server._state.clear()
            server._state.update(old_state)


class HealthWebMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_health_exposes_tavily_call_counts(self):
        """쿼터 사용량을 로그 grep 없이 확인할 수 있어야 한다."""
        reset_web_search_metrics()
        self.addCleanup(reset_web_search_metrics)

        async def rate_limited(request):
            return httpx.Response(429, json={"detail": "rate limit"})

        quota_client = httpx.AsyncClient(
            transport=httpx.MockTransport(rate_limited)
        )
        self.addAsyncCleanup(quota_client.aclose)
        await search_web(
            "스마트팜", quota_client, Settings(tavily_api_key="test-key")
        )

        old_state = dict(server._state)
        server._state.update({
            "models": type("M", (), {"health": lambda self: {}})(),
            "store": type("S", (), {"stats": lambda self: {}})(),
            "http": quota_client,
        })
        try:
            result = await server.health()
        finally:
            server._state.clear()
            server._state.update(old_state)

        self.assertEqual(
            result["web_search"],
            {"calls": 1, "ok": 0, "empty": 0, "errors": 1,
             "skipped_too_long": 0},
        )


if __name__ == "__main__":
    unittest.main()
