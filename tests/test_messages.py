from __future__ import annotations

import unittest

from labrag.rag import (
    Retrieved,
    build_general_messages,
    build_hybrid_messages,
    build_messages,
    build_web_messages,
    web_sources_block,
)
from labrag.store import Hit
from labrag.web_search import WebHit


def result_hit(citation, url, source=None):
    payload = {
        "text": f"{citation} 내용",
        "citation": citation,
        "file_url": url,
        "file_id": citation,
    }
    if source:
        payload["source"] = source
    return Hit(0.9, payload)


class MessageBuilderTests(unittest.TestCase):
    def setUp(self):
        self.history = [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요"},
            {"role": "user", "content": "파이썬 데코레이터를 설명해줘"},
        ]

    def test_general_messages_do_not_include_search_material(self):
        messages = build_general_messages(self.history)
        joined = "\n".join(message["content"] for message in messages)
        self.assertNotIn("[자료]", joined)
        self.assertNotIn("검색 결과", joined)

    def test_general_messages_keep_recent_conversation_and_question(self):
        messages = build_general_messages(self.history)
        self.assertEqual(messages[-1], self.history[-1])
        self.assertIn(self.history[-2], messages)
        self.assertEqual(messages[0]["role"], "system")

    def test_general_prompt_does_not_claim_internal_search(self):
        messages = build_general_messages(self.history)
        self.assertIn("내부 자료를 확인했다고 표현하지 마", messages[0]["content"])
        self.assertIn("요청한 언어", messages[0]["content"])

    def test_rag_messages_keep_material_block(self):
        retrieved = Retrieved([], [], "질문", 0)
        messages = build_messages(retrieved, self.history)
        self.assertIn("[자료]", messages[-1]["content"])
        self.assertIn("자료를 찾지 못했다고", messages[-1]["content"])

    def test_sources_block_groups_drive_and_slack_without_renumbering(self):
        drive = result_hit("회의록.docx", "https://drive.google.com/file/1")
        slack = result_hit(
            "Slack · #프로젝트운영 · thread 1785399154.691819",
            "https://slack.com/archives/C07RXJNFGAC/p1785399154691819",
            source="slack",
        )
        block = Retrieved([drive, slack], [0.9, 0.8], "질문", 2).sources_block()
        self.assertIn("**Google Drive 출처**", block)
        self.assertIn("**Slack 출처**", block)
        self.assertIn("1. [회의록.docx]", block)
        self.assertIn("2. [Slack · #프로젝트운영", block)
        self.assertLess(
            block.index("**Google Drive 출처**"),
            block.index("1. [회의록.docx]"),
        )
        self.assertLess(
            block.index("**Slack 출처**"),
            block.index("2. [Slack · #프로젝트운영"),
        )

    def test_mixed_sources_request_three_answer_sections(self):
        retrieved = Retrieved([
            result_hit("회의록.docx", "https://drive.google.com/file/1"),
            result_hit(
                "Slack · #프로젝트운영 · thread 1785399154.691819",
                "https://slack.com/archives/C07RXJNFGAC/p1785399154691819",
                source="slack",
            ),
        ], [0.9, 0.8], "질문", 2)
        prompt = build_messages(retrieved, self.history)[-1]["content"]
        self.assertIn("Google Drive에서 확인된 내용", prompt)
        self.assertIn("Slack에서 확인된 내용", prompt)
        self.assertIn("두 출처를 종합하면", prompt)

    def test_single_source_requests_only_its_answer_section(self):
        drive_only = Retrieved([
            result_hit("회의록.docx", "https://drive.google.com/file/1")
        ], [0.9], "질문", 1)
        drive_prompt = build_messages(drive_only, self.history)[-1]["content"]
        self.assertIn("Google Drive에서 확인된 내용", drive_prompt)
        self.assertNotIn("Slack에서 확인된 내용", drive_prompt)
        self.assertNotIn("두 출처를 종합하면", drive_prompt)

        slack_only = Retrieved([
            result_hit(
                "Slack · #프로젝트운영 · thread 1785399154.691819",
                "https://slack.com/archives/C07RXJNFGAC/p1785399154691819",
                source="slack",
            )
        ], [0.9], "질문", 1)
        slack_prompt = build_messages(slack_only, self.history)[-1]["content"]
        self.assertIn("Slack에서 확인된 내용", slack_prompt)
        self.assertNotIn("Google Drive에서 확인된 내용", slack_prompt)
        self.assertNotIn("두 출처를 종합하면", slack_prompt)

    def test_structured_intent_is_added_to_rag_prompt(self):
        history = [{
            "role": "user",
            "content": (
                "2026년 6월 홍길동 님의 토마토 업무를 Google Drive와 "
                "Slack #crop-imaging에서 담당자별 표로 정리해줘"
            ),
        }]
        retrieved = Retrieved([
            result_hit("회의록.docx", "https://drive.google.com/file/1"),
            result_hit(
                "Slack · #crop-imaging · thread 1",
                "https://slack.com/archives/C1/p1",
                source="slack",
            ),
        ], [0.9, 0.8], history[0]["content"], 2)
        prompt = build_messages(retrieved, history)[-1]["content"]
        self.assertIn("[질문 해석]", prompt)
        self.assertIn("기간: 2026-06-01 이상, 2026-07-01 미만", prompt)
        self.assertIn("출처: Google Drive, Slack", prompt)
        self.assertIn("담당자: 홍길동", prompt)
        self.assertIn("채널: #crop-imaging", prompt)
        self.assertIn("Markdown 표", prompt)

    def test_web_messages_give_the_url_and_keep_the_informal_source_warning(self):
        """등급 라벨은 없애되, 블로그를 사실로 단정하지 말라는 지시는 남긴다.

        등급을 도메인으로 찍던 방식은 실측 30건에서 정부·연구기관·학술지를
        '경험/비공식'으로 강등시켰다. 라벨 대신 URL 을 주고 모델이 판단하게 한다.
        """
        hits = [WebHit(
            title="스마트팜 사용 후기",
            url="https://example.blog/smartfarm",
            content="농가에서 직접 사용한 경험",
            score=0.72,
        )]

        messages = build_web_messages(hits, self.history)
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("웹 검색 자료", joined)
        self.assertIn("스마트팜 사용 후기", joined)
        self.assertIn("https://example.blog/smartfarm", joined)
        self.assertNotIn("등급", joined)
        self.assertIn("사실로 단정하지 마", joined)

    def test_hybrid_messages_separate_internal_web_and_synthesis_sections(self):
        retrieved = Retrieved([
            result_hit("토마토 과제 회의록", "https://drive.google.com/file/1")
        ], [0.91], "질문", 3)
        web_hits = [WebHit(
            title="최신 토마토 재배 동향",
            url="https://rda.go.kr/strawberry",
            content="최신 기술 요약",
            score=0.88,
        )]

        prompt = build_hybrid_messages(
            retrieved, web_hits, self.history
        )[-1]["content"]

        self.assertIn("[연구실 내부 자료]", prompt)
        self.assertIn("[웹 검색 자료]", prompt)
        self.assertIn("### 연구실 내부 자료", prompt)
        self.assertIn("### 웹에서 확인한 내용", prompt)
        self.assertIn("### 종합", prompt)

    def test_web_source_link_survives_parentheses_in_url(self):
        """위키백과처럼 괄호가 든 URL 은 마크다운 링크를 조기 종료시킨다."""
        block = web_sources_block([WebHit(
            title="로봇",
            url="https://ko.wikipedia.org/wiki/로봇_(기계)",
            content="설명",
            score=0.9,
        )])

        self.assertIn(
            "[로봇](https://ko.wikipedia.org/wiki/로봇_%28기계%29)", block
        )

    def test_web_source_link_escapes_brackets_in_title(self):
        """제목의 대괄호가 링크 텍스트를 끊지 못하게 한다."""
        block = web_sources_block([WebHit(
            title="[공식] 발표]자료",
            url="https://rda.go.kr/a",
            content="설명",
            score=0.9,
        )])

        self.assertIn(
            r"[\[공식\] 발표\]자료](https://rda.go.kr/a)", block
        )

    def test_web_sources_block_lists_plain_markdown_links(self):
        block = web_sources_block([WebHit(
            title="스마트팜 사용 후기",
            url="https://example.blog/smartfarm",
            content="후기",
            score=0.72,
        )])

        self.assertIn("**웹 출처**", block)
        self.assertIn(
            "1. [스마트팜 사용 후기](https://example.blog/smartfarm)", block
        )


if __name__ == "__main__":
    unittest.main()
