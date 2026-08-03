from __future__ import annotations

import unittest
from unittest.mock import patch

from labrag.slack_channels import resolve_channels


# 실제 연구실 Slack 채널 목록에서 가져온 표본. 일상어와 겹치는 이름(논의)과
# 구분자가 제각각인 이름(work-log, project-development)을 일부러 포함한다.
KNOWN = (
    "프로젝트운영",
    "lab-website",
    "work-log",
    "project-development",
    "joint-project",
    "연구과제",
    "논의",
    "seedling-ai",
    "agriculture-ai-contest",
    "김연구-학위논문내용공유-thesis-etc",
)


class ChannelResolutionTests(unittest.TestCase):
    def test_spacing_difference_still_finds_the_channel(self):
        """실제 채널은 '프로젝트운영'인데 사람은 '#프로젝트 운영'이라고 쓴다.

        이 한 칸 때문에 Slack 결과가 전부 걸러져 0건이 나왔다.
        """
        match = resolve_channels(
            "슬랙의 #프로젝트 운영 채널에서 최근 한 내용좀 알려줘", KNOWN
        )

        self.assertEqual(match.names, ("프로젝트운영",))
        self.assertEqual(match.confidence, "confirmed")


    def test_partial_name_resolves_when_only_one_channel_matches(self):
        """사람은 이름 전체를 말하지 않는다 — '운영'만으로 프로젝트운영을 찾아야 한다."""
        match = resolve_channels("슬랙에서 운영 내용 알려줘", KNOWN)

        self.assertEqual(match.names, ("프로젝트운영",))
        self.assertEqual(match.confidence, "inferred")

    def test_partial_name_matching_several_channels_is_not_guessed(self):
        """'참독'은 두 채널에 걸린다 — 하나로 찍으면 틀린 쪽을 고를 수 있다."""
        match = resolve_channels("슬랙 참독 채널 내용 알려줘", KNOWN)

        self.assertEqual(match, resolve_channels("", KNOWN))


    def test_channel_names_are_ignored_without_a_slack_cue(self):
        """'연구과제'·'논의'는 채널 이름이자 일상어다.

        슬랙을 가리키는 말이 없으면 채널 지정으로 읽지 않는다 — 그러지 않으면
        평범한 드라이브 질문이 엉뚱한 채널로 좁혀진다.
        """
        for query in (
            "우리 연구과제 진행상황 알려줘",
            "지난주 논의된 내용 정리해줘",
            "Conference 논문 내용을 요약해줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(resolve_channels(query, KNOWN).names, ())

    def test_slack_cue_enables_resolution_for_the_same_wording(self):
        """같은 표현이라도 슬랙을 가리키면 채널로 읽는다."""
        self.assertEqual(
            resolve_channels("슬랙 연구과제 채널 내용 알려줘", KNOWN).names,
            ("연구과제",),
        )

    def test_ordinary_inflected_word_is_not_confirmed_as_a_channel(self):
        """'논의된'은 #논의 채널을 지목한 말이 아니다."""
        self.assertEqual(
            resolve_channels("슬랙에서 최근 논의된 내용 알려줘", KNOWN).names,
            (),
        )

    def test_generic_work_words_do_not_infer_a_channel(self):
        """'토마토 관련 업무'의 관련을 #project-development 별칭으로 오인하면 안 된다."""
        query = (
            "2026년 06월 토마토 관련 업무 기록을 "
            "구글 드라이브와 슬랙에서 전부 정리해줘"
        )

        self.assertEqual(resolve_channels(query, KNOWN).names, ())

    def test_bare_channel_name_with_slack_cue_is_only_an_inference(self):
        """# 또는 '채널' 표기 없이 이름이 겹치면 사용자에게 추정임을 알려야 한다."""
        match = resolve_channels("슬랙에서 연구과제 진행 상황 알려줘", KNOWN)

        self.assertEqual(match.names, ("연구과제",))
        self.assertEqual(match.confidence, "inferred")


class FakeQdrant:
    def __init__(self, channel_names):
        self._names = list(channel_names)
        self.scroll_calls = 0

    def scroll(self, **kwargs):
        self.scroll_calls += 1
        points = [
            type("P", (), {"payload": {"channel_name": name}})()
            for name in self._names
        ]
        return points, None


class ChannelNameCacheTests(unittest.TestCase):
    def store(self, names):
        from labrag.store import Store
        store = Store.__new__(Store)
        store.collection = "lab_slack"
        store.client = FakeQdrant(names)
        return store

    def test_lists_distinct_channel_names(self):
        store = self.store(["프로젝트운영", "프로젝트운영", "논의"])

        self.assertEqual(store.channel_names(), ("논의", "프로젝트운영"))

    def test_reuses_the_cached_list_instead_of_scanning_every_query(self):
        """질문마다 전체 스캔하면 검색 지연이 그만큼 늘어난다."""
        store = self.store(["프로젝트운영"])

        store.channel_names()
        store.channel_names()

        self.assertEqual(store.client.scroll_calls, 1)

    def test_refreshes_cached_channels_after_ttl(self):
        """주간 인덱싱 뒤 서비스를 재시작하지 않아도 새 채널이 보여야 한다."""
        store = self.store(["프로젝트운영"])
        with patch("labrag.store.time.monotonic", side_effect=[10.0, 20.0, 400.1]):
            self.assertEqual(store.channel_names(), ("프로젝트운영",))
            store.client._names.append("새채널")
            self.assertEqual(store.channel_names(), ("프로젝트운영",))
            self.assertEqual(store.channel_names(), ("새채널", "프로젝트운영"))

        self.assertEqual(store.client.scroll_calls, 2)


if __name__ == "__main__":
    unittest.main()
