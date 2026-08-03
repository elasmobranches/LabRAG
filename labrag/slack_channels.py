"""질문에 언급된 Slack 채널을 실제 채널 이름으로 되돌린다.

사람은 채널 이름을 그대로 쓰지 않는다. 실제 이름이 `프로젝트운영` 이어도
`#프로젝트 운영`, `프로젝트 운영 채널`, `슬랙에서 운영 내용` 처럼 띄어쓰거나 일부만
말한다. 그래도 같은 채널을 가리키는 것이므로 찾아줘야 한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_SEPARATORS = re.compile(r"[\s_\-]+")
#: 채널 이름 중에는 일상어와 겹치는 것이 있다(`논의`, `연구과제`). 슬랙을 가리키는
#: 말이 없으면 채널 지정으로 읽지 않아, 평범한 드라이브 질문이 좁혀지는 것을 막는다.
_SLACK_CUE = re.compile(r"#|채널|슬랙|slack", re.IGNORECASE)


def _normalize(value: str) -> str:
    """구분자와 대소문자 차이를 지운다 — `연구실 운영`·`프로젝트운영`은 같은 이름이다."""
    return _SEPARATORS.sub("", value or "").casefold()


@dataclass(frozen=True)
class ChannelMatch:
    names: tuple[str, ...]
    #: "confirmed" = 채널 이름이 질문에 통째로 있음, "inferred" = 조각으로 추정, "" = 없음
    confidence: str


NO_MATCH = ChannelMatch((), "")

#: 이보다 짧은 조각은 아무 채널에나 걸려서 추정 근거가 못 된다.
MIN_FRAGMENT_CHARS = 2
_WORD_BOUNDARY = re.compile(r"[\s,./?!·]+")
_GENERIC_FRAGMENTS = {
    "관련", "업무", "내용", "기록", "자료", "최근", "최신", "진행", "상황",
    "정리해줘", "알려줘", "찾아줘", "전부", "모두", "에서", "슬랙", "slack",
    "구글", "드라이브",
}


def _fragments(query: str) -> list[str]:
    """질문의 연속된 어절 묶음을 긴 것부터 만든다.

    긴 것부터 보는 이유는 `참독 개발` 이 `참독` 보다 채널을 더 좁게 지목하기
    때문이다 — `참독` 만으로는 두 채널에 걸려 판단을 포기하게 된다.
    """
    words = [word for word in _WORD_BOUNDARY.split(query) if word]
    fragments: list[str] = []
    for size in range(len(words), 0, -1):
        for start in range(len(words) - size + 1):
            fragment = _normalize("".join(words[start:start + size]))
            if (
                len(fragment) >= MIN_FRAGMENT_CHARS
                and fragment not in _GENERIC_FRAGMENTS
            ):
                fragments.append(fragment)
    return fragments


def resolve_channels(query: str, known: Iterable[str]) -> ChannelMatch:
    if not _SLACK_CUE.search(query or ""):
        return NO_MATCH
    names = [name for name in known if _normalize(name)]
    confirmed = sorted(
        name for name in names if _explicit_channel_reference(query, name)
    )
    if confirmed:
        return ChannelMatch(tuple(confirmed), "confirmed")

    # 이름 일부만 말한 경우. 그 조각이 정확히 한 채널에만 들어맞을 때만 채택한다 —
    # 여러 채널에 걸리면 찍는 것이라 틀린 채널로 좁힐 위험이 크다.
    for fragment in _fragments(query):
        matched = [name for name in names if _mentions(name, fragment)]
        if len(matched) == 1:
            return ChannelMatch((matched[0],), "inferred")
    return NO_MATCH


def _explicit_channel_reference(query: str, name: str) -> bool:
    """`#이름` 또는 `이름 채널`처럼 사용자가 명시한 경우만 확정으로 본다."""
    normalized_query = _normalize(query)
    normalized_name = _normalize(name)
    return any(
        marker in normalized_query
        for marker in (
            f"#{normalized_name}",
            f"{normalized_name}채널",
            f"채널{normalized_name}",
        )
    )


def _mentions(name: str, fragment: str) -> bool:
    """조각이 채널 이름의 경계에 맞는가.

    이름 한가운데에 우연히 들어맞는 것은 근거로 치지 않는다 — `내용` 이
    `학위논문내용공유` 에 걸려 엉뚱한 채널로 좁히는 일이 실제로 있었다.
    사람이 이름을 줄여 부를 때는 앞이나 뒤를 잘라 부르지, 가운데만 떼어 부르지
    않는다(`프로젝트운영`→`운영`, `work-log`→`근무일지`).
    """
    normalized = _normalize(name)
    if normalized.startswith(fragment) or normalized.endswith(fragment):
        return True
    return any(
        _normalize(segment).startswith(fragment)
        for segment in _SEPARATORS.split(name) if segment
    )
