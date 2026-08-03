from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

MONTH_PATTERN = re.compile(
    r"(?:(?P<year>20\d{2})\s*년\s*)?(?P<month>0?[1-9]|1[0-2])\s*월"
)
RECENT_PATTERN = re.compile(r"최근\s*(?P<count>\d+)\s*(?P<unit>일|주)")
CHANNEL_PATTERN = re.compile(
    r"#([0-9A-Za-z가-힣_-]+?)(?=(?:에서|으로|의|와|과|,|\s|$))"
)
PERSON_PATTERN = re.compile(r"([가-힣]{2,4})\s*님")
DRIVE_PATTERN = re.compile(r"(?:Google\s*)?Drive|구글\s*드라이브|드라이브", re.I)
SLACK_PATTERN = re.compile(r"Slack|슬랙", re.I)
LOCATION_PATTERN = re.compile(
    r"어디|어느\s*폴더|위치|경로|링크|"
    r"\.(?:pdf|docx?|xlsx?|pptx?|hwp|hwpx|csv|txt)\b",
    re.I,
)

STOPWORDS = {
    "업무", "기록", "내용", "자료", "관련", "에서", "으로", "를", "을", "의",
    "찾아줘", "정리해줘", "요약해줘", "알려줘", "Google", "Drive", "Slack",
    "구글", "드라이브", "슬랙", "담당자별", "날짜별", "표로", "최근", "지난달",
    "이번", "이번달",
}


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
    label: str = ""


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    retrieval_query: str
    task: str
    period: DateRange | None
    sources: tuple[str, ...]
    topics: tuple[str, ...]
    people: tuple[str, ...]
    channels: tuple[str, ...]
    output_format: str
    year_inferred: bool = False

    def prompt_block(self) -> str:
        lines = ["[질문 해석]", f"업무 유형: {self.task}"]
        if self.period:
            lines.append(
                f"기간: {self.period.start.isoformat()} 이상, "
                f"{self.period.end.isoformat()} 미만"
            )
        if self.sources:
            names = {"drive": "Google Drive", "slack": "Slack"}
            lines.append("출처: " + ", ".join(names[s] for s in self.sources))
        if self.topics:
            lines.append("주제: " + ", ".join(self.topics))
        if self.people:
            lines.append("담당자: " + ", ".join(self.people))
        if self.channels:
            lines.append("채널: " + ", ".join(f"#{c}" for c in self.channels))
        lines.append(f"출력 형식: {self.output_format}")
        return "\n".join(lines)


def _next_month(year: int, month: int) -> date:
    return date(year + (month == 12), month % 12 + 1, 1)


def _parse_period(query: str, today: date) -> tuple[DateRange | None, bool]:
    match = MONTH_PATTERN.search(query)
    if match:
        year_inferred = match.group("year") is None
        year = int(match.group("year") or today.year)
        month = int(match.group("month"))
        start = date(year, month, 1)
        return DateRange(start, _next_month(year, month), f"{year}년 {month}월"), year_inferred

    if "지난달" in query:
        end = date(today.year, today.month, 1)
        previous = end - timedelta(days=1)
        start = date(previous.year, previous.month, 1)
        return DateRange(start, end, f"{start.year}년 {start.month}월"), True

    if "이번달" in query or "이번 달" in query:
        start = date(today.year, today.month, 1)
        return DateRange(start, _next_month(today.year, today.month), f"{today.year}년 {today.month}월"), True

    match = RECENT_PATTERN.search(query)
    if match:
        count = int(match.group("count"))
        days = count * (7 if match.group("unit") == "주" else 1)
        end = today + timedelta(days=1)
        return DateRange(today - timedelta(days=max(days - 1, 0)), end, match.group(0)), False

    return None, False


def _topics(query: str, people: tuple[str, ...], channels: tuple[str, ...]) -> tuple[str, ...]:
    cleaned = MONTH_PATTERN.sub(" ", query)
    cleaned = RECENT_PATTERN.sub(" ", cleaned)
    cleaned = CHANNEL_PATTERN.sub(" ", cleaned)
    cleaned = DRIVE_PATTERN.sub(" ", cleaned)
    cleaned = SLACK_PATTERN.sub(" ", cleaned)
    for person in people:
        cleaned = re.sub(rf"{re.escape(person)}\s*님", " ", cleaned)
    tokens = re.findall(r"[0-9A-Za-z가-힣][0-9A-Za-z가-힣_-]*", cleaned)
    result: list[str] = []
    for token in tokens:
        token = re.sub(r"(?:에서|관련|업무|기록|내용|자료|부터|까지)$", "", token)
        if len(token) < 2 or token in STOPWORDS or token.isdigit():
            continue
        if token not in result:
            result.append(token)
    return tuple(result)


def parse_query_intent(query: str, *, today: date | None = None) -> QueryIntent:
    raw_query = query
    current = today or datetime.now(KST).date()
    period, year_inferred = _parse_period(query, current)

    sources: list[str] = []
    if DRIVE_PATTERN.search(query):
        sources.append("drive")
    if SLACK_PATTERN.search(query):
        sources.append("slack")

    people = tuple(dict.fromkeys(PERSON_PATTERN.findall(query)))
    channels = tuple(dict.fromkeys(CHANNEL_PATTERN.findall(query)))

    has_person_group = "담당자별" in query
    has_table = bool(re.search(r"(?:표로|표\s*형식|테이블)", query))
    if has_person_group and has_table:
        output_format = "table_by_person"
    elif "날짜별" in query or "시간순" in query:
        output_format = "chronological"
    elif has_person_group:
        output_format = "by_person"
    elif has_table:
        output_format = "table"
    else:
        output_format = "summary"

    task = "location" if LOCATION_PATTERN.search(query) else "content"
    retrieval_query = query
    if year_inferred and period and not re.search(r"20\d{2}\s*년", query):
        if MONTH_PATTERN.search(query):
            retrieval_query = f"{period.start.year}년 {query}"
        elif "지난달" in query or "이번달" in query or "이번 달" in query:
            retrieval_query = f"{period.start.year}년 {period.start.month}월 {query}"

    return QueryIntent(
        raw_query=raw_query,
        retrieval_query=retrieval_query,
        task=task,
        period=period,
        sources=tuple(sources),
        topics=_topics(query, people, channels),
        people=people,
        channels=channels,
        output_format=output_format,
        year_inferred=year_inferred,
    )
