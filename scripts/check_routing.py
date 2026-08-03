#!/usr/bin/env python
"""돌아가는 서버에 실제 질문을 던져 라우팅과 검색 범위를 점검한다.

## 왜 unittest 로 부족한가

unittest 는 가짜 store 로 로직만 본다. 여기서 잡히는 것은 그게 아니라 **실제 데이터와
어긋나는 문제**다. 실제로 이 점검에서 나온 것들:

  - 실제 채널명은 `프로젝트운영` 인데 사람은 `#프로젝트 운영` 이라고 쓴다. 정확일치
    필터가 Slack 결과를 전부 버려 375건짜리 채널이 0건으로 나왔다.
  - `#` 이 없으면 채널 지정을 무시해서, 536건짜리 `work-log` 를 두고 엉뚱한
    채널에서 6건을 가져왔다.
  - Open WebUI 가 대화 턴마다 보내는 제목·태그 생성 요청이 진짜 질문으로 오인돼
    드라이브 24만 건 검색과 Tavily 호출을 유발했다.

셋 다 코드만 봐서는 안 보이고, 실제 채널 목록·실제 인덱스와 대조해야 드러난다.

## 무엇을 보는가

라우팅 모드가 기대와 같은지, 그리고 채널을 지정한 질문이 **맞는 채널에서** 답을
가져오는지 본다. 답변 내용의 품질은 보지 않는다 — 그건 scripts/eval.py 몫이다.

    python scripts/check_routing.py                 # 전체
    python scripts/check_routing.py --only channel  # 채널 관련만
    python scripts/check_routing.py --verbose       # 출처까지 표시

서버가 떠 있어야 한다. 종료 코드는 실패가 있으면 1 이다.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8100"

# (그룹, 질문, 기대 모드, 기대 채널)
#   기대 채널 None  = 채널을 보지 않음
#   기대 채널 ""    = 어떤 채널로도 좁히면 안 됨(오탐 확인)
CASES: list[tuple[str, str, str, str | None]] = [
    # ── 채널을 지정한 질문: 맞는 채널에서 가져와야 한다 ──────────────────
    ("channel", "슬랙의 #프로젝트 운영 채널에서 최근 한 내용좀 알려줘", "rag", "프로젝트운영"),
    ("channel", "프로젝트 운영 채널 내용 뭐 있나?", "rag", "프로젝트운영"),
    ("channel", "슬랙에서 운영 내용 알려줘", "rag", "프로젝트운영"),
    ("channel", "슬랙 운영 채널 최근 내용 알려줘", "rag", "프로젝트운영"),
    ("channel", "슬랙 근무일지 채널 최근 내용 알려줘", "rag", "work-log"),
    ("channel", "슬랙 참독 개발 채널 최근 내용 알려줘", "rag", "project-development"),
    ("channel", "슬랙의 #프로젝트운영 채널에서 최근 내용 알려줘", "rag", "프로젝트운영"),
    # 없는 채널을 대도 0건으로 끝나면 안 된다. 채널이 비었다고 오해하게 된다.
    ("channel", "슬랙의 #없는채널 에서 최근 내용 알려줘", "rag", None),

    # ── 채널 이름과 겹치는 일상어: 채널로 좁히면 안 된다 ────────────────
    ("noise", "우리 연구과제 진행상황 알려줘", "rag", ""),
    ("noise", "지난주 논의된 내용 정리해줘", None, ""),
    ("noise", "슬랙에서 최근 논의된 내용 알려줘", "rag", ""),
    ("noise", "2026년 06월 토마토 관련 업무 기록을 구글 드라이브와 슬랙에서 전부 정리해줘", "rag", ""),
    ("noise", "토마토 관련 자료 찾아줘", None, ""),

    # ── 인수인계 문서가 지키라고 못박은 라우팅 ──────────────────────────
    ("route", "Conference 논문 내용을 요약해줘", "rag", None),
    ("route", "우리 연구실 제품 장단점", "rag", None),
    ("source", "구글 드라이브에 있는 토마토 논문", "rag", ""),
    ("route", "농업용 로봇 정책이 어떻게 바뀌었어?", "web", None),
    ("route", "스마트팜 제품 장단점은?", "web", None),
    ("route", "연구실 토마토 과제와 웹 최신 동향", "rag_web", None),
    ("route", "Slack과 인터넷에서 최근 부산 내용", "rag_web", None),
    ("route", "Conference_2026.pdf 어디 있어?", "location", None),
    # 출처를 웹이라고 말했으면 "찾아 줘"를 위치 질문으로 읽지 않는다. 예전에는
    # location 으로 갔다가 파일을 못 찾고 rag_web 으로 넘어와 헛돌았다.
    ("route", "웹에서 토마토 재배법 찾아줘", "web", None),
    # 반대로 웹이 주제일 뿐이면 웹 검색을 붙이지 않는다. 위치 질문으로 출발하지만
    # 그런 파일이 없어 rag 로 폴백한다 — 웹까지 가면(rag_web) 잘못이다.
    ("route", "웹 크롤러 코드 찾아줘", "rag", None),
    ("route", "연구실 웹 크롤러 코드 설명해줘", "rag", None),
    # 시의성 단어만 있어 애매하면 문턱값이 내부냐 웹이냐를 가른다.
    ("route", "최근에 어떻게 됐어?", "web", None),
    # 생활·추천 표현은 벡터 문턱을 타기 전에 일반 답변으로 보낸다. 실제 내부 말뭉치에
    # 영화 추천·점심 채널이 있어 0.9대 점수가 나오므로 문턱만으로는 분리가 안 됐다.
    ("route", "기타 배우려면 뭐부터 해야 돼?", "general", None),
    ("route", "밥 뭐 먹을까?", "general", None),
    ("route", "운동 루틴 추천해줘", "general", None),
    ("route", "재미있는 영화 추천해줘", "general", None),
    ("route", "친구 생일 선물 추천해줘", "general", None),

    # ── 키워드가 없는 업무 질문: 문턱 0.65 시절 전부 general 로 샜다 ────────
    # 인덱스에 답이 있는데도 그럴듯하게 틀린 일반 지식으로 답하던 것들이다.
    ("probe", "오로라 배치체계가 뭐야?", "rag", None),
    ("probe", "공기 흐름이 증산에 어떤 영향을 준대?", "rag", None),
    ("probe", "예시 식당 예약 몇 시야?", "rag", None),
    ("probe", "오이 어노테이션 작업은 어떻게 진행됐어?", "rag", None),
    ("probe", "예시 서비스 회원가입 안내가 언제 있었지?", "rag", None),

    # ── Open WebUI 가 대화 턴마다 보내는 부가 요청: 검색하면 안 된다 ────
    ("task", "### Task:\nGenerate a concise title summarizing the chat history.\n"
             "### Guidelines:\n- Keep it short.\n### Chat History:\n<chat_history>\n"
             "USER: 우리 연구실 토마토 과제와 농업용 로봇 정책 알려줘\n</chat_history>",
     "general", None),
    ("task", "### TASK:\nGenerate a concise title summarizing the chat history.\n"
             "USER: 우리 연구실 토마토 과제", "general", None),
    ("task", "Generate a concise title summarizing the chat history:\n"
             "우리 연구실 토마토 과제", "general", None),
]


def ask(base_url: str, question: str) -> dict:
    payload = json.dumps({
        "model": "LabRAG",
        "messages": [{"role": "user", "content": question}],
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/search", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def channels_of(result: dict) -> collections.Counter:
    """결과가 어느 Slack 채널에서 왔는지 센다. Drive 는 세지 않는다."""
    found: collections.Counter = collections.Counter()
    for row in result.get("results") or []:
        parts = [part.strip() for part in (row.get("citation") or "").split("·")]
        if len(parts) > 1 and parts[0] == "Slack" and parts[1].startswith("#"):
            found[parts[1][1:]] += 1
    return found


def check(case, result) -> list[str]:
    group, _question, want_mode, want_channel = case
    problems = []
    mode = result.get("mode")
    if want_mode is not None and mode != want_mode:
        problems.append(f"모드 {mode} (기대 {want_mode})")

    rows = result.get("results") or []
    channels = channels_of(result)
    if want_channel:
        if not rows:
            problems.append("결과 0건")
        elif set(channels) - {want_channel}:
            problems.append(
                f"다른 채널 섞임 {dict(channels)} (기대 {want_channel})"
            )
        elif not channels:
            problems.append(f"#{want_channel} 결과 없음")
    elif want_channel == "":
        if channels and (group == "source" or len(channels) == 1):
            problems.append(f"채널로 좁혀짐 {dict(channels)}")
    elif want_channel is None and want_mode in ("rag", "rag_web"):
        if not rows:
            problems.append("결과 0건")
    if group == "noise" and any(
        "채널로 이해" in note for note in result.get("notes") or []
    ):
        problems.append(f"채널 오인 안내 {result['notes']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--only", help="그룹만 실행 (channel/noise/route/task)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = [c for c in CASES if not args.only or c[0] == args.only]
    if not cases:
        print(f"'{args.only}' 그룹이 없다. 있는 그룹: "
              f"{sorted({c[0] for c in CASES})}")
        return 2

    failures = 0
    for case in cases:
        group, question, _want_mode, _want_channel = case
        label = question.splitlines()[0][:52]
        try:
            result = ask(args.url, question)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  ERROR [{group}] {label} — 서버 응답 없음: {exc}")
            failures += 1
            continue
        problems = check(case, result)
        status = "FAIL" if problems else "ok  "
        failures += bool(problems)
        print(f"  {status} [{group}] {label}")
        if problems:
            for problem in problems:
                print(f"         → {problem}")
        if args.verbose:
            print(f"         mode={result.get('mode')} "
                  f"reason={result.get('route_reason')} "
                  f"결과={len(result.get('results') or [])}건 "
                  f"채널={dict(channels_of(result))}")

    print(f"\n{len(cases) - failures}/{len(cases)} 통과")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
