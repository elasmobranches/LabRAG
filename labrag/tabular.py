"""숫자 표·로그 파일 판별과 요약.

## 왜 필요한가

측정 결과, 인덱스 청크의 50%가 두 종류에서 나왔다:

    robot_pose_output.txt   2,445 청크   로봇 포즈 추정 실행 로그
    RM3_month5.csv                     4,329 청크   온실 센서 월별 측정값

이런 파일은 산문처럼 청킹해도 검색에 쓸 수 없다. "온실 온도가 몇 도였어?"를 숫자
1,200자 조각에서 답할 수는 없다. 사용자 표현대로 **"결국 파일을 우리가 찾아봐야"**
하는 자료이므로, RAG 가 해야 할 일은 답을 만드는 것이 아니라 **맞는 파일로 데려다주는
것**이다. 그러려면 필요한 정보는 위치·컬럼 구성·기간·행 수다.

## 확장자로 자르지 않는 이유

`.csv` 를 일괄 제외하면 손해가 난다. 실제로 검색 상위에 정당하게 올라온
`simulation_output_dictionary.csv` 는 온실 모델 출력 변수의 **이름 사전**이라
단어로 채워진 유용한 문서다. 반대로 `.txt` 인데 숫자 로그인 파일도 있다.

그래서 **내용을 보고 판단한다** — 샘플 줄의 숫자 토큰 비율. 크기도 잘못된 기준이다
(145MB `plant empowerment.pdf` 는 가치 있고 4MB `RM3_month5.csv` 는 아니다).
"""
from __future__ import annotations

import csv
import io
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

_NUMERIC = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
# 날짜·시각도 수치 데이터로 센다. 센서 로그의 첫 컬럼이 대개 이 형태다.
_DATETIME = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([T ]\d{1,2}:\d{2}(:\d{2})?)?$"
    r"|^\d{1,2}:\d{2}(:\d{2})?$"
)
_TOKEN_SPLIT = re.compile(r"[,\t;|\s]+")
# CSV 값은 따옴표로 감싸져 오는 경우가 많다 ("15.29"). 이걸 벗기지 않으면
# 숫자 판별이 전부 실패한다 — 실제로 4MB 센서 CSV 가 '산문'으로 판정됐다.
_WRAP = '"\'`()[]{}%'


# 로그에서 흔한 '라벨:값' / '라벨=값' 쌍. 실측 예: 로봇 포즈 로그가
#   2025-04-02 02:20:27  x:0.002  y:-0.002  z:0.096
# 형태여서, 이걸 인식하지 않으면 숫자 비율이 40%로 떨어져 '산문'으로 오판된다.
_LABELED = re.compile(r"^[A-Za-z_][\w.\[\]]{0,20}[:=](?P<v>.+)$")


def _is_numeric(token: str) -> bool:
    t = token.strip().strip(_WRAP).strip()
    if not t:
        return False
    if _NUMERIC.match(t) or _DATETIME.match(t):
        return True
    if (m := _LABELED.match(t)):
        v = m.group("v").strip().strip(_WRAP).strip()
        return bool(_NUMERIC.match(v) or _DATETIME.match(v))
    return False

# 판별 파라미터
SAMPLE_LINES = 400          # 앞에서 이만큼만 보고 판단한다
NUMERIC_RATIO = 0.55        # 토큰의 이 비율 이상이 숫자면 '숫자 표'
MIN_LINES = 150             # 이보다 짧으면 요약해서 얻는 이득이 없다
MAX_SCAN_BYTES = 200 * 1024 * 1024   # 이보다 크면 행 수 세기를 생략


MIN_JSON_RECORDS = 30       # 이보다 적으면 설정 파일일 수 있으니 산문으로 둔다
MAX_JSON_BYTES = 20 * 1024 * 1024


@dataclass
class Profile:
    n_lines: int | None        # None = 너무 커서 세지 않음
    size: int
    header: list[str] | None   # CSV 컬럼명 또는 JSON 키 경로
    numeric_ratio: float
    first_lines: list[str]
    last_lines: list[str]
    delimiter: str | None
    kind: str = "table"        # table | log | json
    n_records: int | None = None


def _tokens(line: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT.split(line.strip()) if t]


def _sniff_delimiter(sample: str) -> str | None:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except Exception:
        return None


def _flatten_keys(obj, prefix: str = "", out: list[str] | None = None,
                  depth: int = 0) -> list[str]:
    """JSON 레코드의 키 경로를 뽑는다 (odom.position.x 형태)."""
    out = [] if out is None else out
    if depth > 4 or len(out) > 60:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                _flatten_keys(v, p, out, depth + 1)
            else:
                out.append(p)
    elif isinstance(obj, list) and obj:
        _flatten_keys(obj[0], f"{prefix}[]", out, depth + 1)
    return out


def _json_profile(path: Path, size: int) -> Profile | None:
    """JSON / JSON Lines 를 구조적으로 판별한다.

    숫자 비율로는 잡히지 않는다 — `{"x":` 같은 키 토큰이 비숫자로 세어져서
    센서 IMU 로그(JSON Lines)가 43%로 떨어져 '산문'으로 오판됐다.
    JSON 은 애초에 기계가 직렬화한 데이터이므로, 숫자 비율이 아니라
    **스키마(키 경로)와 레코드 수**를 요약하는 것이 맞다.
    """
    import json as _json

    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(8192)
    except OSError:
        return None
    # 앞에 JSON 이 아닌 머리글 줄이 붙어 있을 수 있다 — 실제로 IMU 로그의 첫 줄이
    # 'base' 였고, 첫 글자만 보면 JSON 판별이 실패한다. 앞 몇 줄을 훑어본다.
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    rec, first_line, is_jsonl = None, "", False
    for ln in lines[:5]:
        if ln[:1] not in ("{", "["):
            continue
        try:
            cand = _json.loads(ln)
        except Exception:
            continue
        if isinstance(cand, dict):
            rec, first_line, is_jsonl = cand, ln, True
            break
    if not is_jsonl and not any(ln[:1] in ("{", "[") for ln in lines[:5]):
        return None

    if is_jsonl:
        n = 0
        last = deque(maxlen=2)
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(("{", "[")):
                    n += 1
                    last.append(line[:300])
        if n < MIN_JSON_RECORDS:
            return None
        return Profile(
            n_lines=n, size=size, header=_flatten_keys(rec), numeric_ratio=1.0,
            first_lines=[first_line[:300]], last_lines=list(last),
            delimiter=None, kind="json", n_records=n,
        )

    # ── 통짜 JSON: 레코드 배열을 찾는다 ──
    if size > MAX_JSON_BYTES:
        return None
    try:
        data = _json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

    # 레코드 배열을 재귀적으로 찾는다. 최상위에만 있다고 가정하면 놓친다 —
    # 실제로 IMU 데이터가 {"front_poseX": {"original": [...]}} 처럼 두 단계
    # 안쪽에 배열을 두고 있었다.
    def find_records(obj, path: str = "", depth: int = 0):
        best: tuple[str, list] | None = None
        if depth > 3:
            return None
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                best = (path, obj)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else str(k)
                got = find_records(v, p, depth + 1)
                if got and (best is None or len(got[1]) > len(best[1])):
                    best = got
        return best

    got = find_records(data)
    if got is None:
        return None
    key, records = got
    if len(records) < MIN_JSON_RECORDS:
        return None

    header = _flatten_keys(records[0], f"{key}[]" if key else "")
    return Profile(
        n_lines=None, size=size, header=header, numeric_ratio=1.0,
        first_lines=[_json.dumps(records[0], ensure_ascii=False)[:300]],
        last_lines=[], delimiter=None, kind="json", n_records=len(records),
    )


def profile(path: Path) -> Profile | None:
    """숫자 표/로그/JSON 데이터로 보이면 Profile, 산문이면 None.

    한 번의 순회로 행 수와 마지막 줄까지 얻는다.
    """
    size = path.stat().st_size
    if (jp := _json_profile(path, size)) is not None:
        return jp

    first: list[str] = []
    last: deque[str] = deque(maxlen=3)
    n_lines = 0
    numeric = 0
    total = 0

    scan_rows = size <= MAX_SCAN_BYTES
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i < SAMPLE_LINES:
                    stripped = line.rstrip("\n")
                    if len(first) < 6 and stripped.strip():
                        first.append(stripped[:400])
                    # 첫 줄은 헤더일 수 있으니 숫자 비율 계산에서 제외
                    if i > 0:
                        toks = _tokens(stripped)
                        total += len(toks)
                        numeric += sum(1 for t in toks if _is_numeric(t))
                elif not scan_rows:
                    break
                if scan_rows:
                    last.append(line.rstrip("\n")[:400])
                n_lines = i + 1
    except OSError:
        return None

    if total == 0:
        return None
    ratio = numeric / total
    if ratio < NUMERIC_RATIO:
        return None                      # 산문 — 평소대로 청킹한다
    if scan_rows and n_lines < MIN_LINES:
        return None                      # 짧으면 그냥 넣는 편이 낫다

    sample = "\n".join(first[:5])
    delim = _sniff_delimiter(sample)
    header = None
    if delim and first:
        cells = [c.strip().strip(_WRAP).strip() for c in first[0].split(delim)]
        # 첫 줄 토큰이 대부분 숫자면 헤더가 아니라 데이터다
        if cells and sum(1 for c in cells if _is_numeric(c)) < len(cells) * 0.5:
            header = [c for c in cells[:60] if c]

    return Profile(
        n_lines=n_lines if scan_rows else None,
        size=size,
        header=header,
        numeric_ratio=ratio,
        first_lines=first[:4],
        last_lines=list(last),
        delimiter=delim,
    )


def _human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if v < 1024 or unit == "GB":
            return f"{v:.0f}{unit}" if unit == "B" else f"{v:.1f}{unit}"
        v /= 1024
    return f"{v:.1f}GB"


def summarize(path: Path, p: Profile, file_name: str) -> str:
    """검색에 넣을 요약 텍스트. '이 파일이 무엇이고 어떤 컬럼이 있는지'를 담는다."""
    kindname = {"json": "JSON 데이터 파일", "log": "로그 파일"}.get(p.kind, "수치 데이터 파일")
    out = [
        f"[데이터 파일] {file_name}",
        f"{kindname}이다. 내용은 인덱싱하지 않았으므로, "
        f"값이 필요하면 원본 파일을 직접 열어야 한다.",
    ]
    scale = f"크기 {_human(p.size)}"
    if p.n_records is not None:
        scale += f" · 레코드 {p.n_records:,}개"
    elif p.n_lines is not None:
        scale += f" · {p.n_lines:,}행"
    else:
        scale += " · 행 수 미측정(너무 큼)"
    out.append(scale)
    if p.header:
        label = "필드" if p.kind == "json" else "컬럼"
        out.append(f"{label} {len(p.header)}개: " + ", ".join(p.header))
    elif p.delimiter:
        out.append(f"구분자 '{p.delimiter}' · 헤더 줄 없음")
    if p.first_lines:
        out.append("첫 줄:")
        out += [f"  {ln[:200]}" for ln in p.first_lines[:3]]
    if p.last_lines:
        out.append("마지막 줄:")
        out += [f"  {ln[:200]}" for ln in p.last_lines[-2:]]
    return "\n".join(out)
