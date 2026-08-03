"""Block → Chunk. 검색 단위를 만든다.

두 가지 설계 판단이 들어있다.

1. **블록 경계를 존중한다.** 페이지나 섹션이 다른 내용을 한 청크에 섞으면
   "○○.pdf 3페이지" 같은 인용이 부정확해진다. 같은 (page, section)을 가진
   블록끼리만 병합한다.

2. **임베딩용 텍스트에 출처 헤더를 붙인다.** 청크만 떼어놓으면 "무슨 문서의
   어느 대목인지"라는 정보가 사라져 검색 정확도가 떨어진다. 파일명·섹션을
   앞에 붙여 임베딩하고, 사용자에게 보여줄 때는 본문만 쓴다.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .drive import DriveFile
from .parse import Block

# 문자 기준. 한국어는 1자≈1토큰에 가까워서 1200자면 대략 1000토큰 안쪽이다.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# 이보다 짧은 청크는 검색 노이즈만 된다 (표지, 페이지 번호만 있는 페이지 등)
MIN_CHUNK_CHARS = 60

# 청크 ID 생성용 고정 네임스페이스. 절대 바꾸지 말 것 —
# 바뀌면 모든 포인트 ID 가 달라져서 기존 인덱스가 중복으로 남는다.
_POINT_NAMESPACE = uuid.UUID("6ba7b812-9dad-11d1-80b4-00c04fd430c8")

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    # 한국어 문장 종결(다./요.)과 영어 마침표를 모두 경계 후보로 둔다
    separators=["\n\n", "\n", "다. ", "요. ", ". ", " ", ""],
    keep_separator=True,
)


@dataclass
class Chunk:
    text: str                      # 사용자에게 보여줄 본문
    file_id: str
    file_name: str
    file_url: str
    root: str                      # 어느 드라이브에서 왔는지
    path: str                      # 드라이브 내 경로
    mod_time: str
    index: int                     # 파일 내 청크 순번
    page: int | None = None
    section: str | None = None
    label: str | None = None
    file_context: str | None = None  # 파일 단위 contextual enrichment (labrag/contextual.py)
    id: str = field(init=False)     # Qdrant 포인트 ID (재인덱싱 시 안정적이어야 함)

    def __post_init__(self) -> None:
        # 파일 ID + 순번으로 결정론적 ID를 만든다. 같은 파일을 다시 인덱싱하면
        # 같은 포인트를 덮어쓰므로 중복이 생기지 않는다.
        # Qdrant 포인트 ID 는 부호없는 정수 또는 UUID 만 받는다 (해시 hex 문자열은 거부).
        # 그래서 uuid5 를 쓴다 — 같은 입력이면 항상 같은 UUID 가 나온다.
        self.id = str(uuid.uuid5(_POINT_NAMESPACE, f"{self.file_id}:{self.index}"))

    @property
    def embed_text(self) -> str:
        """임베딩에 넣을 텍스트 — 출처 헤더 + (있으면) 파일 맥락 + 본문."""
        head = self.file_name
        if self.section:
            head += f" · {self.section}"
        if self.page is not None:
            head += f" · {self.page}p"
        if self.file_context:
            return f"[{head}]\n{self.file_context}\n{self.text}"
        return f"[{head}]\n{self.text}"

    @property
    def citation(self) -> str:
        """답변에 붙일 사람이 읽는 출처 표기."""
        parts = [self.file_name]
        if self.page is not None:
            parts.append(f"{self.page}쪽")
        if self.section:
            parts.append(f"§{self.section}")
        if self.label:
            parts.append(self.label)
        return " · ".join(parts)

    def payload(self) -> dict:
        """Qdrant에 저장할 페이로드."""
        return {
            "text": self.text,
            "file_id": self.file_id,
            "file_name": self.file_name,
            "file_url": self.file_url,
            "root": self.root,
            "path": self.path,
            "mod_time": self.mod_time,
            "index": self.index,
            "page": self.page,
            "section": self.section,
            "label": self.label,
            "citation": self.citation,
            "file_context": self.file_context,
        }


def _mergeable(a: Block, b: Block) -> bool:
    """두 블록을 한 청크에 합쳐도 인용이 정확한가."""
    return a.page == b.page and a.section == b.section and a.label == b.label


def _merge_blocks(blocks: Iterable[Block]) -> list[Block]:
    """작은 블록들을 CHUNK_SIZE 근처까지 병합한다 (같은 위치인 것만)."""
    merged: list[Block] = []
    for b in blocks:
        text = b.text.strip()
        if not text:
            continue
        if merged and _mergeable(merged[-1], b) and \
                len(merged[-1].text) + len(text) + 1 <= CHUNK_SIZE:
            merged[-1] = Block(
                text=merged[-1].text + "\n" + text,
                page=b.page, section=b.section, label=b.label,
            )
        else:
            merged.append(Block(text=text, page=b.page, section=b.section, label=b.label))
    return merged


def _merge_undersized(blocks: list[Block]) -> list[Block]:
    """MIN_CHUNK_CHARS 미만인 블록은 섹션이 달라도 다음 블록과 합쳐서, 통째로
    버려지는 걸 막는다.

    실측: 랩미팅 회의록에서 한 사람의 한 주 업데이트 전체가 60자 미만이라는
    이유로 조용히 삭제되고 있었다 (예: "이연구 · 부산 출장 출장후기
    스마트농업 온라인 경진대회 소감 논문 작성중" 전체가 사라짐). 오늘 회의록을
    사람별 섹션으로 나누는 개선을 하면서 섹션이 더 잘게 쪼개졌고, 그만큼 어느
    한 사람의 업데이트가 그 자체로 60자를 못 넘기는 경우가 늘어 이 문제가
    더 커졌다 — 구조를 더 정확히 인식할수록 조용히 버려지는 조각도 늘어난
    역설이다. `_mergeable`(같은 page/section/label)만 따르는 `_merge_blocks`는
    이걸 못 잡는다 — 섹션이 다르면 절대 안 합치기 때문이다. 여기서는 반대로
    "짧으면 섹션이 달라도 합친다"로 손실을 막는다.
    """
    if not blocks:
        return blocks
    out: list[Block] = [blocks[0]]
    for b in blocks[1:]:
        if len(out[-1].text) < MIN_CHUNK_CHARS:
            prev = out[-1]
            out[-1] = Block(
                text=prev.text + "\n" + b.text,
                page=prev.page if prev.page == b.page else None,
                section=_combine(prev.section, b.section),
                label=prev.label or b.label,
            )
        else:
            out.append(b)
    if len(out) >= 2 and len(out[-1].text) < MIN_CHUNK_CHARS:
        last = out.pop()
        prev = out[-1]
        out[-1] = Block(
            text=prev.text + "\n" + last.text,
            page=prev.page if prev.page == last.page else None,
            section=_combine(prev.section, last.section),
            label=prev.label or last.label,
        )
    return out


def _combine(a: str | None, b: str | None) -> str | None:
    """섹션이 다른 두 블록을 합칠 때 라벨도 같이 합친다.

    한쪽만 유지하면(예: "이연구"만 남기고 뒤에 합쳐진 "이도현" 내용을 숨기면)
    인용이 실제 내용과 안 맞게 된다 — 검색은 본문 전체를 보고 찾아오는데
    인용은 앞사람 이름만 보여주면 "이 사람이 이 얘기를 했다"고 잘못 알려준다.
    """
    if not a:
        return b
    if not b or a == b:
        return a
    return f"{a} / {b}"


def chunk_file(f: DriveFile, blocks: list[Block], file_context: str | None = None) -> list[Chunk]:
    """한 파일의 블록들을 청크 리스트로 만든다.

    file_context 는 파일 단위 contextual enrichment(labrag/contextual.py, 파일럿 단계)
    결과다 — 없으면(기본값) 기존과 완전히 동일하게 동작한다.
    """
    chunks: list[Chunk] = []
    idx = 0
    for block in _merge_undersized(_merge_blocks(blocks)):
        pieces = _splitter.split_text(block.text) if len(block.text) > CHUNK_SIZE \
            else [block.text]
        for piece in pieces:
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                text=piece,
                file_id=f.id, file_name=f.name, file_url=f.web_url,
                root=f.root, path=f.path, mod_time=f.mod_time,
                index=idx, page=block.page, section=block.section, label=block.label,
                file_context=file_context,
            ))
            idx += 1
    return chunks
