"""파일 종류 분류.

스캔 리포트와 파서 디스패치가 같은 분류를 공유하도록 한 곳에 모아둔다.

## Google 네이티브 문서에 대한 중요한 사실

rclone은 Google Docs/Sheets/Slides를 `vnd.google-apps.*` 로 보고하지 **않는다**.
기본 `--drive-export-formats`(docx,xlsx,pptx,svg)를 적용한 뒤의 결과를 보여주므로,
lsjson에는 이미 `.docx` / `.xlsx` / `.pptx` 로 나타난다.

실제 판별 신호는 **`Size == -1`** 이다 (export 전이라 용량을 알 수 없음).
따라서:
  - 카테고리 판정은 확장자만 보면 된다 → 별도 분기가 필요 없다.
  - "이게 Google 문서인가"는 size로 판단한다 (인용 링크 형식이 달라지므로 필요).
"""
from __future__ import annotations

# Google 네이티브가 export되어 나타나는 MIME → 원본 종류.
# Size == -1 과 함께 봐야 의미가 있다 (업로드된 실제 Office 파일도 같은 MIME이므로).
GOOGLE_EXPORT_MIME = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "spreadsheets",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "presentation",
}

EXT_CATEGORY = {
    # 텍스트 레이어가 있는 문서
    ".pdf": "pdf",
    # 한글
    ".hwp": "hwp",
    ".hwpx": "hwp",
    # MS Office
    ".docx": "docx",
    ".doc": "doc_legacy",
    ".pptx": "pptx",
    ".ppt": "ppt_legacy",
    ".xlsx": "xlsx",
    ".xls": "xls_legacy",
    # 평문
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".rtf": "text",
    ".tex": "text",
    ".bib": "text",
    ".csv": "table",
    ".tsv": "table",
    ".json": "text",
    ".xml": "text",
    ".html": "text",
    ".htm": "text",
    # 이미지 (OCR 후보)
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".webp": "image",
    ".heic": "image",
    # 코드/노트북
    ".py": "code",
    ".ipynb": "notebook",
    ".c": "code",
    ".cpp": "code",
    ".h": "code",
    ".hpp": "code",
    ".m": "code",
    ".java": "code",
    ".js": "code",
    ".ts": "code",
    ".sh": "code",
    ".yaml": "code",
    ".yml": "code",
    ".toml": "code",
    # 미디어
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
    ".webm": "video",
    ".mp3": "audio",
    ".wav": "audio",
    ".m4a": "audio",
    # 압축/바이너리 데이터
    ".zip": "archive",
    ".tar": "archive",
    ".gz": "archive",
    ".bz2": "archive",
    ".xz": "archive",
    ".7z": "archive",
    ".rar": "archive",
    ".npy": "data",
    ".npz": "data",
    ".pkl": "data",
    ".pt": "data",
    ".pth": "data",
    ".h5": "data",
    ".hdf5": "data",
    ".bag": "data",
    ".pcd": "data",
    ".ply": "data",
    ".mat": "data",
}

# 1차 인덱싱 대상 — 텍스트를 바로 뽑을 수 있는 것
# (Google Docs/Sheets/Slides도 rclone이 docx/xlsx/pptx로 주므로 여기 포함된다)
# doc_legacy/ppt_legacy 도 포함된다 — parse.py가 LibreOffice 컨테이너로 변환한 뒤
# docx/pptx 파서로 넘긴다 (labrag/parse.py 의 convert_legacy_office 참고).
TEXT_CATEGORIES = {
    "pdf", "hwp", "docx", "pptx", "xlsx", "text", "table", "notebook",
    "doc_legacy", "ppt_legacy",
}

# 추가 작업(OCR/ASR)이 필요한 것 — 2단계
NEEDS_MODEL = {"image", "video", "audio"}

# 레거시 포맷 — 변환 한 단계가 더 필요 (libreoffice 등). xls_legacy는 실측 0건이라
# 변환 경로를 아직 만들지 않았다 — 나타나면 doc_legacy/ppt_legacy 와 같은 방식으로 추가.
NEEDS_CONVERT = {"xls_legacy"}


def categorize(name: str, mime_type: str | None = None) -> str:
    """파일 이름으로 카테고리를 정한다.

    rclone이 Google 네이티브를 이미 Office 확장자로 바꿔주므로 확장자만 보면 된다.
    mime_type은 확장자가 없는 파일의 보조 판단용으로만 남겨둔다.
    """
    lowered = name.lower()
    for ext, cat in EXT_CATEGORY.items():
        if lowered.endswith(ext):
            return cat
    if mime_type:
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("audio/"):
            return "audio"
        if mime_type.startswith("text/"):
            return "text"
    return "other"
