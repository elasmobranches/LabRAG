"""연구실 RAG 전역 설정.

환경변수로 덮어쓸 수 있음 (LABRAG_ 접두사). 예: LABRAG_QDRANT_URL=...
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LABRAG_", extra="ignore")

    # --- 경로 ---
    root: Path = Path(".")
    data_dir: Path = Path("data")
    rclone: Path = Path("rclone")

    # --- 드라이브 ---
    # rclone 리모트 이름. 공유 드라이브는 "gdrive,team_drive=<id>:" 형태로 확장.
    remote: str = "gdrive"
    rclone_config: Path = Path(".config/rclone/rclone.conf")

    # --- 실시간 Drive 위치 검색 ---
    live_drive_enabled: bool = True
    live_drive_root_id: str = ""
    live_drive_timeout: float = 5.0
    live_drive_remote: str = "gdrive"

    # --- 질문 자동 라우팅 ---
    # 키워드(연구실·과제·논문 등)가 없는 질문만 이 문턱을 탄다. 실제 인덱스 내용에
    # 근거한 업무 질문 21개와 잡담 24개로 실측한 결과:
    #
    #   업무 질문  0.506 ~ 0.746 (중앙 0.633)
    #   잡담      0.350 ~ 0.549 (중앙 0.449)
    #
    #   문턱 0.65 → 업무 12개 놓침, 잡담 오탐 0
    #   문턱 0.50 → 업무  0개 놓침, 잡담 오탐 4
    #
    # 두 오류의 비용이 대칭이 아니라서 낮은 쪽을 골랐다. 잡담이 검색을 타면 모델이
    # "검색된 자료에 없다"고 우아하게 답하고 끝나지만, 업무 질문을 놓치면 그럴듯하게
    # 틀린 답이 나온다 — 실측에서 "트라이포트 물류체계"를 인덱스의 정의(공항·항만·
    # 철도)가 아니라 일반 지식으로 "3자 간 거래 구조"라고 답했다.
    #
    # 두 분포가 0.50~0.55 에서 겹쳐 완벽한 값은 없다. 남는 놓침은 문턱이 아니라
    # server.py 의 안내 문구로 사용자가 알아채게 한다. 재조정할 때는 값만 바꾸지 말고
    # 양쪽 질문군을 다시 측정할 것.
    route_probe_threshold: float = 0.50
    route_probe_candidates: int = 3
    auto_route_enabled: bool = True

    # --- 외부 웹 검색 (Tavily) ---
    tavily_api_key: str = ""
    web_search_enabled: bool = True
    web_search_timeout: float = 5.0
    web_search_max_results: int = 5

    # --- 서비스 엔드포인트 (가이드의 포트 배치) ---
    qdrant_url: str = "http://localhost:6333"
    embed_url: str = "http://localhost:8080"
    rerank_url: str = "http://localhost:8081"
    vllm_url: str = "http://localhost:8000/v1"
    vllm_model: str = "qwen"
    model_id: str = "LabRAG"

    collection: str = "lab_docs"

    @property
    def scan_dir(self) -> Path:
        return self.data_dir / "scan"

    @property
    def raw_dir(self) -> Path:
        """드라이브에서 내려받은/변환한 원본 캐시."""
        return self.data_dir / "raw"

    @property
    def manifest_db(self) -> Path:
        return self.data_dir / "manifest.sqlite"

    @property
    def canonical_records(self) -> Path:
        return self.root / "config" / "canonical_records.json"

    @property
    def ancestry_db(self) -> Path:
        return self.data_dir / "drive_ancestry.sqlite"


settings = Settings()
