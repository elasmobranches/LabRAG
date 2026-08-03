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
    # 명시적인 업무 키워드가 없는 질문만 이 문턱을 사용한다. 내부 질문을 일반
    # 지식으로 답하는 오류가 잡담을 한 번 검색하는 오류보다 비싸다고 보고 낮게
    # 잡았다. 코퍼스나 임베딩 모델을 바꾸면 업무 질문과 일반 질문을 함께 재평가해야
    # 하며, 값만 단독으로 조정하지 않는다.
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
