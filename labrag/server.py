"""RAG 백엔드 — OpenAI 호환 API.

Open WebUI 는 이 서버를 그냥 '모델' 하나로 인식한다. 그래서 UI 를 직접 만들 필요가
없고, 검색 로직·인용 표기는 전부 우리가 통제한다. Open WebUI 내장 RAG 를 쓰지 않는
이유가 이것이다 — 드라이브 증분 동기화, 섹션/페이지 메타데이터, 정확한 원본 링크를
내장 기능으로는 만들 수 없다.

    uvicorn labrag.server:app --host 0.0.0.0 --port 8100

Open WebUI 에서 OPENAI_API_BASE_URL 을 http://host.docker.internal:8100/v1 로 두면
모델 목록에 'LabRAG' 이 나타난다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .catalog import ManifestCatalog, load_canonical_config
from .config import settings
from .docsynopsis import DOC_COLLECTION
from .intent import parse_query_intent
from .location import maybe_locate_hybrid, render_hybrid
from .models import Models
from .rclone_auth import load_rclone_oauth
from .rag import (
    CANDIDATES,
    TOP_K,
    build_general_messages,
    build_hybrid_messages,
    build_messages,
    build_web_messages,
    retrieve,
    web_sources_block,
)
from .router import (
    WEB_SOURCE_HINT,
    RouteDecision,
    classify_query,
    probe_internal_relevance,
    resolve_probe,
    split_hybrid_queries,
)
from .store import Store
from .web_search import (
    WEB_CONTENT_PREVIEW_CHARS,
    WebHit,
    search_web,
    web_search_metrics,
)

MODEL_ID = settings.model_id
SLACK_COLLECTION = "lab_slack"
SLACK_PARENT_COLLECTION = "lab_slack_parent"
SLACK_UNAVAILABLE_MESSAGE = (
    "Slack 검색 저장소에 일시적으로 연결할 수 없습니다. "
    "잠시 후 다시 시도해 주세요."
)

# ── thinking(추론 과정)을 기본으로 끄는 이유 ──────────────────────────────
# Qwen3.6 은 추론 모델이라 기본적으로 사고 과정을 먼저 생성한다. 그런데 이 과정이
# reasoning_content 로 분리되지 않고 content 에 그대로 섞여 나온다:
#
#   "Thinking Process:  1. Analyze the Request: * Question: ... * Constraint: ..."
#
# 그러면 (1) 사용자에게 잡음이 보이고 (2) 토큰 예산을 사고 과정이 다 먹어서
# 정작 답변이 중간에 잘린다. 실제로 max_tokens=700 으로 물었을 때 답이 끊겼다.
#
# RAG 는 근거 문단이 이미 주어진 추출·요약 작업이라 긴 숙고의 이득이 작고,
# 응답 지연만 커진다. 그래서 끈다. 모델의 chat template 이 이 플래그를 지원한다.
# 복잡한 다중 문서 종합이 필요하면 요청에 "thinking": true 를 넣으면 된다.
THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}

_state: dict[str, Any] = {}
route_logger = logging.getLogger("uvicorn.error")
route_logger.setLevel(logging.INFO)

FOLLOWUP_HINT = re.compile(
    r"^(?:그|당시|그때|이때|이전|그 사람|참여자|담당|결과|일정)"
)


def _is_followup_query(query: str) -> bool:
    return bool(FOLLOWUP_HINT.search(query.strip()))


def _routing_query(history: list[dict[str, str]]) -> str:
    """Keep the current question, adding one prior user topic for follow-ups."""
    user_queries = [m["content"].strip() for m in history if m["role"] == "user"]
    current = user_queries[-1]
    if len(user_queries) < 2 or not _is_followup_query(current):
        return current
    return f"{user_queries[-2]}\n{current}"


def _ensure_slack_stores() -> bool:
    """Reconnect optional Slack stores after a delayed Qdrant startup."""
    if _state.get("slack_store") is None:
        try:
            slack_store = Store(collection=SLACK_COLLECTION)
            if slack_store.exists():
                _state["slack_store"] = slack_store
                route_logger.info("[slack] 요청 시 저장소 연결 복구")
        except Exception as exc:
            route_logger.warning(
                "[slack] 저장소 재연결 실패: %s", type(exc).__name__
            )

    if _state.get("slack_parent_store") is None:
        try:
            parent_store = Store(collection=SLACK_PARENT_COLLECTION)
            if parent_store.exists():
                _state["slack_parent_store"] = parent_store
                route_logger.info("[slack-parent] 요청 시 저장소 연결 복구")
        except Exception as exc:
            route_logger.warning(
                "[slack-parent] 저장소 재연결 실패: %s", type(exc).__name__
            )

    return _state.get("slack_store") is not None


def _is_slack_only_query(query: str) -> bool:
    return set(parse_query_intent(query).sources) == {"slack"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["models"] = Models()
    _state["store"] = Store()
    _state["http"] = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    _state["drive_live"] = None
    _state["ancestry"] = None
    if settings.live_drive_enabled:
        try:
            from .ancestry import AncestryVerifier
            from .drive_live import DriveLiveClient
            oauth = load_rclone_oauth(settings.rclone_config, settings.live_drive_remote)
            _state["drive_live"] = DriveLiveClient(oauth, _state["http"])
            _state["ancestry"] = AncestryVerifier(
                _state["drive_live"], root_id=settings.live_drive_root_id,
                db_path=settings.ancestry_db,
            )
            print("[drive-live] 읽기 전용 실시간 검색 준비됨")
        except Exception as e:
            print(f"[drive-live] 초기화 실패, 문서 인덱스로 계속함: {type(e).__name__}")
    # manifest.sqlite 기반 파일명/경로 검색 — 없어도 dense 검색은 그대로 동작해야
    # 하므로 실패해도 서버 기동을 막지 않고 그냥 꺼둔다(로그만 남김).
    try:
        _state["catalog"] = ManifestCatalog(settings.manifest_db)
        _state["canonical_config"] = load_canonical_config(settings.canonical_records)
        print(f"[catalog] 문서 {len(_state['catalog'])}개, "
              f"canonical intent {len(_state['canonical_config'])}개 로드")
    except Exception as e:
        print(f"[catalog] 로드 실패, dense-only로 계속함: {e}")
        _state["catalog"] = None
        _state["canonical_config"] = None
    # 문서 synopsis 계층(labrag/docsynopsis.py) — 아직 컬렉션이 없으면(빌드 전)
    # 그냥 꺼둔다. scripts/build_docsynopsis.py 로 채운 뒤 재시작하면 켜진다.
    try:
        docsyn_store = Store(collection=DOC_COLLECTION)
        if docsyn_store.exists():
            _state["docsyn_store"] = docsyn_store
            print(f"[docsyn] 문서 synopsis {docsyn_store.count()}개 로드")
        else:
            print(f"[docsyn] 컬렉션 '{DOC_COLLECTION}' 없음 — dense-only로 계속함")
            _state["docsyn_store"] = None
    except Exception as e:
        print(f"[docsyn] 로드 실패, dense-only로 계속함: {e}")
        _state["docsyn_store"] = None
    try:
        slack_store = Store(collection=SLACK_COLLECTION)
        if slack_store.exists():
            _state["slack_store"] = slack_store
            print(f"[slack] Slack 컬렉션 {slack_store.count()}개 로드")
        else:
            print(f"[slack] 컬렉션 '{SLACK_COLLECTION}' 없음 — Slack 검색 끔")
            _state["slack_store"] = None
    except Exception as e:
        print(f"[slack] 로드 실패, Slack 검색 끔: {e}")
        _state["slack_store"] = None
    try:
        parent_store = Store(collection=SLACK_PARENT_COLLECTION)
        if parent_store.exists():
            _state["slack_parent_store"] = parent_store
            print(f"[slack-parent] {parent_store.count()}개 스레드 parent 로드")
        else:
            _state["slack_parent_store"] = None
    except Exception as e:
        print(f"[slack-parent] 로드 실패: {e}")
        _state["slack_parent_store"] = None
    try:
        yield
    finally:
        _state["models"].close()
        await _state["http"].aclose()


app = FastAPI(title="LabRAG", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None

    def text(self) -> str:
        """멀티모달 형식(content 가 배열)으로 와도 텍스트만 뽑는다."""
        if isinstance(self.content, list):
            return "\n".join(
                p.get("text", "") for p in self.content if p.get("type") == "text"
            )
        return self.content or ""


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # 우리가 추가한 옵션 (Open WebUI 는 안 보내지만 API 로 직접 쓸 때 유용)
    top_k: int | None = None
    candidates: int | None = None
    roots: list[str] | None = None
    thinking: bool = False       # 기본 끔 — 아래 THINKING 설명 참고
    mode: Literal[
        "auto", "general", "rag", "location", "web", "rag_web"
    ] = "auto"


def _requires_store(mode: str) -> bool:
    return mode in {"rag", "rag_web", "location"}


async def _safe_search_web(query: str) -> list[WebHit]:
    """웹 검색 장애가 채팅 응답 전체를 실패시키지 않게 한다."""
    try:
        return await search_web(query, _state["http"], settings)
    except Exception as exc:
        route_logger.warning("[web] 검색 실패: %s", type(exc).__name__)
        return []


def _web_source_metadata(
    hits: list[WebHit], *, include_content: bool = False
) -> list[dict]:
    sources = []
    for hit in hits:
        source = {
            "title": hit.title,
            "url": hit.url,
            "score": hit.score,
        }
        if include_content:
            source["content"] = hit.content[:WEB_CONTENT_PREVIEW_CHARS]
        sources.append(source)
    return sources


def _decide_route(
    query: str,
    requested_mode: str,
    models,
    store,
    slack_store,
    roots,
) -> RouteDecision:
    if not settings.auto_route_enabled and requested_mode == "auto":
        return RouteDecision("rag", "auto_route_disabled")
    decision = classify_query(query, requested_mode)
    if not decision.probe_required:
        return decision
    score, error = probe_internal_relevance(
        query, models, store, slack_store=slack_store, roots=roots,
        candidates=settings.route_probe_candidates,
        threshold=settings.route_probe_threshold,
    )
    return resolve_probe(decision, score, settings.route_probe_threshold, error)


def _after_location_attempt(
    decision: RouteDecision,
    requested_mode: str,
    located,
    query: str,
) -> RouteDecision:
    if (
        decision.mode == "location"
        and located is None
        and requested_mode == "auto"
    ):
        # 출처를 웹이라고 말한 경우에만 웹까지 본다. `웹 크롤러`처럼 웹이 주제일
        # 뿐인 질문까지 받으면 위치 검색 실패가 엉뚱한 웹 검색으로 이어진다.
        if WEB_SOURCE_HINT.search(query):
            return RouteDecision("rag_web", "location_no_result_web")
        return RouteDecision("rag", "location_no_result")
    return decision


def _location_fallback(
    live_enabled: bool,
    drive_client,
    ancestry,
    located,
) -> tuple[bool, str]:
    if not live_enabled:
        return False, ""
    status = getattr(getattr(located, "live_status", None), "state", None)
    unavailable = drive_client is None or ancestry is None
    fallback = unavailable or status is None or status != "ok"
    if not fallback:
        return False, ""
    return (
        True,
        "실시간 Google Drive 검색을 사용할 수 없어 로컬 색인 결과를 사용했어.",
    )


def _log_route_event(
    request_id: str,
    decision: RouteDecision,
    *,
    fallback: bool,
    n_candidates: int,
    n_sources: int,
    elapsed_ms: int,
) -> None:
    route_logger.info(json.dumps({
        "event": "route_complete",
        "request_id": request_id,
        "mode": decision.mode,
        "reason": decision.reason,
        "probe_score": decision.probe_score,
        "fallback": fallback,
        "n_candidates": n_candidates,
        "n_sources": n_sources,
        "elapsed_ms": elapsed_ms,
    }, ensure_ascii=False))


@app.get("/health")
async def health() -> dict:
    models: Models = _state["models"]
    store: Store = _state["store"]
    out = {"models": models.health(), "qdrant": store.stats()}
    try:
        r = await _state["http"].get(f"{settings.vllm_url}/models", timeout=5)
        out["gen"] = "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
    except Exception as e:
        out["gen"] = type(e).__name__
    out["drive_live"] = {
        "enabled": settings.live_drive_enabled,
        "auth": "ok" if _state.get("drive_live") is not None else "unavailable",
        "root_id": settings.live_drive_root_id,
    }
    # Tavily 는 호출량 과금·쿼터가 있다. 로그를 grep 하지 않고도 누적 호출 수와
    # 실패 수를 볼 수 있게 노출한다 (errors 가 늘면 쿼터·키를 먼저 본다).
    out["web_search"] = web_search_metrics()
    return out


@app.get("/v1/models")
async def list_models() -> dict:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "lab",
        }],
    }


def _chunk(cid: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    route_started = time.monotonic()
    history = [{"role": m.role, "content": m.text()} for m in req.messages]
    user_msgs = [m for m in history if m["role"] == "user"]
    if not user_msgs:
        raise HTTPException(400, "user 메시지가 없음")
    query = user_msgs[-1]["content"].strip()
    if not query:
        raise HTTPException(400, "질문이 비어있음")
    routing_query = _routing_query(history)

    models: Models = _state["models"]
    store: Store = _state["store"]
    slack_ready = _ensure_slack_stores()
    decision = _decide_route(
        routing_query, req.mode, models, store, _state.get("slack_store"), req.roots
    )
    slack_only = _is_slack_only_query(query)
    slack_web_unavailable = (
        decision.mode == "rag_web"
        and slack_only
        and not slack_ready
    )
    if (
        slack_only
        and not slack_ready
        and req.mode not in {"web", "general"}
        and not slack_web_unavailable
    ):
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        _log_route_event(
            cid, decision, fallback=True, n_candidates=0, n_sources=0,
            elapsed_ms=int((time.monotonic() - route_started) * 1000),
        )
        if not req.stream:
            return {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": SLACK_UNAVAILABLE_MESSAGE,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {},
                "lab_rag": {
                    "query": query,
                    "mode": decision.mode,
                    "route_reason": decision.reason,
                    "status": "unavailable",
                    "n_candidates": 0,
                    "sources": [],
                },
            }

        async def unavailable_gen() -> AsyncIterator[str]:
            yield _chunk(cid, {"role": "assistant", "content": ""})
            yield _chunk(cid, {"content": SLACK_UNAVAILABLE_MESSAGE})
            yield _chunk(cid, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(unavailable_gen(), media_type="text/event-stream")

    # ── 위치 질문은 생성 모델을 아예 거치지 않는다 ──────────────────────
    # "그 파일 어디 있어?" 는 정답이 경로 하나뿐이라, 생성 모델이 문장으로 옮겨
    # 쓰다 틀릴 이유가 없다. 코드가 드라이브 목록의 값을 그대로 표에 넣는다.
    # 위치 질문이 아니면 None 이 와서 아래 기존 경로로 그대로 흐른다.
    located = None
    if decision.mode == "location":
        if not store.exists():
            raise HTTPException(
                503, f"Qdrant 컬렉션 '{store.collection}' 이 없음. "
                     f"먼저 인덱싱: python scripts/index.py"
            )
        located = await maybe_locate_hybrid(
            routing_query, models, _state["catalog"], _state["docsyn_store"], store,
            drive_client=_state.get("drive_live"), ancestry=_state.get("ancestry"),
            canonical_config=_state["canonical_config"], roots=req.roots,
            timeout=settings.live_drive_timeout,
            force=req.mode == "location",
        )
        decision = _after_location_attempt(decision, req.mode, located, routing_query)
    if decision.mode == "location":
        answer = (render_hybrid(located) if located is not None
                  else "해당하는 파일이나 폴더 위치를 찾지 못했어.")
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        fallback, warning = _location_fallback(
            settings.live_drive_enabled, _state.get("drive_live"),
            _state.get("ancestry"), located,
        )
        if warning:
            answer += f"\n\n> {warning}"
        _log_route_event(
            cid, decision, fallback=fallback, n_candidates=0,
            n_sources=len(located.files if located is not None else []),
            elapsed_ms=int((time.monotonic() - route_started) * 1000),
        )
        if not req.stream:
            return {
                "id": cid, "object": "chat.completion", "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": answer}}],
                "usage": {},
                "lab_rag": {
                    "query": query,
                    "intent": getattr(located, "intent", "location"),
                    "mode": "location",
                    "route_reason": decision.reason,
                    "fallback": fallback,
                    "files": [{
                        "name": getattr(f, "name", getattr(getattr(f, "doc", None), "name", "")),
                        "location": getattr(f, "path", getattr(f, "location", "")),
                        "url": getattr(f, "file_url", ""),
                        "evidence": getattr(f, "evidence", []),
                    } for f in (located.files if located is not None else [])],
                },
            }

        async def loc_gen() -> AsyncIterator[str]:
            yield _chunk(cid, {"role": "assistant", "content": ""})
            yield _chunk(cid, {"content": answer})
            yield _chunk(cid, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(loc_gen(), media_type="text/event-stream")

    # 내부 검색은 동기이므로 rag_web에서는 작업 스레드로 보낸다.
    # 그러면 Tavily 대기와 Qdrant·리랭크 대기가 서로를 막지 않는다.
    retrieved = None
    web_hits: list[WebHit] = []
    fallback = False
    internal_query, web_query = (
        split_hybrid_queries(routing_query)
        if decision.mode == "rag_web"
        else (routing_query, routing_query)
    )
    retrieve_args = (internal_query, models, store)
    retrieve_kwargs = {
        "candidates": req.candidates or CANDIDATES,
        "top_k": req.top_k or TOP_K,
        "roots": req.roots,
        "catalog": _state["catalog"],
        "canonical_config": _state["canonical_config"],
        "docsyn_store": _state["docsyn_store"],
        "slack_store": _state["slack_store"],
        "slack_parent_store": _state["slack_parent_store"],
    }
    if decision.mode in {"rag", "rag_web"}:
        if not store.exists():
            raise HTTPException(
                503, f"Qdrant 컬렉션 '{store.collection}' 이 없음. "
                     f"먼저 인덱싱: python scripts/index.py"
            )
    if decision.mode == "rag":
        retrieved = retrieve(*retrieve_args, **retrieve_kwargs)
        messages = build_messages(retrieved, history)
    elif decision.mode == "web":
        web_hits = await _safe_search_web(routing_query)
        fallback = not web_hits
        messages = (
            build_web_messages(web_hits, history)
            if web_hits else build_general_messages(history)
        )
    elif decision.mode == "rag_web":
        retrieved, web_hits = await asyncio.gather(
            asyncio.to_thread(retrieve, *retrieve_args, **retrieve_kwargs),
            _safe_search_web(web_query),
        )
        fallback = not web_hits
        if retrieved.hits or web_hits:
            messages = build_hybrid_messages(retrieved, web_hits, history)
        else:
            messages = build_general_messages(history)
    else:
        messages = build_general_messages(history)

    upstream = {
        "model": settings.vllm_model,
        "messages": messages,
        "temperature": req.temperature if req.temperature is not None else 0.2,
        "stream": req.stream,
        # 인용 목록까지 온전히 나오도록 넉넉히 준다. Open WebUI 는 max_tokens 를 안 보낸다.
        "max_tokens": req.max_tokens or 2048,
    }
    if not req.thinking:
        upstream.update(THINKING_OFF)

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    internal_sources = (
        retrieved.sources_block() if retrieved and retrieved.hits else ""
    )
    sources = internal_sources + web_sources_block(web_hits)
    fallback_notice = ""
    if fallback and decision.mode == "web":
        fallback_notice = (
            "\n\n> 웹 검색 결과를 가져오지 못해 일반 지식으로 "
            "답변했어. 최신 정보는 다시 확인해 줘."
        )
    elif fallback and decision.mode == "rag_web":
        if retrieved and retrieved.hits:
            fallback_notice = (
                "\n\n> 웹 검색 결과를 가져오지 못해 확인 가능한 "
                "연구실 내부 자료를 우선해 답변했어."
            )
        else:
            fallback_notice = (
                "\n\n> 연구실 내부 자료와 웹 검색 모두에서 "
                "답변할 근거를 찾지 못해 일반 지식으로 답변했어."
            )
    slack_notice = ""
    if slack_web_unavailable:
        slack_notice = (
            f"\n\n> {SLACK_UNAVAILABLE_MESSAGE} "
            "웹 검색은 계속 진행했어."
        )
    # 어느 채널로 이해했는지 등, 검색이 조용히 좁히거나 포기한 것을 사용자에게 알린다.
    notes = list(retrieved.notes) if retrieved else []
    if decision.reason == "probe_low":
        # 내부 관련도가 문턱을 못 넘어 검색을 건너뛴 경우. 말없이 일반 답변을 하면
        # 사용자는 연구실에 자료가 없다고 오해한다 — 실제로 인덱스에 정의가 있는
        # "오로라 배치체계"가 틀린 일반 지식으로 답해졌다. 문턱을 아무리 잘
        # 맞춰도 업무 질문과 잡담의 점수 분포가 겹치므로, 놓침은 알려서 복구한다.
        notes.append(
            "연구실 자료에서 관련 근거를 찾지 못해 일반 지식으로 답했어. "
            "내부 자료를 찾으려면 '연구실'·'과제' 같은 말을 붙여 다시 물어봐 줘."
        )
    retrieval_notice = "".join(f"\n\n> {note}" for note in notes)
    _log_route_event(
        cid, decision, fallback=fallback or slack_web_unavailable,
        n_candidates=retrieved.n_candidates if retrieved else 0,
        n_sources=(len(retrieved.hits) if retrieved else 0) + len(web_hits),
        elapsed_ms=int((time.monotonic() - route_started) * 1000),
    )

    if not req.stream:
        r = await _state["http"].post(
            f"{settings.vllm_url}/chat/completions", json=upstream
        )
        if r.status_code != 200:
            raise HTTPException(502, f"생성 모델 오류 {r.status_code}: {r.text[:300]}")
        data = r.json()
        answer = data["choices"][0]["message"]["content"]
        return {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (answer + retrieval_notice + fallback_notice
                                + slack_notice + sources),
                },
                "finish_reason": data["choices"][0].get("finish_reason", "stop"),
            }],
            "usage": data.get("usage", {}),
            # 디버깅·평가용 — 표준 필드가 아니라 Open WebUI 는 무시한다
            "lab_rag": {
                "query": query,
                "mode": decision.mode,
                "route_reason": decision.reason,
                "probe_score": decision.probe_score,
                "fallback": fallback or slack_web_unavailable,
                "n_candidates": retrieved.n_candidates if retrieved else 0,
                "sources": [
                    {"citation": h.citation, "url": h.url, "rerank_score": s}
                    for h, s in zip(
                        retrieved.hits if retrieved else [],
                        retrieved.scores if retrieved else [],
                    )
                ],
                "web_sources": _web_source_metadata(web_hits),
                **(
                    {"slack_status": "unavailable"}
                    if slack_web_unavailable else {}
                ),
            },
        }

    async def gen() -> AsyncIterator[str]:
        yield _chunk(cid, {"role": "assistant", "content": ""})
        try:
            async with _state["http"].stream(
                "POST", f"{settings.vllm_url}/chat/completions", json=upstream
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()[:300]
                    yield _chunk(cid, {"content": f"\n\n[생성 모델 오류 "
                                                  f"{resp.status_code}] {body}"})
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            d = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        delta = d.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            yield _chunk(cid, {"content": delta["content"]})
        except Exception as e:
            yield _chunk(cid, {"content": f"\n\n[오류] {type(e).__name__}: {e}"})

        if retrieval_notice:
            yield _chunk(cid, {"content": retrieval_notice})
        if fallback_notice:
            yield _chunk(cid, {"content": fallback_notice})
        if slack_notice:
            yield _chunk(cid, {"content": slack_notice})
        if sources:
            yield _chunk(cid, {"content": "\n" + sources})
        yield _chunk(cid, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/search")
async def search_only(req: ChatRequest) -> dict:
    """생성 없이 검색 결과만. eval 셋 만들 때와 검색 품질 점검에 쓴다."""
    user_msgs = [m for m in req.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(400, "user 메시지가 없음")
    query = user_msgs[-1].text()
    history = [{"role": m.role, "content": m.text()} for m in req.messages]
    routing_query = _routing_query(history)
    slack_ready = _ensure_slack_stores()
    decision = _decide_route(
        routing_query, req.mode, _state["models"], _state["store"],
        _state.get("slack_store"), req.roots,
    )
    slack_only = _is_slack_only_query(query)
    slack_web_unavailable = (
        decision.mode == "rag_web"
        and slack_only
        and not slack_ready
    )
    if (
        slack_only
        and not slack_ready
        and req.mode not in {"web", "general"}
        and not slack_web_unavailable
    ):
        return {
            "query": query,
            "mode": decision.mode,
            "route_reason": decision.reason,
            "status": "unavailable",
            "error": SLACK_UNAVAILABLE_MESSAGE,
            "results": [],
        }
    if decision.mode == "general":
        return {
            "query": query,
            "mode": "general",
            "route_reason": decision.reason,
            "probe_score": decision.probe_score,
            "results": [],
        }
    if decision.mode == "web":
        web_hits = await _safe_search_web(routing_query)
        return {
            "query": query,
            "mode": "web",
            "route_reason": decision.reason,
            "fallback": not web_hits,
            "results": [],
            "web_results": _web_source_metadata(
                web_hits, include_content=True
            ),
        }
    if not _state["store"].exists():
        raise HTTPException(
            503, f"Qdrant 컬렉션 '{_state['store'].collection}' 이 없음"
        )
    located = None
    if decision.mode == "location":
        located = await maybe_locate_hybrid(
            routing_query, _state["models"], _state["catalog"], _state["docsyn_store"],
            _state["store"], drive_client=_state.get("drive_live"),
            ancestry=_state.get("ancestry"),
            canonical_config=_state["canonical_config"], roots=req.roots,
            timeout=settings.live_drive_timeout,
            force=req.mode == "location",
        )
        decision = _after_location_attempt(decision, req.mode, located, routing_query)
    if decision.mode == "location":
        return {
            "query": query, "mode": "location",
            "route_reason": decision.reason,
            "intent": getattr(located, "intent", "location"),
            "results": [
                {"file_name": getattr(f, "name", getattr(getattr(f, "doc", None), "name", "")),
                 "location": getattr(f, "path", getattr(f, "location", "")),
                 "url": getattr(f, "file_url", ""),
                 "evidence": getattr(f, "evidence", [])}
                for f in (located.files if located is not None else [])
            ],
        }
    internal_query, web_query = (
        split_hybrid_queries(routing_query)
        if decision.mode == "rag_web"
        else (routing_query, routing_query)
    )
    retrieve_args = (internal_query, _state["models"], _state["store"])
    retrieve_kwargs = {
        "candidates": req.candidates or CANDIDATES,
        "top_k": req.top_k or TOP_K,
        "roots": req.roots,
        "catalog": _state["catalog"],
        "canonical_config": _state["canonical_config"],
        "docsyn_store": _state["docsyn_store"],
        "slack_store": _state["slack_store"],
        "slack_parent_store": _state["slack_parent_store"],
    }
    web_hits: list[WebHit] = []
    if decision.mode == "rag_web":
        retrieved, web_hits = await asyncio.gather(
            asyncio.to_thread(retrieve, *retrieve_args, **retrieve_kwargs),
            _safe_search_web(web_query),
        )
    else:
        retrieved = retrieve(*retrieve_args, **retrieve_kwargs)
    response = {
        "query": query,
        "mode": decision.mode,
        "route_reason": decision.reason,
        "probe_score": decision.probe_score,
        "n_candidates": retrieved.n_candidates,
        "results": [
            {"rerank_score": s, "citation": h.citation, "url": h.url,
             "root": h.payload.get("root"), "text": h.text}
            for h, s in zip(retrieved.hits, retrieved.scores)
        ],
    }
    if getattr(retrieved, "notes", None):
        response["notes"] = list(retrieved.notes)
    if decision.mode == "rag_web":
        response["fallback"] = not web_hits or slack_web_unavailable
        response["web_results"] = _web_source_metadata(
            web_hits, include_content=True
        )
        if slack_web_unavailable:
            response["slack_status"] = "unavailable"
            response["warnings"] = [SLACK_UNAVAILABLE_MESSAGE]
    return response
