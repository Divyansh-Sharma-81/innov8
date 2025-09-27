from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from sqlalchemy.exc import NoResultFound

from backend.services.transcriber.schema import TranscriptChunk, WindowPayload
from backend.services.transcriber.service import TranscriberRunner, TranscriberService
from backend.services.mini_llm.schema import MiniInput, MiniOutput
from backend.services.mini_llm.service import MiniLLMService
from backend.services.problemset.schema import MonacoPayload, ProblemDTO
from backend.services.problemset.seed import seed_from_csv
from backend.services.problemset.service import ProblemsetService
from backend.services.judge.schema import ExecResult
from backend.services.judge.service import JudgeService


load_dotenv()
logger = logging.getLogger("backend.api")


class Settings(BaseSettings):
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    MINI_MODEL_ID: str = "llama-3.3-70b-versatile"
    MINI_USE_MOCK: bool = True
    WINDOW_SECONDS: int = 10
    PRETEXT_SECONDS: int = 60
    WS_ENABLED: bool = True
    ASSEMBLYAI_API_KEY: str | None = None
    ASR_SAMPLE_RATE: int = 16000
    ASR_CHANNELS: int = 1
    ASR_FORMAT: str = "pcm_s16le"
    ASR_RECONNECT_SECS: float = 2.0
    DATABASE_URL: str
    JUDGE0_HOST: str | None = None
    JUDGE0_KEY: str | None = None
    JUDGE0_LANG_PY: int = 71
    USE_JUDGE0_MOCK: bool = True
    ENABLE_DEV_ENDPOINTS: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
app = FastAPI(title="Innov8 Mini Orchestrator")
transcriber_service = TranscriberService()
problem_service = ProblemsetService(database_url=settings.DATABASE_URL)
mini_llm_service = MiniLLMService(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_BASE_URL,
    model_id=settings.MINI_MODEL_ID,
    use_mock=settings.MINI_USE_MOCK,
)
judge_service = JudgeService(
    host=settings.JUDGE0_HOST,
    api_key=settings.JUDGE0_KEY,
    language_id=settings.JUDGE0_LANG_PY,
    use_mock=settings.USE_JUDGE0_MOCK,
)
transcriber_runner = TranscriberRunner(
    transcriber=transcriber_service,
    api_key=settings.ASSEMBLYAI_API_KEY,
    window_seconds=settings.WINDOW_SECONDS,
    pretext_seconds=settings.PRETEXT_SECONDS,
    sample_rate=settings.ASR_SAMPLE_RATE,
    channels=settings.ASR_CHANNELS,
    audio_format=settings.ASR_FORMAT,
    reconnect_secs=settings.ASR_RECONNECT_SECS,
)

SESSIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class SessionControlRequest(BaseModel):
    session_id: str
    device_label: str | None = None
    device_index: int | None = None


class RunRequest(BaseModel):
    problem_id: str
    source: str
    language: str = "python"


class WebSocketHub:
    def __init__(self) -> None:
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[session_id].add(ws)

    async def unregister(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._connections[session_id].discard(ws)
            if not self._connections[session_id]:
                self._connections.pop(session_id, None)

    async def broadcast(self, session_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            receivers = list(self._connections.get(session_id, set()))
        if not receivers:
            return
        message = json.dumps(payload)
        await asyncio.gather(
            *(ws.send_text(message) for ws in receivers),
            return_exceptions=True,
        )


ws_hub = WebSocketHub()


def get_settings() -> Settings:
    return settings


def append_session_event(session_id: str, event: Dict[str, Any]) -> None:
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    line = json.dumps(event, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def log_asr_chunk(chunk: TranscriptChunk) -> None:
    append_session_event(
        chunk.session_id,
        {
            "ts": time.time(),
            "type": "asr.finalized",
            "text": chunk.text,
            "start_ts": chunk.start_ts,
            "end_ts": chunk.end_ts,
        },
    )


async def broadcast_problem_presented(session_id: str, payload: MonacoPayload) -> None:
    event = {
        "type": "problem.presented",
        "session_id": session_id,
        "payload": payload.dict(),
    }
    append_session_event(
        session_id,
        {
            "ts": time.time(),
            "type": "problem.presented",
            "payload": payload.dict(),
        },
    )
    if settings.WS_ENABLED:
        await ws_hub.broadcast(session_id, event)


async def process_window(window: WindowPayload) -> MiniOutput:
    try:
        mini_out = await mini_llm_service.call(window)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Mini-LLM call failed: %s", exc)
        mini_out = MiniOutput(
            tags=[],
            scores={},
            speak_gate="ok",
            reason="LLM error",
            timestamp=time.time(),
        )
    event_payload = {
        "ts": mini_out.timestamp,
        "type": "mini.tag",
        "window_s": window.window_s,
        "pretext_s": window.pretext_s,
        "payload": mini_out.dict(),
    }
    append_session_event(window.session_id, event_payload)
    if settings.WS_ENABLED:
        await ws_hub.broadcast(
            window.session_id,
            {"type": "mini.tag", "session_id": window.session_id, "payload": mini_out.dict()},
        )
    return mini_out


@app.on_event("startup")
async def _startup() -> None:
    await mini_llm_service.start()
    await judge_service.start()
    transcriber_service.set_window_consumer(process_window)
    transcriber_runner.set_chunk_logger(log_asr_chunk)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await transcriber_runner.stop_all()
    await mini_llm_service.close()
    await judge_service.close()


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/asr/start")
async def asr_start(request: SessionControlRequest) -> Dict[str, bool]:
    try:
        await transcriber_runner.start_stream(
            request.session_id,
            request.device_label,
            request.device_index,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/asr/stop")
async def asr_stop(request: SessionControlRequest) -> Dict[str, bool]:
    try:
        await transcriber_runner.stop_stream(request.session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/asr/ingest", response_model=MiniOutput)
async def ingest(chunk: TranscriptChunk, cfg: Settings = Depends(get_settings)) -> MiniOutput:
    if chunk.is_final:
        log_asr_chunk(chunk)
    result = await transcriber_service.handle_chunk(
        chunk,
        cfg.WINDOW_SECONDS,
        cfg.PRETEXT_SECONDS,
        debounce=False,
    )
    if not result:
        raise HTTPException(status_code=400, detail="insufficient transcript context")
    window, processed = result
    if processed is None:
        processed = await process_window(window)
    return processed


@app.post("/mini/tick", response_model=MiniOutput)
async def tick(body: MiniInput) -> MiniOutput:
    return await mini_llm_service.call(body)


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(session_id: str, ws: WebSocket, cfg: Settings = Depends(get_settings)) -> None:
    if not cfg.WS_ENABLED:
        await ws.close(code=1008)
        return
    await ws_hub.register(session_id, ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await ws_hub.unregister(session_id, ws)


# ------------------------ Problemset Endpoints -------------------------


@app.get("/problems/ratings", response_model=List[int])
async def list_ratings() -> List[int]:
    return problem_service.list_ratings()


@app.get("/problems/search", response_model=List[ProblemDTO])
async def search_problems(
    rating: Optional[int] = Query(default=None),
    tags: Optional[str] = Query(default=None, description="Comma separated tags"),
    limit: int = Query(default=1, ge=1, le=10),
) -> List[ProblemDTO]:
    tag_list = [tag.strip() for tag in (tags.split(",") if tags else []) if tag.strip()]
    return problem_service.search(rating=rating, tags=tag_list, limit=limit)


@app.get("/problems/{problem_id}", response_model=ProblemDTO)
async def get_problem(problem_id: str) -> ProblemDTO:
    try:
        return problem_service.get_problem(problem_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/problems/seed")
async def seed_problems(force: bool = Body(default=False)) -> Dict[str, bool]:
    if not settings.ENABLE_DEV_ENDPOINTS and not force:
        raise HTTPException(status_code=403, detail="Seeding endpoint disabled")
    seed_from_csv(problem_service)
    return {"ok": True}


@app.get("/editor/payload", response_model=MonacoPayload)
async def get_monaco_payload(
    problem_id: Optional[str] = Query(default=None),
    rating: Optional[int] = Query(default=None),
    tags: Optional[str] = Query(default=None, description="Comma separated tags"),
    limit: int = Query(default=1, ge=1, le=10),
    session_id: Optional[str] = Query(default=None),
    broadcast: bool = Query(default=False),
) -> MonacoPayload:
    tag_list = [tag.strip() for tag in (tags.split(",") if tags else []) if tag.strip()]
    try:
        payload = problem_service.monaco_payload(
            problem_id=problem_id,
            rating=rating,
            tags=tag_list,
            limit=limit,
        )
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if broadcast and session_id:
        await broadcast_problem_presented(session_id, payload)
    return payload


@app.post("/run", response_model=ExecResult)
async def run_submission(request: RunRequest) -> ExecResult:
    if request.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only Python is supported currently")
    try:
        samples = problem_service.get_samples(request.problem_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = await judge_service.run_python_cases(request.source, samples)
    return result
