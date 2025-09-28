from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
from backend.services.orchestrator.schema import Decision, TriggerEvent
from backend.services.orchestrator.service import Orchestrator
from backend.services.snapshots.schema import (
    SnapshotIn,
    SnapshotOut,
    TimelinePage,
    TimelineSummary,
)
from backend.services.snapshots.service import SnapshotService
from backend.services.main_agent import AgentIn as AgentInPayload
from backend.services.main_agent import AgentOut as AgentOutPayload
from backend.services.main_agent import FinalEval, MainAgentService
from backend.services.templates import TemplatesService
from backend.services.templates.schema import InterviewPlan


load_dotenv()
logger = logging.getLogger("backend.api")


class Settings(BaseSettings):
    GROQ_API_KEY: str | None = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    MINI_MODEL_ID: str = "llama-3.3-70b-versatile"
    MINI_USE_MOCK: bool = True
    AGENT_MODEL_ID: str = "gpt-oss-120b"
    USE_AGENT_MOCK: bool = True
    AGENT_BASE_URL: str = "https://api.cerebras.ai/v1"
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
    ANALYZER_ENABLED: bool = True
    HINT_WINDOW_SECONDS: int = 25
    SNAPSHOTS_STORE_FULL: bool = False
    ENABLE_DEV_ENDPOINTS: bool = False
    CEREBRAS_API_KEY: str | None = None

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
templates_service = TemplatesService(database_url=settings.DATABASE_URL)
main_agent_service = MainAgentService(
    api_key=settings.CEREBRAS_API_KEY or settings.GROQ_API_KEY,
    base_url=settings.AGENT_BASE_URL,
    model_id=settings.AGENT_MODEL_ID,
    use_mock=settings.USE_AGENT_MOCK,
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
TIMELINE_KINDS = {
    "any",
    "diff.snapshot",
    "exec.result",
    "problem.presented",
    "mini.tag",
    "agent.message",
    "agent.hint",
    "agent.hr",
    "agent.aptitude",
    "agent.stay_silent",
    "agent.action",
    "running.summary.updated",
    "interview.final",
}


class SessionControlRequest(BaseModel):
    session_id: str
    device_label: str | None = None
    device_index: int | None = None


class RunRequest(BaseModel):
    session_id: str
    problem_id: str
    source: str
    language: str = "python"
    cursor_line: Optional[int] = None


class PresentProblemRequest(BaseModel):
    session_id: str
    rating: Optional[int] = None
    tags: Optional[List[str]] = None
    problem_id: Optional[str] = None
    broadcast: bool = True


class IdleTickRequest(BaseModel):
    session_id: str


class AgentPlanRequest(BaseModel):
    session_id: str


class AgentFinalizeRequest(BaseModel):
    session_id: str


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


RUNNING_SUMMARY: Dict[str, str] = {}
BEST_PROBLEM_TIMES: Dict[str, float] = {}


def append_running_summary(session_id: str, update: str) -> str:
    snippet = (update or "").strip()
    if not snippet:
        return RUNNING_SUMMARY.get(session_id, "")
    existing = RUNNING_SUMMARY.get(session_id)
    combined = snippet if not existing else f"{existing}\n{snippet}".strip()
    RUNNING_SUMMARY[session_id] = combined
    append_session_event(
        session_id,
        {
            "ts": time.time(),
            "type": "running.summary.updated",
            "update": snippet,
            "summary": combined,
        },
    )
    return combined


def get_running_summary(session_id: str) -> str:
    return RUNNING_SUMMARY.get(session_id, "")


_COMPLEXITY_SCALE = ["O(n)", "O(n log n)", "O(n^2)"]
_COMPLEXITY_INDEX = {value: idx for idx, value in enumerate(_COMPLEXITY_SCALE)}


def infer_expected_complexity(rating: int, tags: List[str]) -> str:
    lowered = {tag.lower() for tag in tags}
    if any("sort" in tag for tag in lowered):
        return "O(n log n)"
    if any("binary" in tag and "search" in tag for tag in lowered):
        return "O(n log n)"
    if any("two pointer" in tag or "two-pointers" in tag for tag in lowered):
        return "O(n)"
    if any(tag in lowered for tag in {"hash", "hashing", "map", "set", "dictionary"}):
        return "O(n)"
    if rating <= 900:
        return "O(n^2)"
    if rating >= 1300:
        return "O(n log n)"
    return "O(n)"


def closeness_from_complexity(expected: str, observed: str) -> float:
    expected_idx = _COMPLEXITY_INDEX.get(expected)
    observed_idx = _COMPLEXITY_INDEX.get(observed)
    if expected_idx is None:
        return 0.5
    if observed_idx is None:
        return 0.5
    diff = observed_idx - expected_idx
    if diff <= 0:
        return 1.0
    if diff == 1:
        return 0.6
    return 0.3


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


async def broadcast_snapshot_event(session_id: str, payload: Dict[str, Any]) -> None:
    if settings.WS_ENABLED:
        await ws_hub.broadcast(session_id, payload)


snapshot_service = SnapshotService(
    logger_append_fn=append_session_event,
    ws_broadcast_fn=broadcast_snapshot_event,
    analyzer_enabled=settings.ANALYZER_ENABLED,
    hint_window_seconds=settings.HINT_WINDOW_SECONDS,
    store_full_source=settings.SNAPSHOTS_STORE_FULL,
)

orchestrator = Orchestrator(
    problem_service=problem_service,
    broadcast_problem=broadcast_problem_presented,
    logger=append_session_event,
)


async def emit_agent_event(
    event_type: str,
    session_id: str,
    *,
    text: str,
    level: str,
    reason: str,
) -> None:
    event = {
        "ts": time.time(),
        "type": event_type,
        "text": text,
        "level": level,
        "reason": reason,
    }
    append_session_event(session_id, event)
    if settings.WS_ENABLED:
        await ws_hub.broadcast(
            session_id,
            {
                "type": event_type,
                "session_id": session_id,
                "text": text,
                "level": level,
                "reason": reason,
            },
        )


def log_agent_silence(session_id: str, reason: str) -> None:
    append_session_event(
        session_id,
        {
            "ts": time.time(),
            "type": "agent.stay_silent",
            "reason": reason,
        },
    )


async def emit_agent_question(
    event_type: str,
    session_id: str,
    *,
    text: str,
    category: str,
    reason: str,
) -> None:
    event = {
        "ts": time.time(),
        "type": event_type,
        "text": text,
        "category": category,
        "reason": reason,
    }
    append_session_event(session_id, event)
    if settings.WS_ENABLED:
        await ws_hub.broadcast(
            session_id,
            {
                "type": event_type,
                "session_id": session_id,
                "text": text,
                "category": category,
                "reason": reason,
            },
        )


class AgentActionRegistry:
    """Executes AgentOut actions, respecting orchestration rules."""

    def __init__(self, orchestrator_service: Orchestrator, templates: TemplatesService) -> None:
        self._orchestrator = orchestrator_service
        self._templates = templates

    async def execute(self, payload: AgentInPayload, decision: AgentOutPayload) -> AgentOutPayload:
        session_id = payload.session_id
        now = time.time()
        requested_action = decision.action
        final_action = requested_action

        if payload.speak_gate == "hold" and requested_action not in {"stay_silent", "present_problem"}:
            final_action = "stay_silent"

        if not self._orchestrator.can_execute_agent_action(session_id, final_action, now):
            final_action = "stay_silent"

        executed = decision.copy(update={"action": final_action})
        append_session_event(
            session_id,
            {
                "ts": now,
                "type": "agent.action",
                "requested": decision.dict(),
                "executed": executed.dict(),
            },
        )

        if final_action == "present_problem":
            payload_tags = executed.desired_tags or None
            next_rating = executed.next_rating
            await self._orchestrator.present_problem(
                session_id,
                rating=next_rating,
                tags=payload_tags,
                broadcast=True,
            )
            self._templates.increment(session_id, "coding")
            self._orchestrator.register_agent_action(session_id, "present_problem", now)
            return executed

        if final_action == "send_message":
            text = executed.text or "Let me know how it's going."
            await emit_agent_event(
                "agent.message",
                session_id,
                text=text,
                level="plain",
                reason="main_agent",
            )
            self._orchestrator.register_agent_action(session_id, "send_message", now)
            return executed.copy(update={"text": text})

        if final_action == "give_hint":
            level = executed.hint_level or "guide"
            text = executed.text or "Try restating the problem in your own words and outline the data structures you need."
            await emit_agent_event(
                "agent.hint",
                session_id,
                text=text,
                level=level,
                reason="main_agent",
            )
            snapshot_service.record_hint(session_id, text, level, now)
            self._orchestrator.register_agent_action(session_id, "give_hint", now)
            return executed.copy(update={"text": text, "hint_level": level})

        if final_action == "ask_hr":
            question = self._templates.sample_hr()
            await emit_agent_question(
                "agent.hr",
                session_id,
                text=question.text,
                category=question.category or "general",
                reason="main_agent",
            )
            self._templates.increment(session_id, "hr")
            self._orchestrator.register_agent_action(session_id, "ask_hr", now)
            return executed.copy(update={"text": question.text})

        if final_action == "ask_aptitude":
            question = self._templates.sample_aptitude()
            await emit_agent_question(
                "agent.aptitude",
                session_id,
                text=question.text,
                category=question.category or "general",
                reason="main_agent",
            )
            self._templates.increment(session_id, "aptitude")
            self._orchestrator.register_agent_action(session_id, "ask_aptitude", now)
            return executed.copy(update={"text": question.text})

        log_agent_silence(session_id, "main_agent")
        self._orchestrator.register_agent_action(session_id, "stay_silent", now)
        return executed.copy(update={"text": "", "hint_level": None})


action_registry = AgentActionRegistry(orchestrator, templates_service)


def read_session_events(session_id: str) -> List[Dict[str, Any]]:
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


async def execute_decision(session_id: str, decision: Decision) -> List[Dict[str, Any]]:
    executed: List[Dict[str, Any]] = []
    for action in decision.actions:
        if action.type == "present_problem":
            payload = await orchestrator.present_problem(
                session_id,
                rating=action.data.get("rating"),
                tags=action.data.get("tags"),
                problem_id=action.data.get("problem_id"),
                broadcast=action.data.get("broadcast", True),
            )
            templates_service.increment(session_id, "coding")
            executed.append({
                "action": action.dict(),
                "payload": payload.dict(),
            })
        elif action.type == "send_message":
            text = action.data.get("text", "All good.")
            level = action.data.get("level", "plain")
            await emit_agent_event(
                "agent.message",
                session_id,
                text=text,
                level=level,
                reason=action.reason,
            )
            executed.append({"action": action.dict()})
        elif action.type == "give_hint":
            text = action.data.get("text", "Consider outlining your solution before coding.")
            level = action.data.get("level", "guide")
            await emit_agent_event(
                "agent.hint",
                session_id,
                text=text,
                level=level,
                reason=action.reason,
            )
            snapshot_service.record_hint(session_id, text, level, time.time())
            executed.append({"action": action.dict()})
        elif action.type == "ask_hr":
            question = templates_service.sample_hr()
            await emit_agent_question(
                "agent.hr",
                session_id,
                text=question.text,
                category=question.category or "general",
                reason=action.reason,
            )
            templates_service.increment(session_id, "hr")
            executed.append({"action": action.dict(), "question": question.dict()})
        elif action.type == "ask_aptitude":
            question = templates_service.sample_aptitude()
            await emit_agent_question(
                "agent.aptitude",
                session_id,
                text=question.text,
                category=question.category or "general",
                reason=action.reason,
            )
            templates_service.increment(session_id, "aptitude")
            executed.append({"action": action.dict(), "question": question.dict()})
        elif action.type == "stay_silent":
            log_agent_silence(session_id, action.reason)
            executed.append({"action": action.dict()})
    return executed


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
    decision = orchestrator.handle_event(
        TriggerEvent(
            session_id=window.session_id,
            kind="mini.tag",
            payload=mini_out.dict(),
            ts=mini_out.timestamp or time.time(),
        )
    )
    await execute_decision(window.session_id, decision)
    return mini_out


@app.on_event("startup")
async def _startup() -> None:
    await mini_llm_service.start()
    await judge_service.start()
    await main_agent_service.start()
    transcriber_service.set_window_consumer(process_window)
    transcriber_runner.set_chunk_logger(log_asr_chunk)


@app.on_event("shutdown")
async def _shutdown() -> None:
    await transcriber_runner.stop_all()
    await mini_llm_service.close()
    await judge_service.close()
    await main_agent_service.close()


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


@app.post("/editor/save", response_model=SnapshotOut)
async def editor_save(snapshot: SnapshotIn) -> SnapshotOut:
    if snapshot.language != "python":
        raise HTTPException(status_code=400, detail="Only Python snapshots are supported")
    out = await snapshot_service.snapshot(snapshot)
    decision = orchestrator.handle_event(
        TriggerEvent(
            session_id=snapshot.session_id,
            kind="ui.editor.save",
            payload={
                "version": out.version,
                "lines_changed": out.diff_summary.lines_changed,
            },
            ts=out.diff_summary.timestamp,
        )
    )
    await execute_decision(snapshot.session_id, decision)
    return out


@app.post("/run", response_model=ExecResult)
async def run_submission(request: RunRequest) -> ExecResult:
    if request.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only Python is supported currently")
    _snapshot = await snapshot_service.snapshot(
        SnapshotIn(
            session_id=request.session_id,
            problem_id=request.problem_id,
            language="python",
            source=request.source,
            cursor_line=request.cursor_line,
            reason="run",
        )
    )
    try:
        problem_dto = problem_service.get_problem(request.problem_id)
        samples = problem_service.get_samples(request.problem_id)
    except NoResultFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = await judge_service.run_python_cases(request.source, samples)
    complexity_snapshot = snapshot_service.analyze_source(request.source)
    closeness_correct = (result.passed / result.total) if result.total else 0.0
    expected_complexity = infer_expected_complexity(problem_dto.rating, problem_dto.tags)
    observed_complexity = complexity_snapshot.get("time", "unknown")
    closeness_optimal = closeness_from_complexity(expected_complexity, observed_complexity)
    case_times = [case.time_ms for case in result.cases if case.time_ms is not None]
    if case_times:
        observed_time = min(case_times)
        best_time = BEST_PROBLEM_TIMES.get(problem_dto.id)
        if best_time is None or observed_time < best_time:
            BEST_PROBLEM_TIMES[problem_dto.id] = observed_time
            best_time = observed_time
        if best_time:
            if observed_time <= best_time * 2:
                closeness_optimal = min(1.0, closeness_optimal + 0.1)
            else:
                closeness_optimal = max(0.0, closeness_optimal - 0.1)
    result.closeness_to_correct = round(closeness_correct, 3)
    result.closeness_to_optimal = round(closeness_optimal, 3)
    snapshot_service.set_pending_closeness(
        request.session_id,
        request.problem_id,
        "python",
        {
            "closeness_to_correct": float(result.closeness_to_correct or 0.0),
            "closeness_to_optimal": float(result.closeness_to_optimal or 0.0),
        },
    )
    result_dict = result.dict()
    result_ts = time.time()
    append_session_event(
        request.session_id,
        {
            "ts": result_ts,
            "type": "exec.result",
            "problem_id": request.problem_id,
            "result": result_dict,
        },
    )
    trigger_payload = dict(result_dict)
    trigger_payload["problem_id"] = request.problem_id
    decision = orchestrator.handle_event(
        TriggerEvent(
            session_id=request.session_id,
            kind="exec.result",
            payload=trigger_payload,
            ts=result_ts,
        )
    )
    await execute_decision(request.session_id, decision)
    return result


@app.post("/agent/decide", response_model=AgentOutPayload)
async def agent_decide(body: AgentInPayload) -> AgentOutPayload:
    plan = templates_service.get_or_create_plan(body.session_id)
    summary_text = get_running_summary(body.session_id)
    enriched = body.copy(update={"plan_state": plan.dict(), "last_agent_summary": summary_text})
    try:
        agent_choice = await main_agent_service.decide(enriched)
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.exception("Main agent call failed: %s", exc)
        agent_choice = AgentOutPayload(action="stay_silent", text="")
    executed = await action_registry.execute(enriched, agent_choice)
    if executed.update_running_summary:
        new_summary = append_running_summary(body.session_id, executed.update_running_summary)
        executed = executed.copy(update={"update_running_summary": new_summary})
    return executed


@app.get("/agent/plan", response_model=InterviewPlan)
async def agent_plan(session_id: str = Query(...)) -> InterviewPlan:
    return templates_service.get_or_create_plan(session_id)


@app.post("/agent/plan/reset", response_model=InterviewPlan)
async def agent_plan_reset(body: AgentPlanRequest) -> InterviewPlan:
    plan = templates_service.reset_plan(body.session_id)
    RUNNING_SUMMARY.pop(body.session_id, None)
    append_session_event(
        body.session_id,
        {
            "ts": time.time(),
            "type": "running.summary.updated",
            "update": "reset",
            "summary": "",
        },
    )
    return plan


@app.post("/agent/finalize", response_model=FinalEval)
async def agent_finalize(body: AgentFinalizeRequest) -> FinalEval:
    events = read_session_events(body.session_id)
    closeness_correct: List[float] = []
    closeness_optimal: List[float] = []
    hints_given = 0
    hints_followed = 0

    for event in events:
        event_type = event.get("type")
        if event_type == "exec.result":
            result = event.get("result", {})
            cc = result.get("closeness_to_correct")
            co = result.get("closeness_to_optimal")
            if isinstance(cc, (int, float)):
                closeness_correct.append(float(cc))
            if isinstance(co, (int, float)):
                closeness_optimal.append(float(co))
        elif event_type == "agent.hint":
            hints_given += 1
        elif event_type == "diff.snapshot":
            summary = event.get("summary", {})
            if summary.get("hint_followed"):
                hints_followed += 1

    avg_correct = sum(closeness_correct) / len(closeness_correct) if closeness_correct else 0.0
    avg_optimal = sum(closeness_optimal) / len(closeness_optimal) if closeness_optimal else 0.0
    follow_rate = (hints_followed / hints_given) if hints_given else 0.0

    problem_score = max(1.0, min(5.0, round(avg_correct * 5, 2)))
    efficiency_score = max(1.0, min(5.0, round(avg_optimal * 5, 2)))
    communication_score = 3.0
    if get_running_summary(body.session_id):
        communication_score += 0.5
    if hints_given:
        communication_score -= 0.5
        communication_score += min(1.0, follow_rate * 2)
    communication_score = max(1.0, min(5.0, round(communication_score, 2)))
    readiness_score = round((problem_score + efficiency_score + communication_score) / 3, 2)

    summary_text = get_running_summary(body.session_id)
    parts: List[str] = []
    if summary_text:
        parts.append(summary_text)
    parts.append(
        f"Correctness closeness {avg_correct:.0%}, efficiency closeness {avg_optimal:.0%}."
    )
    if hints_given:
        parts.append(f"Hints followed {hints_followed}/{hints_given}.")
    else:
        parts.append("No hints were needed.")
    narrative = " ".join(parts)

    final = FinalEval(
        session_id=body.session_id,
        scores={
            "problem_solving": problem_score,
            "efficiency": efficiency_score,
            "communication": communication_score,
            "readiness": readiness_score,
        },
        narrative=narrative,
        running_summary=summary_text or None,
    )
    append_session_event(
        body.session_id,
        {
            "ts": time.time(),
            "type": "interview.final",
            "payload": final.dict(),
        },
    )
    return final


# ------------------------ Orchestrator Endpoints -------------------------


@app.post("/orchestrator/trigger")
async def orchestrator_trigger(event: TriggerEvent) -> Dict[str, Any]:
    decision = orchestrator.handle_event(event)
    executed = await execute_decision(event.session_id, decision)
    return {
        "actions": [action.dict() for action in decision.actions],
        "executed": executed,
        "state": orchestrator.get_state(event.session_id),
    }


@app.post("/orchestrator/present-problem", response_model=MonacoPayload)
async def orchestrator_present_problem(body: PresentProblemRequest) -> MonacoPayload:
    payload = await orchestrator.present_problem(
        body.session_id,
        rating=body.rating,
        tags=body.tags,
        problem_id=body.problem_id,
        broadcast=body.broadcast,
    )
    return payload


@app.post("/orchestrator/idle-tick")
async def orchestrator_idle_tick(body: IdleTickRequest) -> Dict[str, Any]:
    now = time.time()
    state_snapshot = orchestrator.get_state(body.session_id)
    last_activity = state_snapshot.get("last_activity_ts") if state_snapshot else None
    if last_activity is None:
        idle_ms = 0
    else:
        idle_ms = int(max(0.0, (now - last_activity) * 1000))
    event = TriggerEvent(
        session_id=body.session_id,
        kind="idle.tick",
        payload={"idle_ms": idle_ms},
        ts=now,
    )
    decision = orchestrator.handle_event(event)
    executed = await execute_decision(body.session_id, decision)
    return {
        "idle_ms": idle_ms,
        "actions": [action.dict() for action in decision.actions],
        "executed": executed,
        "state": orchestrator.get_state(body.session_id),
    }


@app.get("/timeline", response_model=TimelinePage)
async def get_timeline(
    session_id: str = Query(...),
    kind: str = Query("any"),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> TimelinePage:
    if kind not in TIMELINE_KINDS:
        raise HTTPException(status_code=400, detail="Unsupported timeline kind filter")
    events = read_session_events(session_id)
    if kind == "any":
        filtered = events
    else:
        filtered = [event for event in events if event.get("type") == kind]
    total = len(filtered)
    slice_end = min(offset + limit, total)
    items = filtered[offset:slice_end]
    next_offset = slice_end if slice_end < total else None
    return TimelinePage(
        session_id=session_id,
        total=total,
        items=items,
        next_offset=next_offset,
    )


@app.get("/timeline/summary", response_model=TimelineSummary)
async def timeline_summary(session_id: str = Query(...)) -> TimelineSummary:
    events = read_session_events(session_id)
    counts: Dict[str, int] = {}
    diff_totals = {"added": 0, "removed": 0, "changed": 0}
    last_versions: Dict[Tuple[str, str], int] = {}
    last_pass_ratio: Optional[float] = None

    for event in events:
        event_type = event.get("type")
        if not event_type:
            continue
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == "diff.snapshot":
            summary = event.get("summary", {})
            diff_totals["added"] += int(summary.get("added", 0))
            diff_totals["removed"] += int(summary.get("removed", 0))
            diff_totals["changed"] += int(summary.get("lines_changed", 0))
            problem_id = event.get("problem_id")
            language = event.get("language")
            version = event.get("version") or summary.get("version")
            if problem_id and language and version:
                last_versions[(str(problem_id), str(language))] = int(version)
        elif event_type == "exec.result":
            result = event.get("result", {})
            total = result.get("total")
            passed = result.get("passed")
            if total:
                try:
                    last_pass_ratio = (float(passed or 0) / float(total)) if total else None
                except Exception:  # pragma: no cover - defensive
                    last_pass_ratio = None

    last_versions_list = [
        {"problem_id": key[0], "language": key[1], "version": version}
        for key, version in last_versions.items()
    ]

    return TimelineSummary(
        session_id=session_id,
        counts=counts,
        diff_totals=diff_totals,
        last_versions=last_versions_list,
        last_pass_ratio=last_pass_ratio,
    )
