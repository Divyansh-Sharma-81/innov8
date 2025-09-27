from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple

from .schema import TranscriptChunk, WindowPayload


_FILLERS = ["um", "uh", "like", "you know", "maybe", "i guess"]
_DEBOUNCE_SECONDS = 0.5

WindowConsumer = Callable[[WindowPayload], Awaitable[Any]]


class TranscriberService:
    """Stores transcript chunks, builds analysis windows, and forwards them."""

    def __init__(self) -> None:
        self._buffers: Dict[str, Deque[TranscriptChunk]] = defaultdict(deque)
        self._last_window_ts: Dict[str, float] = defaultdict(float)
        self._window_consumer: Optional[WindowConsumer] = None
        self._lock = RLock()

    def set_window_consumer(self, fn: Optional[WindowConsumer]) -> None:
        self._window_consumer = fn

    async def handle_chunk(
        self,
        chunk: TranscriptChunk,
        window_seconds: int,
        pretext_seconds: int,
        *,
        debounce: bool = True,
    ) -> Optional[Tuple[WindowPayload, Any]]:
        self.ingest_chunk(chunk)
        window = self.build_window(
            session_id=chunk.session_id,
            window_seconds=window_seconds,
            pretext_seconds=pretext_seconds,
        )
        if not window:
            return None

        if debounce and self._should_skip(chunk.session_id, chunk.end_ts):
            return window, None

        consumer_result = None
        if self._window_consumer:
            consumer_result = await self._window_consumer(window)
        self._mark_processed(chunk.session_id, chunk.end_ts)
        return window, consumer_result

    def ingest_chunk(self, chunk: TranscriptChunk) -> None:
        if not chunk.is_final:
            return
        with self._lock:
            buf = self._buffers[chunk.session_id]
            buf.append(chunk)
            self._prune(chunk.session_id, tail_hint=chunk.end_ts)

    def build_window(
        self,
        session_id: str,
        window_seconds: int = 10,
        pretext_seconds: int = 60,
    ) -> Optional[WindowPayload]:
        with self._lock:
            buf = self._buffers.get(session_id)
            if not buf:
                return None
            now = buf[-1].end_ts
            window_floor = now - window_seconds
            pretext_floor = now - pretext_seconds
            window_chunks = [c for c in buf if window_floor < c.end_ts <= now]
            if not window_chunks:
                return None
            pretext_chunks = [c for c in buf if pretext_floor < c.end_ts <= now]

        transcript_window = " ".join(filter(None, (c.text.strip() for c in window_chunks))).strip()
        transcript_pretext = " ".join(
            filter(None, (c.text.strip() for c in pretext_chunks))
        ).strip()

        prosody = self._prosody(transcript_window, window_seconds)
        recent_context = {
            "idle_ms": int(max(0.0, now - window_chunks[-1].end_ts) * 1000),
            "last_agent_action": None,
            "last_verdict": None,
        }

        return WindowPayload(
            session_id=session_id,
            window_s=window_seconds,
            pretext_s=pretext_seconds,
            transcript_window=transcript_window,
            transcript_pretext_60s=transcript_pretext,
            prosody_window=prosody,
            recent_context=recent_context,
        )

    def _prune(self, session_id: str, tail_hint: float, keep_seconds: int = 65) -> None:
        buf = self._buffers.get(session_id)
        if not buf:
            return
        cutoff = tail_hint - keep_seconds
        while buf and buf[0].end_ts < cutoff:
            buf.popleft()
        if not buf:
            self._buffers.pop(session_id, None)

    def _should_skip(self, session_id: str, end_ts: float) -> bool:
        with self._lock:
            last = self._last_window_ts.get(session_id, 0.0)
        return bool(last and end_ts - last < _DEBOUNCE_SECONDS)

    def _mark_processed(self, session_id: str, end_ts: float) -> None:
        with self._lock:
            self._last_window_ts[session_id] = end_ts

    @staticmethod
    def _prosody(transcript: str, window_seconds: int) -> Dict[str, float]:
        if window_seconds <= 0:
            window_seconds = 1
        words = re.findall(r"\b\w+\b", transcript.lower())
        word_count = len(words)
        wpm = (word_count / max(window_seconds, 1e-9)) * 60
        non_space = len(re.sub(r"\s+", "", transcript))
        total_chars = max(1, len(transcript))
        speech_ms = int((non_space / total_chars) * window_seconds * 1000)
        silence_ms = max(0, window_seconds * 1000 - speech_ms)
        filler_count = 0
        lowered = transcript.lower()
        for filler in _FILLERS:
            if " " in filler:
                filler_count += lowered.count(filler)
            else:
                filler_count += len(re.findall(rf"\b{re.escape(filler)}\b", lowered))
        filler_ratio = filler_count / max(1, word_count)
        return {
            "speech_ms": speech_ms,
            "silence_ms": silence_ms,
            "wpm": round(wpm, 2),
            "filler_ratio": round(filler_ratio, 3),
        }


@dataclass
class _TurnData:
    text: str
    start_ts: float
    end_ts: float


@dataclass
class _LiveSession:
    session_id: str
    queue: asyncio.Queue[_TurnData] = field(default_factory=asyncio.Queue)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task[None]] = None
    t0: float = field(default_factory=time.time)
    backoff: float = 0.0
    device_label: Optional[str] = None
    device_index: Optional[int] = None


class TranscriberRunner:
    """Manages live AssemblyAI streams and feeds finalized turns into the service."""

    def __init__(
        self,
        *,
        transcriber: TranscriberService,
        api_key: Optional[str],
        window_seconds: int,
        pretext_seconds: int,
        sample_rate: int,
        channels: int,
        audio_format: str,
        reconnect_secs: float = 2.0,
    ) -> None:
        self._transcriber = transcriber
        self._api_key = api_key
        self._window_seconds = window_seconds
        self._pretext_seconds = pretext_seconds
        self._sample_rate = sample_rate
        self._channels = channels
        self._audio_format = audio_format
        self._reconnect_secs = max(0.5, reconnect_secs)
        self._sessions: Dict[str, _LiveSession] = {}
        self._chunk_logger: Optional[Callable[[TranscriptChunk], None]] = None
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger("transcriber.runner")
        if not self._logger.handlers:
            logging.basicConfig(level=logging.INFO)
        self._logger.setLevel(logging.INFO)

    def set_chunk_logger(self, fn: Optional[Callable[[TranscriptChunk], None]]) -> None:
        self._chunk_logger = fn

    async def start_stream(
        self,
        session_id: str,
        device_label: Optional[str] = None,
        device_index: Optional[int] = None,
    ) -> None:
        if not self._api_key:
            raise RuntimeError("ASSEMBLYAI_API_KEY not configured")
        async with self._lock:
            if session_id in self._sessions:
                raise RuntimeError(f"Session {session_id} already running")
            session = _LiveSession(
                session_id=session_id,
                device_label=device_label,
                device_index=device_index,
            )
            session.backoff = self._reconnect_secs
            self._sessions[session_id] = session
            session.task = asyncio.create_task(self._session_worker(session))
            self._logger.info(
                "Started ASR stream for session %s (device_label=%s, device_index=%s)",
                session_id,
                device_label,
                device_index,
            )

    async def stop_stream(self, session_id: str) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
        if not session:
            raise RuntimeError(f"Session {session_id} not running")
        session.stop_event.set()
        if session.task:
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            self._sessions.pop(session_id, None)
        self._logger.info("Stopped ASR stream for session %s", session_id)

    async def stop_all(self) -> None:
        async with self._lock:
            session_ids = list(self._sessions.keys())
        for sid in session_ids:
            try:
                await self.stop_stream(sid)
            except RuntimeError:
                pass

    async def _session_worker(self, session: _LiveSession) -> None:
        loop = asyncio.get_running_loop()
        backoff = session.backoff
        while not session.stop_event.is_set():
            session.queue = asyncio.Queue()
            runner = loop.run_in_executor(
                None,
                self._run_stream_once,
                session,
                loop,
            )
            try:
                while not session.stop_event.is_set():
                    try:
                        turn = await asyncio.wait_for(session.queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    if turn is None:
                        break
                    await self._handle_turn(session, turn)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.exception("Session %s worker error: %s", session.session_id, exc)
                if session.stop_event.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            finally:
                try:
                    await asyncio.wrap_future(runner)
                except Exception as exc:  # pragma: no cover
                    self._logger.exception("Streaming task failed for %s: %s", session.session_id, exc)
            if session.stop_event.is_set():
                break
            backoff = self._reconnect_secs
        self._logger.info("ASR worker exiting for session %s", session.session_id)

    async def _handle_turn(self, session: _LiveSession, turn: _TurnData) -> None:
        chunk = TranscriptChunk(
            session_id=session.session_id,
            text=turn.text,
            start_ts=max(0.0, turn.start_ts),
            end_ts=max(turn.start_ts, turn.end_ts),
            is_final=True,
        )
        self._logger.info(
            "Session %s finalized turn %.2f→%.2f: %s",
            session.session_id,
            chunk.start_ts,
            chunk.end_ts,
            chunk.text,
        )
        if self._chunk_logger:
            try:
                self._chunk_logger(chunk)
            except Exception:  # pragma: no cover - logging safety
                self._logger.exception("Chunk logger failed", exc_info=True)
        await self._transcriber.handle_chunk(
            chunk,
            self._window_seconds,
            self._pretext_seconds,
            debounce=True,
        )

    def _run_stream_once(self, session: _LiveSession, loop: asyncio.AbstractEventLoop) -> None:
        try:
            from assemblyai.streaming.v3 import (
                StreamingClient,
                StreamingClientOptions,
                StreamingEvents,
                StreamingParameters,
            )
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("assemblyai package is required for live streaming") from exc

        queue_put = session.queue.put_nowait
        logger = self._logger

        def on_begin(_c, event) -> None:
            session.t0 = time.time()
            logger.info(
                "AAI stream began for %s: %s",
                session.session_id,
                getattr(event, "id", "?"),
            )

        def on_turn(_c, event) -> None:
            if not getattr(event, "end_of_turn", False):
                return
            text = getattr(event, "formatted_transcript", None) or getattr(event, "transcript", "")
            text = (text or "").strip()
            if not text:
                return
            now = time.time()
            end_ts = max(0.0, now - session.t0)
            approx_duration = max(0.3, len(text.split()) * 0.35)
            start_ts = max(0.0, end_ts - approx_duration)
            loop.call_soon_threadsafe(queue_put, _TurnData(text=text, start_ts=start_ts, end_ts=end_ts))

        def on_error(_c, err) -> None:
            logger.warning("AAI stream error for %s: %s", session.session_id, err)

        client = StreamingClient(
            StreamingClientOptions(api_key=self._api_key, api_host="streaming.assemblyai.com")
        )
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)

        device_index = None
        try:
            device_index = self._resolve_device_index(session.device_label, session.device_index)
            audio_iter = self._mic_chunks(session, device_index)
        except Exception as exc:  # pragma: no cover - environment specific
            loop.call_soon_threadsafe(queue_put, None)
            raise RuntimeError(f"Audio input error: {exc}") from exc

        client.connect(
            StreamingParameters(
                sample_rate=self._sample_rate,
                encoding=self._audio_format,
                format_turns=True,
                min_end_of_turn_silence_when_confident="400",
                max_turn_silence="1500",
            )
        )

        try:
            client.stream(audio_iter)
        finally:
            try:
                client.disconnect(terminate=True)
            finally:
                loop.call_soon_threadsafe(queue_put, None)

    def _mic_chunks(self, session: _LiveSession, device_index: Optional[int]):
        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("PyAudio is required for live streaming") from exc

        pa = pyaudio.PyAudio()
        stream = None
        frames_per_buffer = 4096
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self._channels,
                rate=self._sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=frames_per_buffer,
            )
            self._logger.info(
                "Session %s capturing audio (device_index=%s)",
                session.session_id,
                device_index,
            )
            while not session.stop_event.is_set():
                yield stream.read(frames_per_buffer, exception_on_overflow=False)
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()
            self._logger.info("Session %s microphone closed", session.session_id)

    def _resolve_device_index(
        self,
        device_label: Optional[str],
        device_index: Optional[int],
    ) -> Optional[int]:
        try:
            import pyaudio  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment specific
            raise RuntimeError("PyAudio is required for device resolution") from exc

        pa = pyaudio.PyAudio()
        try:
            if device_index is not None:
                self._logger.info("Using explicit device index %s", device_index)
                return device_index
            if not device_label:
                self._logger.info("No device label provided; using system default input")
                return None
            match = device_label.lower()
            for index in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(index)
                name = info.get("name", "").lower()
                if match in name:
                    self._logger.info("Resolved device '%s' to index %s", device_label, index)
                    return index
            self._logger.warning(
                "No input device matched '%s'; using default",
                device_label,
            )
            return None
        finally:
            pa.terminate()
