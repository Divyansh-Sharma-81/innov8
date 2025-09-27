from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import httpx
import yaml

from backend.services.transcriber.schema import WindowPayload

from .schema import MiniInput, MiniOutput

_ALLOWED_TAGS = {
    "still_speaking",
    "stuck",
    "claims_done",
    "nervous",
    "asking_hint",
    "rambling",
    "silence_long",
    "confident",
}

_SYSTEM_PROMPT = (
    "You label the candidate’s speaking state from transcript. Output ONLY YAML with keys: "
    "tags, scores, speak_gate, reason. Tags allowed: still_speaking, stuck, claims_done, "
    "nervous, asking_hint, rambling, silence_long, confident. Choose at most 3 tags. "
    "Scores only for the chosen tags (0..1). speak_gate = hold if actively speaking, else ok. "
    "One-sentence reason. No extra commentary."
)


class MiniLLMService:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: str,
        model_id: str,
        use_mock: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.use_mock = use_mock
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if not self.use_mock and self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=20.0)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def call(self, payload: WindowPayload | MiniInput) -> MiniOutput:
        if self.use_mock:
            return self._mock(payload)
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        request_body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "session_id": payload.session_id,
                            "window_s": payload.window_s,
                            "transcript_window": payload.transcript_window,
                            "transcript_pretext_60s": payload.transcript_pretext_60s,
                            "prosody_window": payload.prosody_window,
                            "recent_context": payload.recent_context,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        client = self._client or httpx.AsyncClient(base_url=self.base_url, timeout=20.0)
        created_here = client is not self._client
        try:
            response = await client.post("/chat/completions", json=request_body, headers=headers)
            response.raise_for_status()
            data = response.json()
        finally:
            if created_here:
                await client.aclose()
        content = data["choices"][0]["message"]["content"]
        parsed = yaml.safe_load(content) or {}
        return self._coerce_output(parsed)

    def _coerce_output(self, data: Dict[str, Any]) -> MiniOutput:
        raw_tags = list(data.get("tags") or [])
        tags = [t for t in raw_tags if t in _ALLOWED_TAGS][:3]
        scores_src = data.get("scores") or {}
        scores: Dict[str, float] = {}
        for tag in tags:
            value = float(scores_src.get(tag, 0.7))
            scores[tag] = max(0.0, min(1.0, value))
        speak_gate = "hold" if data.get("speak_gate") == "hold" else "ok"
        reason = str(data.get("reason", "")) or "No reason supplied"
        return MiniOutput(tags=tags, scores=scores, speak_gate=speak_gate, reason=reason, timestamp=time.time())

    def _mock(self, payload: WindowPayload | MiniInput) -> MiniOutput:
        text_lower = payload.transcript_window.lower()
        tags: list[str] = []
        if "done" in text_lower:
            tags.append("claims_done")
        if "stuck" in text_lower or "don't know" in text_lower or "dont know" in text_lower:
            tags.append("stuck")
        prosody = payload.prosody_window or {}
        wpm = float(prosody.get("wpm", 0))
        filler_ratio = float(prosody.get("filler_ratio", 0))
        if wpm > 180 and filler_ratio > 0.12:
            tags.append("nervous")
        idle_ms = int(payload.recent_context.get("idle_ms", payload.window_s * 1000))
        if idle_ms < 2000 and payload.transcript_window.strip():
            tags.append("still_speaking")
        deduped: list[str] = []
        for tag in tags:
            if tag in _ALLOWED_TAGS and tag not in deduped:
                deduped.append(tag)
            if len(deduped) == 3:
                break
        speak_gate = "hold" if "still_speaking" in deduped else "ok"
        scores = {tag: round(0.6 + idx * 0.1, 2) for idx, tag in enumerate(deduped)}
        reason = ", ".join(deduped) if deduped else "no notable state"
        return MiniOutput(tags=deduped, scores=scores, speak_gate=speak_gate, reason=reason, timestamp=time.time())
