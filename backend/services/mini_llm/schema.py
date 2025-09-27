from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from backend.services.transcriber.schema import WindowPayload


class MiniInput(WindowPayload):
    pass


class MiniOutput(BaseModel):
    tags: List[str] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)
    speak_gate: Literal["hold", "ok"] = "ok"
    reason: str
    timestamp: float
