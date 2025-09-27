from __future__ import annotations

from typing import Any, Dict, Union

from pydantic import BaseModel, Field


class TranscriptChunk(BaseModel):
    session_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    start_ts: float = Field(..., ge=0)
    end_ts: float = Field(..., ge=0)
    is_final: bool = True

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "demo",
                "text": "uh I think I am done",
                "start_ts": 0.0,
                "end_ts": 9.7,
                "is_final": True,
            }
        }
    }


class WindowPayload(BaseModel):
    session_id: str
    window_s: int
    pretext_s: int
    transcript_window: str
    transcript_pretext_60s: str
    prosody_window: Dict[str, Union[int, float]]
    recent_context: Dict[str, Any] = Field(default_factory=dict)
