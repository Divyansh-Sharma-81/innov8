from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel


class PolicyConfig(BaseModel):
    base_rating: int = 800
    step_up: int = 200
    step_keep: int = 100
    step_down: int = 100
    min_rating: int = 800
    max_rating: int = 1400
    chat_cooldown_s: int = 10
    hint_cooldown_s: int = 25
    idle_threshold_ms: int = 60000


class TriggerEvent(BaseModel):
    session_id: str
    kind: Literal["mini.tag", "exec.result", "ui.editor.save", "idle.tick"]
    payload: Dict[str, Any] = {}
    ts: float


class AgentAction(BaseModel):
    type: Literal["present_problem", "send_message", "give_hint", "stay_silent"]
    session_id: str
    data: Dict[str, Any] = {}
    reason: str
    cooldown_s: int = 0


class Decision(BaseModel):
    actions: List[AgentAction]
    updated_state: Dict[str, Any]
