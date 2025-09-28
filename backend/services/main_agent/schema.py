from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


AllowedAction = Literal[
    "stay_silent",
    "send_message",
    "give_hint",
    "present_problem",
    "ask_hr",
    "ask_aptitude",
]


class AgentIn(BaseModel):
    session_id: str
    transcript_pretext_60s: str
    transcript_window_10s: str
    mini_tags: List[str]
    speak_gate: Literal["hold", "ok"]
    last_exec_summary: str
    last_diff_summary: str
    plan_state: Dict[str, int]
    last_agent_summary: str
    candidate_name: Optional[str] = None
    role: str = "entry_swe"


class AgentOut(BaseModel):
    action: AllowedAction
    text: str
    hint_level: Optional[Literal["nudge", "guide", "direction"]] = None
    next_rating: Optional[int] = None
    desired_tags: Optional[List[str]] = None
    update_running_summary: Optional[str] = None


class RunningSummary(BaseModel):
    session_id: str
    summary: str
    updated_at: float


class FinalEval(BaseModel):
    session_id: str
    scores: Dict[str, float]
    narrative: str
    running_summary: Optional[str] = None
