from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class InterviewPlan(BaseModel):
    session_id: str
    coding_goal: int
    coding_done: int
    hr_goal: int
    hr_done: int
    apt_goal: int
    apt_done: int


class Question(BaseModel):
    id: int
    text: str
    kind: Literal["hr", "aptitude"]
    category: Optional[str] = None
