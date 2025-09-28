from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class SnapshotIn(BaseModel):
    session_id: str
    problem_id: str
    language: Literal["python"]
    source: str
    cursor_line: Optional[int] = None
    reason: Literal["save", "run"] = "save"


class DiffSummary(BaseModel):
    version: int
    timestamp: float
    lines_total: int
    lines_changed: int
    added: int
    removed: int
    hunks: int
    funcs_touched: List[str]


class SnapshotOut(BaseModel):
    session_id: str
    problem_id: str
    language: Literal["python"]
    version: int
    diff_summary: DiffSummary
    preview_diff: Optional[str] = None


class TimelineQuery(BaseModel):
    session_id: str
    kind: Literal["any", "diff.snapshot", "exec.result", "problem.presented", "mini.tag"] = "any"
    offset: int = 0
    limit: int = 100


class TimelinePage(BaseModel):
    session_id: str
    total: int
    items: List[Dict]
    next_offset: Optional[int]


class TimelineSummary(BaseModel):
    session_id: str
    counts: Dict[str, int]
    diff_totals: Dict[str, int]
    last_versions: List[Dict[str, Any]]
    last_pass_ratio: Optional[float] = None
