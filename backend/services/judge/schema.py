from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ExecRequest(BaseModel):
    language: str
    source: str
    stdin: str


class ExecCaseResult(BaseModel):
    stdin: str
    expected: str
    stdout: str
    passed: bool
    time_ms: Optional[float] = None
    memory_kb: Optional[int] = None
    stderr: Optional[str] = None


class ExecResult(BaseModel):
    status: str
    passed: int
    total: int
    cases: List[ExecCaseResult]
    stderr: Optional[str] = None
    mock: bool = False
    closeness_to_correct: Optional[float] = None
    closeness_to_optimal: Optional[float] = None
