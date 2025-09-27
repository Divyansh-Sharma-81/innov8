from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class IOSampleDTO(BaseModel):
    stdin: str
    expected_stdout: str


class ProblemDTO(BaseModel):
    id: str
    url: str
    title: str
    rating: int
    statement_md: str
    boilerplate_py: str
    boilerplate_cpp: Optional[str] = None
    tags: List[str]


class MonacoPayload(BaseModel):
    problem: ProblemDTO
    tests: List[IOSampleDTO]
