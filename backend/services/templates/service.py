from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import Column, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .schema import InterviewPlan, Question

Base = declarative_base()


class InterviewPlanRow(Base):
    __tablename__ = "interview_plans"

    session_id = Column(String(64), primary_key=True)
    coding_goal = Column(Integer, nullable=False, default=2)
    coding_done = Column(Integer, nullable=False, default=0)
    hr_goal = Column(Integer, nullable=False, default=1)
    hr_done = Column(Integer, nullable=False, default=0)
    apt_goal = Column(Integer, nullable=False, default=1)
    apt_done = Column(Integer, nullable=False, default=0)
    created_at = Column(Float, nullable=False, default=lambda: time.time())
    updated_at = Column(Float, nullable=False, default=lambda: time.time(), onupdate=lambda: time.time())


class HRQuestionRow(Base):
    __tablename__ = "hr_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    category = Column(String(64), nullable=True)


class AptitudeQuestionRow(Base):
    __tablename__ = "aptitude_questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, nullable=False)
    category = Column(String(64), nullable=True)


def _load_database_url(database_url: Optional[str] = None) -> str:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured for templates service")
    return url


class TemplatesService:
    """Interview templates and counters stored in Postgres."""

    def __init__(self, database_url: Optional[str] = None, *, create_tables: bool = True) -> None:
        self._engine: Engine = create_engine(_load_database_url(database_url))
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        if create_tables:
            Base.metadata.create_all(self._engine)
        self._ensure_seed_data()

    @contextmanager
    def session(self) -> Iterator[Session]:  # type: ignore[override]
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_or_create_plan(self, session_id: str) -> InterviewPlan:
        now = time.time()
        with self.session() as sess:
            plan = sess.get(InterviewPlanRow, session_id)
            if plan is None:
                plan = InterviewPlanRow(session_id=session_id, created_at=now, updated_at=now)
                sess.add(plan)
                sess.flush()
            else:
                plan.updated_at = now
            return self._to_plan(plan)

    def increment(self, session_id: str, kind: str) -> InterviewPlan:
        kind = kind.lower()
        with self.session() as sess:
            plan = sess.get(InterviewPlanRow, session_id)
            if plan is None:
                plan = InterviewPlanRow(session_id=session_id)
                sess.add(plan)
            if kind == "coding":
                plan.coding_done += 1
            elif kind in {"hr", "ask_hr"}:
                plan.hr_done += 1
            elif kind in {"apt", "aptitude", "ask_aptitude"}:
                plan.apt_done += 1
            else:
                raise ValueError(f"Unsupported increment kind: {kind}")
            plan.updated_at = time.time()
            sess.flush()
            return self._to_plan(plan)

    def reset_plan(self, session_id: str) -> InterviewPlan:
        with self.session() as sess:
            plan = sess.get(InterviewPlanRow, session_id)
            if plan is None:
                plan = InterviewPlanRow(session_id=session_id)
                sess.add(plan)
            plan.coding_done = 0
            plan.hr_done = 0
            plan.apt_done = 0
            plan.updated_at = time.time()
            sess.flush()
            return self._to_plan(plan)

    def sample_hr(self) -> Question:
        with self.session() as sess:
            stmt = select(HRQuestionRow).order_by(func.random()).limit(1)
            row = sess.execute(stmt).scalars().first()
            if row is None:
                raise RuntimeError("HR question bank is empty")
            return Question(id=row.id, text=row.text, kind="hr", category=row.category)

    def sample_aptitude(self) -> Question:
        with self.session() as sess:
            stmt = select(AptitudeQuestionRow).order_by(func.random()).limit(1)
            row = sess.execute(stmt).scalars().first()
            if row is None:
                raise RuntimeError("Aptitude question bank is empty")
            return Question(id=row.id, text=row.text, kind="aptitude", category=row.category)

    def _ensure_seed_data(self) -> None:
        defaults_hr = [
            ("Tell me about a time you had to give difficult feedback to a teammate.", "collaboration"),
            ("What kind of engineering culture helps you do your best work?", "culture"),
            ("How do you keep learning outside your day-to-day responsibilities?", "growth"),
        ]
        defaults_apt = [
            ("You have two jugs of 3L and 5L. How do you measure exactly 4L?", "brain-teaser"),
            ("If a function doubles input size each minute starting from 1, when does it reach 64?", "math"),
            ("How many edges does a complete graph with n nodes have?", "combinatorics"),
        ]
        with self.session() as sess:
            hr_count = sess.execute(select(func.count(HRQuestionRow.id))).scalar_one()
            if hr_count == 0:
                sess.bulk_save_objects(
                    [HRQuestionRow(text=text, category=category) for text, category in defaults_hr]
                )
            apt_count = sess.execute(select(func.count(AptitudeQuestionRow.id))).scalar_one()
            if apt_count == 0:
                sess.bulk_save_objects(
                    [AptitudeQuestionRow(text=text, category=category) for text, category in defaults_apt]
                )

    def _to_plan(self, row: InterviewPlanRow) -> InterviewPlan:
        return InterviewPlan(
            session_id=row.session_id,
            coding_goal=row.coding_goal,
            coding_done=row.coding_done,
            hr_goal=row.hr_goal,
            hr_done=row.hr_done,
            apt_goal=row.apt_goal,
            apt_done=row.apt_done,
        )


__all__ = ["TemplatesService"]
