from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, List, Optional

from sqlalchemy import Column, ForeignKey, Integer, String, Text, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker, joinedload

from .schema import IOSampleDTO, MonacoPayload, ProblemDTO

from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class Problem(Base):
    __tablename__ = "problems"

    id = Column(String(36), primary_key=True)
    url = Column(Text, unique=True, nullable=False)
    title = Column(Text, nullable=False)
    rating = Column(Integer, index=True, nullable=False)
    statement_md = Column(Text, nullable=False)
    boilerplate_py = Column(Text, nullable=False)
    boilerplate_cpp = Column(Text, nullable=True)

    tags = relationship("ProblemTag", back_populates="problem", cascade="all, delete-orphan")
    io_samples = relationship("IOSample", back_populates="problem", cascade="all, delete-orphan")


class ProblemTag(Base):
    __tablename__ = "problem_tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(String(36), ForeignKey("problems.id", ondelete="CASCADE"), index=True, nullable=False)
    tag = Column(Text, index=True, nullable=False)

    problem = relationship("Problem", back_populates="tags")


class IOSample(Base):
    __tablename__ = "io_samples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    problem_id = Column(String(36), ForeignKey("problems.id", ondelete="CASCADE"), index=True, nullable=False)
    stdin = Column(Text, nullable=False)
    expected_stdout = Column(Text, nullable=False)

    problem = relationship("Problem", back_populates="io_samples")


def _load_database_url(database_url: Optional[str] = None) -> str:
    url = database_url or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return url


class ProblemsetService:
    """Data access layer for coding problems and associated metadata."""

    def __init__(self, database_url: Optional[str] = None, *, create_tables: bool = True) -> None:
        self._engine: Engine = create_engine(_load_database_url(database_url))
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        if create_tables:
            Base.metadata.create_all(self._engine)

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

    def list_ratings(self) -> List[int]:
        with self.session() as sess:
            rows = sess.execute(select(Problem.rating).distinct().order_by(Problem.rating)).scalars().all()
        return list(rows)

    def get_problem(self, problem_id: str) -> ProblemDTO:
        with self.session() as sess:
            query = (
                sess.query(Problem)
                .options(joinedload(Problem.tags), joinedload(Problem.io_samples))
                .filter(Problem.id == problem_id)
            )
            instance = query.one_or_none()
            if instance is None:
                raise NoResultFound(f"Problem {problem_id} not found")
            return _to_problem_dto(instance)

    def get_samples(self, problem_id: str) -> List[IOSampleDTO]:
        with self.session() as sess:
            samples = (
                sess.query(IOSample)
                .filter(IOSample.problem_id == problem_id)
                .order_by(IOSample.id)
                .all()
            )
            return [_to_sample_dto(sample) for sample in samples]

    def search(
        self,
        *,
        rating: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 1,
    ) -> List[ProblemDTO]:
        tags = [tag.lower().strip() for tag in tags or [] if tag.strip()]
        with self.session() as sess:
            query = (
                sess.query(Problem)
                .options(joinedload(Problem.tags))
            )
            if rating is not None:
                query = query.filter(Problem.rating == rating)
            if tags:
                tag_subquery = (
                    select(ProblemTag.problem_id)
                    .filter(ProblemTag.tag.in_(tags))
                    .group_by(ProblemTag.problem_id)
                )
                query = query.filter(Problem.id.in_(tag_subquery))
            query = query.order_by(func.random()).limit(max(1, limit))
            results = query.all()
            return [_to_problem_dto(problem) for problem in results]

    def monaco_payload(
        self,
        *,
        problem_id: Optional[str] = None,
        rating: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 1,
    ) -> MonacoPayload:
        if problem_id:
            problem = self.get_problem(problem_id)
            samples = self.get_samples(problem_id)
            return MonacoPayload(problem=problem, tests=samples)

        matches = self.search(rating=rating, tags=tags, limit=limit)
        if not matches:
            raise NoResultFound("No problems match the requested criteria")
        chosen = matches[0]
        samples = self.get_samples(chosen.id)
        return MonacoPayload(problem=chosen, tests=samples)


_DEFAULT_BOILERPLATE = """import sys\n\n\ndef solve():\n    data = sys.stdin.read().strip().split()\n    # TODO: implement\n    print()\n\n\nif __name__ == "__main__":\n    solve()\n"""


def ensure_boilerplate(value: Optional[str]) -> str:
    if value and value.strip():
        return value
    return _DEFAULT_BOILERPLATE


def _to_problem_dto(problem: Problem) -> ProblemDTO:
    tags = sorted({tag.tag for tag in problem.tags})
    return ProblemDTO(
        id=problem.id,
        url=problem.url,
        title=problem.title,
        rating=problem.rating,
        statement_md=problem.statement_md,
        boilerplate_py=ensure_boilerplate(problem.boilerplate_py),
        boilerplate_cpp=problem.boilerplate_cpp,
        tags=tags,
    )


def _to_sample_dto(sample: IOSample) -> IOSampleDTO:
    return IOSampleDTO(stdin=sample.stdin, expected_stdout=sample.expected_stdout)
