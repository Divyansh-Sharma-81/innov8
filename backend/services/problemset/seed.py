from __future__ import annotations

import argparse
import csv
import json
import uuid
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sqlalchemy.orm import Session

from .service import IOSample, Problem, ProblemTag, ProblemsetService, ensure_boilerplate


DATA_DIR = Path(__file__).resolve().parents[2] / "problemset"
TAGS_FILE = DATA_DIR / "tags.json"
CSV_PATTERN = "data-*.csv"


def load_rating_tags(path: Path) -> Dict[int, List[str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(rating): [tag.lower() for tag in tags] for rating, tags in data.items()}


def _normalize_blob(value: str, *, trim: bool = False) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\\n", "\n").replace("\xa0", " ")
    return text.strip() if trim else text


def _normalize_payload(items: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        value = "" if item is None else str(item)
        normalized.append(_normalize_blob(value, trim=False))
    return normalized


def _parse_json_samples(raw: str) -> List[Tuple[str, str]]:
    payload = json.loads(raw)
    pairs: List[Tuple[str, str]] = []
    if isinstance(payload, dict):
        inputs = payload.get("inputs") or payload.get("input") or []
        outputs = payload.get("outputs") or payload.get("output") or []
        for stdin, expected in zip_longest(_normalize_payload(inputs), _normalize_payload(outputs), fillvalue=""):
            pairs.append((_normalize_blob(stdin, trim=True), _normalize_blob(expected, trim=True)))
    elif isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            stdin = item.get("stdin") or item.get("input") or ""
            expected = item.get("stdout") or item.get("output") or ""
            pairs.append((_normalize_blob(stdin, trim=True), _normalize_blob(expected, trim=True)))
    return pairs


def _parse_line_pairs(raw: str) -> List[Tuple[str, str]]:
    chunks = [chunk.strip() for chunk in raw.split("\n\n") if chunk.strip()]
    pairs: List[Tuple[str, str]] = []
    pending_input: str | None = None
    for chunk in chunks:
        lower = chunk.lower()
        header, _, body = chunk.partition("\n")
        content = body if body else ""
        if lower.startswith("input"):
            pending_input = content
            continue
        if lower.startswith("output") and pending_input is not None:
            pairs.append((_normalize_blob(pending_input, trim=True), _normalize_blob(content, trim=True)))
            pending_input = None
            continue
        if pending_input is None:
            pending_input = chunk
        else:
            pairs.append((_normalize_blob(pending_input, trim=True), _normalize_blob(chunk, trim=True)))
            pending_input = None
    if pending_input is not None:
        pairs.append((_normalize_blob(pending_input, trim=True), ""))
    return pairs


def parse_samples(raw: str) -> List[Tuple[str, str]]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        pairs = _parse_json_samples(text)
    except json.JSONDecodeError:
        pairs = _parse_line_pairs(text)
    return [(stdin, expected) for stdin, expected in pairs if stdin or expected]


def derive_title(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) >= 2:
        return f"{segments[-2]} {segments[-1]}".replace("-", " ").title()
    if segments:
        return segments[-1].replace("-", " ").title()
    if parsed.netloc:
        return parsed.netloc
    return url


def normalize_tags(raw: str, allowed: List[str]) -> List[str]:
    tags = [tag.strip().lower() for tag in raw.split(",") if tag.strip()]
    if allowed:
        allowed_set = set(allowed)
        tags = [tag for tag in tags if tag in allowed_set]
    return sorted(set(tags))


def upsert_problem(session: Session, rating: int, row: Dict[str, str], allowed_tags: List[str]) -> None:
    url = row.get("problem link", "").strip()
    if not url:
        return
    statement = _normalize_blob(row.get("problem_statement", ""))
    title = derive_title(url)
    boilerplate_py = ensure_boilerplate(None)
    boilerplate_cpp = None
    tags = normalize_tags(row.get("tags", ""), allowed_tags)
    samples = parse_samples(row.get("input-output", ""))

    problem = session.query(Problem).filter(Problem.url == url).one_or_none()
    if not problem:
        problem_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}:{rating}"))
        problem = Problem(id=problem_id, url=url, rating=rating)
        session.add(problem)
    problem.title = title
    problem.rating = rating
    problem.statement_md = statement
    problem.boilerplate_py = boilerplate_py
    problem.boilerplate_cpp = boilerplate_cpp

    problem.tags.clear()
    for tag in tags:
        problem.tags.append(ProblemTag(tag=tag))

    problem.io_samples.clear()
    for stdin, expected in samples:
        problem.io_samples.append(IOSample(stdin=stdin, expected_stdout=expected))


def seed_from_csv(service: ProblemsetService, data_dir: Path = DATA_DIR) -> None:
    rating_tags = load_rating_tags(TAGS_FILE)
    csv_files = sorted(data_dir.glob(CSV_PATTERN))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    with service.session() as session:
        for csv_path in csv_files:
            stem = csv_path.stem  # e.g. data-800
            try:
                rating = int(stem.split("-")[-1])
            except ValueError:
                continue
            allowed = rating_tags.get(rating, [])
            with csv_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    upsert_problem(session, rating, row, allowed)
    # session context commits on success


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed problems into the database")
    parser.add_argument("--db", dest="database_url", default=None, help="Database URL (defaults to env DATABASE_URL)")
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        default=str(DATA_DIR),
        help="Directory containing data-*.csv files",
    )
    args = parser.parse_args()
    service = ProblemsetService(database_url=args.database_url)
    seed_from_csv(service, Path(args.data_dir))


if __name__ == "__main__":
    main()
