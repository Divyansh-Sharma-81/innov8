#!/usr/bin/env python3
"""Normalize innov8 problem CSVs so multi-line fields become single-line."""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE_PATTERN = re.compile(r"innov8 dataset - (\d+)\.csv$")


def flatten_field(value: str) -> str:
    """Replace real newlines in a CSV field with literal \n tokens."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", r"\n")


def process_file(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        rows = [[flatten_field(cell) for cell in row] for row in reader]

    with destination.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)


def main() -> None:
    targets = []
    for candidate in ROOT.glob("innov8 dataset - *.csv"):
        match = SOURCE_PATTERN.match(candidate.name)
        if not match:
            continue
        rating = match.group(1)
        targets.append((candidate, ROOT / f"data-{rating}.csv"))

    if not targets:
        print("No matching CSV files found. Nothing to do.")
        return

    for source, destination in targets:
        process_file(source, destination)
        print(f"Wrote {destination.name} ({source.name} -> {destination.name})")


if __name__ == "__main__":
    main()
