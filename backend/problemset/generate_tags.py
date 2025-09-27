#!/usr/bin/env python3
"""Collect unique tags per rating from normalized data-*.csv files."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
DATA_PATTERN = re.compile(r"data-(\d+)\.csv$")


def extract_tags(cell: str) -> set[str]:
    return {tag.strip() for tag in cell.split(',') if tag.strip()}


def main() -> None:
    tags_by_rating: dict[str, set[str]] = {}

    for path in sorted(ROOT.glob("data-*.csv")):
        match = DATA_PATTERN.match(path.name)
        if not match:
            continue
        rating = match.group(1)
        tags = tags_by_rating.setdefault(rating, set())

        with path.open("r", encoding="utf-8", newline="") as infile:
            reader = csv.reader(infile)
            header = next(reader, None)
            if not header:
                continue
            try:
                tags_index = header.index("tags")
            except ValueError:
                continue

            for row in reader:
                if len(row) <= tags_index:
                    continue
                tags.update(extract_tags(row[tags_index]))

    output = {rating: sorted(values) for rating, values in sorted(tags_by_rating.items())}

    with (ROOT / "tags.json").open("w", encoding="utf-8") as outfile:
        json.dump(output, outfile, indent=2, ensure_ascii=False)

    print(f"Wrote tags.json with ratings: {', '.join(output)}")


if __name__ == "__main__":
    main()
