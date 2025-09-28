from __future__ import annotations

import difflib
import re
import time
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

from .schema import DiffSummary, SnapshotIn, SnapshotOut

BroadcastFn = Callable[[str, Dict[str, Any]], Awaitable[None]]
LoggerFn = Callable[[str, Dict[str, Any]], None]

class _DiffMetrics:
    __slots__ = ("added", "removed", "lines_changed", "hunks", "funcs_touched")

    def __init__(self, added: int, removed: int, hunks: int, funcs_touched: Iterable[str]) -> None:
        self.added = added
        self.removed = removed
        self.lines_changed = added + removed
        self.hunks = hunks
        self.funcs_touched = sorted(set(funcs_touched))


class SnapshotService:
    """Computes incremental code snapshots and emits diff summaries."""

    def __init__(
        self,
        logger_append_fn: LoggerFn,
        ws_broadcast_fn: Optional[BroadcastFn] = None,
    ) -> None:
        self._append = logger_append_fn
        self._broadcast = ws_broadcast_fn
        self._last: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def diff(self, old: str, new: str) -> Tuple[_DiffMetrics, str]:
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile="previous",
                tofile="current",
                lineterm="",
            )
        )
        added = removed = hunks = 0
        changed_line_numbers: set[int] = set()
        new_line_no = 0

        for line in diff_lines:
            if line.startswith("@@"):
                hunks += 1
                header = line.split(" ")
                for part in header:
                    if part.startswith("+"):
                        start = part[1:].split(",", 1)[0]
                        try:
                            new_line_no = int(start) - 1
                        except ValueError:
                            new_line_no = 0
                        break
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("+"):
                added += 1
                new_line_no += 1
                changed_line_numbers.add(new_line_no)
                continue
            if line.startswith("-"):
                removed += 1
                changed_line_numbers.add(new_line_no + 1)
                continue
            if line.startswith(" "):
                new_line_no += 1

        funcs_touched = self._functions_touched(new, changed_line_numbers)
        preview = "\n".join(diff_lines)
        return _DiffMetrics(added, removed, hunks, funcs_touched), preview

    async def snapshot(self, snap: SnapshotIn) -> SnapshotOut:
        key = (snap.session_id, snap.problem_id, snap.language)
        previous = self._last.get(key)
        version = (previous["version"] + 1) if previous else 1
        prev_source = previous["source"] if previous else ""
        metrics, preview = self.diff(prev_source, snap.source)
        timestamp = time.time()
        lines_total = (snap.source.count("\n") + 1) if snap.source else 0

        diff_summary = DiffSummary(
            version=version,
            timestamp=timestamp,
            lines_total=lines_total,
            lines_changed=metrics.lines_changed,
            added=metrics.added,
            removed=metrics.removed,
            hunks=metrics.hunks,
            funcs_touched=metrics.funcs_touched,
        )

        out = SnapshotOut(
            session_id=snap.session_id,
            problem_id=snap.problem_id,
            language=snap.language,
            version=version,
            diff_summary=diff_summary,
            preview_diff=preview[:4000] if preview else None,
        )

        self._last[key] = {"version": version, "source": snap.source}

        event = {
            "ts": timestamp,
            "type": "diff.snapshot",
            "problem_id": snap.problem_id,
            "language": snap.language,
            "version": version,
            "reason": snap.reason,
            "cursor_line": snap.cursor_line,
            "summary": diff_summary.dict(),
            "preview_diff": out.preview_diff,
        }
        self._append(snap.session_id, event)
        if self._broadcast is not None:
            await self._broadcast(
                snap.session_id,
                {
                    "type": "diff.snapshot",
                    "session_id": snap.session_id,
                    "payload": out.dict(),
                },
            )
        return out

    def _functions_touched(self, new_source: str, changed_lines: Iterable[int]) -> List[str]:
        if not changed_lines:
            return []
        lines = new_source.splitlines()
        if not lines:
            return []
        func_by_line: Dict[int, str] = {}
        for idx, content in enumerate(lines, start=1):
            match = re.match(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", content)
            if match:
                func_by_line[idx] = match.group(1)
        if not func_by_line:
            return []
        touched: set[str] = set()
        for line_no in changed_lines:
            for offset in range(line_no - 5, line_no + 6):
                name = func_by_line.get(offset)
                if name:
                    touched.add(name)
        return sorted(touched)
