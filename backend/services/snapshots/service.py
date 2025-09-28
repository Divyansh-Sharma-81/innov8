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
        *,
        analyzer_enabled: bool = True,
        hint_window_seconds: int = 25,
        store_full_source: bool = False,
    ) -> None:
        self._append = logger_append_fn
        self._broadcast = ws_broadcast_fn
        self._last: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._pending_closeness: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        self._recent_hints: Dict[str, List[Dict[str, Any]]] = {}
        self._analyzer_enabled = analyzer_enabled
        self._hint_window_seconds = hint_window_seconds
        self._store_full_source = store_full_source

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

    def record_hint(self, session_id: str, text: str, level: str, ts: Optional[float] = None) -> None:
        """Store recent hints for hint-followed detection."""

        ts = ts or time.time()
        keywords = self._extract_hint_keywords(text)
        bucket = self._recent_hints.setdefault(session_id, [])
        bucket.append({
            "ts": ts,
            "text": text,
            "keywords": keywords,
            "level": level,
        })
        self._recent_hints[session_id] = [
            item
            for item in bucket
            if (ts - item["ts"]) <= max(self._hint_window_seconds * 2, self._hint_window_seconds + 5)
        ]

    def set_pending_closeness(
        self,
        session_id: str,
        problem_id: str,
        language: str,
        closeness: Dict[str, float],
    ) -> None:
        key = (session_id, problem_id, language)
        self._pending_closeness[key] = dict(closeness)

    def analyze_source(self, source: str) -> Dict[str, str]:
        if not self._analyzer_enabled:
            return {
                "time": "unknown",
                "space": "unknown",
                "detected_algorithm": "unknown",
            }
        return self._analyze_complexity(source)

    async def snapshot(self, snap: SnapshotIn) -> SnapshotOut:
        key = (snap.session_id, snap.problem_id, snap.language)
        previous = self._last.get(key)
        version = (previous["version"] + 1) if previous else 1
        prev_source = previous["source"] if previous else ""
        metrics, preview = self.diff(prev_source, snap.source)
        timestamp = time.time()
        lines_total = (snap.source.count("\n") + 1) if snap.source else 0
        complexity = self._analyze_complexity(snap.source) if self._analyzer_enabled else None
        closeness = self._pending_closeness.pop(key, None)
        hint_followed = self._detect_hint_followed(
            snap.session_id,
            timestamp,
            (complexity or {}).get("detected_algorithm"),
            preview,
        )

        diff_summary = DiffSummary(
            version=version,
            timestamp=timestamp,
            lines_total=lines_total,
            lines_changed=metrics.lines_changed,
            added=metrics.added,
            removed=metrics.removed,
            hunks=metrics.hunks,
            funcs_touched=metrics.funcs_touched,
            complexity_analysis=complexity,
            closeness=closeness,
            hint_followed=hint_followed,
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
            "summary": diff_summary.dict(exclude_none=True),
            "preview_diff": out.preview_diff,
        }
        if self._store_full_source:
            event["source"] = snap.source
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

    # ------------------------------------------------------------------
    # Complexity & hint helpers
    # ------------------------------------------------------------------

    def _analyze_complexity(self, source: str) -> Dict[str, str]:
        base = {
            "time": "unknown",
            "space": "unknown",
            "detected_algorithm": "unknown",
        }
        if not source.strip():
            return base

        if self._detect_nested_loops(source):
            detected = "nested_loops"
        elif self._detect_sorting(source):
            detected = "sorting"
        elif self._detect_hash_usage(source):
            detected = "hash_set"
        elif self._detect_two_pointers(source):
            detected = "two_pointers"
        elif self._detect_stack_usage(source):
            detected = "stack"
        else:
            detected = "unknown"

        complexity_map = {
            "nested_loops": ("O(n^2)", "O(1)"),
            "sorting": ("O(n log n)", "O(1)"),
            "hash_set": ("O(n)", "O(n)"),
            "two_pointers": ("O(n)", "O(1)"),
            "stack": ("O(n)", "O(n)"),
            "unknown": ("unknown", "unknown"),
        }
        time_c, space_c = complexity_map.get(detected, ("unknown", "unknown"))
        base["time"] = time_c
        base["space"] = space_c
        base["detected_algorithm"] = detected
        return base

    def _detect_hint_followed(
        self,
        session_id: str,
        snapshot_ts: float,
        detected_algorithm: Optional[str],
        preview_diff: Optional[str],
    ) -> Optional[bool]:
        bucket = self._recent_hints.get(session_id)
        if not bucket:
            return None
        window = self._hint_window_seconds
        current: List[Dict[str, Any]] = [
            item for item in bucket if (snapshot_ts - item["ts"]) <= window
        ]
        if not current:
            self._recent_hints.pop(session_id, None)
            return None
        self._recent_hints[session_id] = current
        diff_text = (preview_diff or "").lower()
        for item in reversed(current):
            keywords = item.get("keywords", [])
            if detected_algorithm and detected_algorithm in keywords:
                return True
            if any(keyword in diff_text for keyword in keywords if len(keyword) >= 3):
                return True
        return False

    def _extract_hint_keywords(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        keywords: set[str] = set()
        keyword_map = {
            "hash_set": ["hash", "set", "dictionary", "dict", "map"],
            "sorting": ["sort", "sorted", "order", "ordering"],
            "two_pointers": ["two pointers", "left", "right", "pointer"],
            "stack": ["stack", "push", "pop"],
        }
        for algo, tokens in keyword_map.items():
            if any(token in lowered for token in tokens):
                keywords.add(algo)
                keywords.update(token.strip() for token in tokens)
        words = re.findall(r"[a-zA-Z]{3,}", lowered)
        for word in words[:12]:
            keywords.add(word)
        return sorted(keywords)

    def _detect_nested_loops(self, source: str) -> bool:
        loop_stack: List[int] = []
        for line in source.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            while loop_stack and indent <= loop_stack[-1]:
                loop_stack.pop()
            if stripped.startswith(("for ", "while ")):
                if loop_stack:
                    return True
                loop_stack.append(indent)
        return False

    def _detect_sorting(self, source: str) -> bool:
        lowered = source.lower()
        return ".sort(" in lowered or "sorted(" in lowered or "sort(" in lowered

    def _detect_hash_usage(self, source: str) -> bool:
        lowered = source.lower()
        if re.search(r"\b(set|dict|defaultdict|counter)\s*\(", lowered):
            return True
        if re.search(r"\{[^}]*:\s*[^}]*\}", lowered):
            return True
        return False

    def _detect_two_pointers(self, source: str) -> bool:
        lowered = source.lower()
        if "two pointers" in lowered:
            return True
        patterns = [
            r"while\s+left\s*[<!=]",
            r"while\s+right\s*[>!=]",
            r"left\s*\+=\s*1",
            r"right\s*-=\s*1",
            r"right\s*=\s*len",
            r"i\s*<\s*j",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _detect_stack_usage(self, source: str) -> bool:
        lowered = source.lower()
        if "stack = []" in lowered or "stack=[]" in lowered:
            return True
        if re.search(r"stack\.append\(", lowered) and re.search(r"stack\.pop\(", lowered):
            return True
        return False
