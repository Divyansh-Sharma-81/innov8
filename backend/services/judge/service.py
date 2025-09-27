from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from typing import Iterable, List, Optional
from urllib.parse import urlparse

import httpx

from backend.services.problemset.schema import IOSampleDTO

from .schema import ExecCaseResult, ExecResult


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _normalize_output(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.replace("\r\n", "\n").rstrip()


class JudgeService:
    """Adapter for executing Python submissions via Judge0 or a local mock."""

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        language_id: Optional[int] = None,
        use_mock: Optional[bool] = None,
    ) -> None:
        self.base_url = (host or _env("JUDGE0_HOST") or "https://judge0-ce.p.rapidapi.com").rstrip("/")
        self.api_key = api_key or _env("JUDGE0_KEY")
        self.language_id = language_id or int(_env("JUDGE0_LANG_PY", "71"))
        self.use_mock = bool(use_mock if use_mock is not None else _env("USE_JUDGE0_MOCK", "true").lower() == "true")
        self._client: Optional[httpx.AsyncClient] = None
        parsed = urlparse(self.base_url)
        self._host_header = parsed.netloc

    async def start(self) -> None:
        if self.use_mock or self._client is not None:
            return
        headers = {
            "Content-Type": "application/json",
            "X-RapidAPI-Host": self._host_header,
        }
        if self.api_key:
            headers["X-RapidAPI-Key"] = self.api_key
        timeout = httpx.Timeout(25.0, connect=10.0)
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def run_python_cases(self, source: str, cases: Iterable[IOSampleDTO]) -> ExecResult:
        tests = list(cases)
        if not tests:
            return ExecResult(status="success", passed=0, total=0, cases=[], mock=self.use_mock)
        if self.use_mock:
            return await asyncio.to_thread(self._run_mock, source, tests)
        if self._client is None:
            await self.start()
        assert self._client is not None
        results: List[ExecCaseResult] = []
        passed = 0
        aggregate_stderr: List[str] = []
        overall_status = "success"
        for sample in tests:
            payload = {
                "language_id": self.language_id,
                "source_code": source,
                "stdin": sample.stdin,
                "expected_output": None,
            }
            try:
                response = await self._client.post(
                    "/submissions",
                    params={"base64_encoded": "false", "wait": "true"},
                    json=payload,
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:  # pragma: no cover - integration path
                overall_status = "runtime_error"
                results.append(
                    ExecCaseResult(
                        stdin=sample.stdin,
                        expected=sample.expected_stdout,
                        stdout="",
                        passed=False,
                        stderr=str(exc),
                    )
                )
                aggregate_stderr.append(str(exc))
                continue

            data = response.json()
            stdout = _normalize_output(data.get("stdout"))
            stderr = data.get("stderr")
            status_info = data.get("status", {})
            status_id = status_info.get("id", 0)
            status_desc = status_info.get("description", "")
            case_passed = status_id == 3 and stdout == _normalize_output(sample.expected_stdout)
            if case_passed:
                passed += 1
            else:
                if status_id in {5, 6}:
                    overall_status = "compile_error"
                elif overall_status == "success":
                    overall_status = "runtime_error"
            results.append(
                ExecCaseResult(
                    stdin=sample.stdin,
                    expected=sample.expected_stdout,
                    stdout=stdout,
                    passed=case_passed,
                    time_ms=float(data.get("time", 0) or 0) * 1000.0 if data.get("time") else None,
                    memory_kb=int(data.get("memory", 0)) if data.get("memory") else None,
                    stderr=stderr or status_desc,
                )
            )
            if stderr:
                aggregate_stderr.append(stderr)
        if passed == len(results):
            overall_status = "success"
        return ExecResult(
            status=overall_status,
            passed=passed,
            total=len(results),
            cases=results,
            stderr="\n".join(aggregate_stderr) or None,
            mock=False,
        )

    # Mock executor -----------------------------------------------------
    def _run_mock(self, source: str, tests: List[IOSampleDTO]) -> ExecResult:
        results: List[ExecCaseResult] = []
        passed = 0
        stderr_messages: List[str] = []
        for sample in tests:
            stdout, stderr, exit_code = self._execute_python(source, sample.stdin)
            normalized_stdout = _normalize_output(stdout)
            expected = _normalize_output(sample.expected_stdout)
            ok = exit_code == 0 and normalized_stdout == expected
            if ok:
                passed += 1
            else:
                if stderr:
                    stderr_messages.append(stderr)
            results.append(
                ExecCaseResult(
                    stdin=sample.stdin,
                    expected=sample.expected_stdout,
                    stdout=stdout,
                    passed=ok,
                    stderr=stderr if stderr else None,
                )
            )
        status = "success" if passed == len(results) else "runtime_error"
        return ExecResult(
            status=status,
            passed=passed,
            total=len(results),
            cases=results,
            stderr="\n".join(stderr_messages) or None,
            mock=True,
        )

    def _execute_python(self, source: str, stdin: str, timeout: float = 5.0) -> tuple[str, str, int]:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        try:
            proc = subprocess.run(
                ["python3", tmp_path],
                input=stdin.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")
            stderr = proc.stderr.decode("utf-8", errors="replace")
            return stdout, stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timeout path
            return "", f"Timeout after {timeout}s", 1
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
