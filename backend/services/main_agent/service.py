from __future__ import annotations

import textwrap
import time
from typing import Any, Dict, Optional

import httpx
import yaml

from .schema import AgentIn, AgentOut

_ALLOWED_ACTIONS = {
    "stay_silent",
    "send_message",
    "give_hint",
    "present_problem",
    "ask_hr",
    "ask_aptitude",
}

_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the calm primary interviewer guiding a live coding interview. Select exactly one action from:
    stay_silent, send_message, give_hint, present_problem, ask_hr, ask_aptitude.
    Obey speak_gate: if it is "hold", you must stay_silent.
    When speaking, keep responses ≤2 sentences, offer coaching not solutions, and prefer gently guiding the candidate.
    Respond ONLY with YAML using keys: action, text, hint_level (for give_hint), next_rating, desired_tags, update_running_summary.
    hint_level must be one of: nudge, guide, direction.
    Provide concise text appropriate for the chosen action.
    """
)


class MainAgentService:
    """LLM-backed main interviewer agent with mock fallback."""

    def __init__(
        self,
        api_key: Optional[str],
        base_url: str,
        model_id: str,
        use_mock: bool = False,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.use_mock = use_mock
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self.use_mock or self._client is not None:
            return
        timeout = httpx.Timeout(25.0, connect=10.0)
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def decide(self, payload: AgentIn) -> AgentOut:
        if self.use_mock:
            return self._mock(payload)
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured for main agent")
        client = self._client
        if client is None:
            await self.start()
            client = self._client
        assert client is not None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        request_body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(payload)},
            ],
            "temperature": 0.3,
        }
        response = await client.post("/chat/completions", json=request_body, headers=headers)
        response.raise_for_status()
        data = response.json()
        message = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        parsed: Dict[str, Any]
        try:
            parsed = yaml.safe_load(message) or {}
        except yaml.YAMLError:
            parsed = {}
        return self._coerce_output(parsed, payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_user_prompt(self, payload: AgentIn) -> str:
        plan_state = payload.plan_state or {}
        lines = [
            f"session_id: {payload.session_id}",
        ]
        if payload.candidate_name:
            lines.append(f"candidate_name: {payload.candidate_name}")
        lines.extend(
            [
                f"role: {payload.role}",
                f"speak_gate: {payload.speak_gate}",
                f"mini_tags: {', '.join(payload.mini_tags) if payload.mini_tags else 'none'}",
                f"plan_state: {plan_state}",
                "last_exec_summary:",
                payload.last_exec_summary or "(none)",
                "last_diff_summary:",
                payload.last_diff_summary or "(none)",
                "last_agent_summary:",
                payload.last_agent_summary or "(none)",
                "transcript_pretext_60s:",
                payload.transcript_pretext_60s or "(none)",
                "transcript_window_10s:",
                payload.transcript_window_10s or "(none)",
            ]
        )
        return "\n".join(str(part) for part in lines)

    def _coerce_output(self, data: Dict[str, Any], payload: AgentIn) -> AgentOut:
        action = str(data.get("action", "stay_silent")).strip()
        if action not in _ALLOWED_ACTIONS:
            action = "stay_silent"
        text = str(data.get("text", "")) or ""
        hint_level = data.get("hint_level")
        if action != "give_hint":
            hint_level = None
        elif hint_level not in {"nudge", "guide", "direction"}:
            hint_level = "guide"
        next_rating = data.get("next_rating")
        try:
            next_rating_value = int(next_rating) if next_rating is not None else None
        except (TypeError, ValueError):
            next_rating_value = None
        desired_tags = data.get("desired_tags") or None
        if desired_tags is not None:
            desired_tags = [str(tag).strip() for tag in desired_tags if str(tag).strip()]
            if not desired_tags:
                desired_tags = None
        summary_update = data.get("update_running_summary")
        summary_text = str(summary_update) if summary_update else None
        if payload.speak_gate == "hold" and action != "present_problem":
            action = "stay_silent"
            text = ""
            hint_level = None
        return AgentOut(
            action=action,  # type: ignore[arg-type]
            text=text.strip(),
            hint_level=hint_level,
            next_rating=next_rating_value,
            desired_tags=desired_tags,
            update_running_summary=summary_text,
        )

    def _mock(self, payload: AgentIn) -> AgentOut:
        tags = {tag.lower() for tag in payload.mini_tags}
        plan_state = payload.plan_state or {}
        if payload.speak_gate == "hold":
            return AgentOut(action="stay_silent", text="")
        if "stuck" in tags:
            return AgentOut(
                action="give_hint",
                text="Try outlining your approach on paper and focus on the main data structure you need.",
                hint_level="guide",
                update_running_summary="Candidate needed mid-problem guidance.",
            )
        hr_goal = int(plan_state.get("hr_goal", 1))
        hr_done = int(plan_state.get("hr_done", 0))
        if hr_done < hr_goal:
            return AgentOut(
                action="ask_hr",
                text="",
                update_running_summary="Asked an HR-style culture question.",
            )
        aptitude_goal = int(plan_state.get("apt_goal", 1))
        aptitude_done = int(plan_state.get("apt_done", 0))
        if aptitude_done < aptitude_goal:
            return AgentOut(
                action="ask_aptitude",
                text="",
            )
        return AgentOut(
            action="send_message",
            text="How are you feeling about the current approach?",
            update_running_summary="Checked in on candidate confidence.",
        )


__all__ = ["MainAgentService"]
