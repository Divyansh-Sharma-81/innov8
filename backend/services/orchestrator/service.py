from __future__ import annotations

import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from backend.services.problemset.schema import MonacoPayload
from backend.services.problemset.service import ProblemsetService

from .schema import AgentAction, Decision, PolicyConfig, TriggerEvent

BroadcastFunc = Callable[[str, MonacoPayload], Awaitable[None]]
LoggerFunc = Callable[[str, Dict[str, Any]], None]


class Orchestrator:
    """Rule-based orchestrator that turns trigger events into agent actions."""

    def __init__(
        self,
        problem_service: ProblemsetService,
        *,
        config: Optional[PolicyConfig] = None,
        broadcast_problem: Optional[BroadcastFunc] = None,
        logger: Optional[LoggerFunc] = None,
    ) -> None:
        self.problem_service = problem_service
        self.config = config or PolicyConfig()
        self._broadcast_problem = broadcast_problem
        self._logger = logger
        self._state: Dict[str, Dict[str, Any]] = {}
        self._clock = time.time

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_event(self, event: TriggerEvent) -> Decision:
        state = self._ensure_state(event.session_id, event.ts)
        state["last_event_ts"] = event.ts
        handlers = {
            "mini.tag": self._handle_mini_tag,
            "exec.result": self._handle_exec_result,
            "ui.editor.save": self._handle_editor_save,
            "idle.tick": self._handle_idle_tick,
        }
        handler = handlers.get(event.kind)
        if handler:
            handler(event, state)
        actions = self._evaluate_rules(event, state)
        snapshot = self._snapshot_state(state)
        self._log_state(event.session_id, snapshot)
        return Decision(actions=actions, updated_state=snapshot)

    async def present_problem(
        self,
        session_id: str,
        *,
        rating: Optional[int] = None,
        tags: Optional[List[str]] = None,
        problem_id: Optional[str] = None,
        broadcast: bool = True,
    ) -> MonacoPayload:
        now = self._clock()
        state = self._ensure_state(session_id, now)
        selected_rating = rating or state.get("current_rating") or self.config.base_rating
        payload = self.problem_service.monaco_payload(
            problem_id=problem_id,
            rating=selected_rating,
            tags=tags,
            limit=1,
        )
        state["current_problem_id"] = payload.problem.id
        state["current_rating"] = payload.problem.rating
        state["consecutive_fail"] = 0
        state["last_action_ts"] = now
        state["last_activity_ts"] = now
        snapshot = self._snapshot_state(state)
        self._log_state(session_id, snapshot)
        if broadcast and self._broadcast_problem:
            await self._broadcast_problem(session_id, payload)
        return payload

    def get_state(self, session_id: str) -> Dict[str, Any]:
        state = self._state.get(session_id)
        if not state:
            return {}
        return self._snapshot_state(state)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _handle_mini_tag(self, event: TriggerEvent, state: Dict[str, Any]) -> None:
        payload = event.payload or {}
        tags = [tag.lower() for tag in payload.get("tags", [])]
        state["latest_tags"] = tags
        state["speak_gate"] = payload.get("speak_gate", state.get("speak_gate", "ok"))
        state["last_mini_ts"] = event.ts
        if "silence_long" in tags:
            state["silence_long_ts"] = event.ts

    def _handle_exec_result(self, event: TriggerEvent, state: Dict[str, Any]) -> None:
        payload = event.payload or {}
        passed = payload.get("passed") or 0
        total = payload.get("total") or 0
        ratio = (passed / total) if total else 0.0
        existing = state.get("verdict_score", 0.5)
        state["verdict_score"] = round(existing * 0.7 + ratio * 0.3, 4)
        if total and passed == total:
            state["consecutive_fail"] = 0
        else:
            state["consecutive_fail"] = state.get("consecutive_fail", 0) + 1
        state["last_activity_ts"] = event.ts
        state["last_result_ts"] = event.ts
        problem_id = payload.get("problem_id")
        if problem_id:
            state["current_problem_id"] = problem_id

    def _handle_editor_save(self, event: TriggerEvent, state: Dict[str, Any]) -> None:
        state["last_activity_ts"] = event.ts
        state["last_save_ts"] = event.ts
        payload = event.payload or {}
        if "version" in payload:
            state["last_save_version"] = payload.get("version")
        if "lines_changed" in payload:
            state["last_save_lines_changed"] = payload.get("lines_changed")

    def _handle_idle_tick(self, event: TriggerEvent, state: Dict[str, Any]) -> None:
        payload = event.payload or {}
        idle_ms = payload.get("idle_ms")
        if idle_ms is None:
            last_activity = state.get("last_activity_ts", event.ts)
            idle_ms = int(max(0.0, (event.ts - last_activity) * 1000))
        state["last_idle_ms"] = int(idle_ms)

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------
    def _evaluate_rules(self, event: TriggerEvent, state: Dict[str, Any]) -> List[AgentAction]:
        now = event.ts
        session_id = event.session_id
        speak_gate = state.get("speak_gate", "ok")
        if speak_gate == "hold":
            return [
                AgentAction(
                    type="stay_silent",
                    session_id=session_id,
                    data={},
                    reason="speaker is holding the floor",
                    cooldown_s=0,
                )
            ]

        # Rule: stuck hint
        if event.kind == "mini.tag":
            tags = set(state.get("latest_tags", []))
            if "stuck" in tags and self._cooldown_passed(now, state.get("last_hint_ts", 0.0), self.config.hint_cooldown_s):
                state["last_hint_ts"] = now
                state["last_action_ts"] = now
                return [
                    AgentAction(
                        type="give_hint",
                        session_id=session_id,
                        data={
                            "text": "Try walking through a small example step-by-step and outline your approach.",
                            "level": "guide",
                        },
                        reason="stuck signal",
                        cooldown_s=self.config.hint_cooldown_s,
                    )
                ]

            if "claims_done" in tags and state.get("verdict_score", 0.5) >= 0.9:
                actions: List[AgentAction] = []
                if self._cooldown_passed(now, state.get("last_message_ts", 0.0), self.config.chat_cooldown_s):
                    state["last_message_ts"] = now
                    state["last_action_ts"] = now
                    actions.append(
                        AgentAction(
                            type="send_message",
                            session_id=session_id,
                            data={"text": "Nice work! Ready for the next one?", "level": "plain"},
                            reason="claims_done high score",
                            cooldown_s=self.config.chat_cooldown_s,
                        )
                    )
                next_rating = self._select_next_rating(state.get("current_rating", self.config.base_rating), state.get("verdict_score", 0.5))
                state["current_rating"] = next_rating
                state["last_action_ts"] = now
                actions.append(
                    AgentAction(
                        type="present_problem",
                        session_id=session_id,
                        data={"rating": next_rating},
                        reason="advance difficulty",
                        cooldown_s=0,
                    )
                )
                return actions

        # Rule: consecutive failures
        if event.kind == "exec.result" and state.get("consecutive_fail", 0) >= 2:
            actions = []
            if self._cooldown_passed(now, state.get("last_message_ts", 0.0), self.config.chat_cooldown_s):
                state["last_message_ts"] = now
                state["last_action_ts"] = now
                actions.append(
                    AgentAction(
                        type="send_message",
                        session_id=session_id,
                        data={
                            "text": "Let's regroup and try an easier variation for a moment.",
                            "level": "plain",
                        },
                        reason="consecutive failures",
                        cooldown_s=self.config.chat_cooldown_s,
                    )
                )
            lowered = max(state.get("current_rating", self.config.base_rating) - self.config.step_down, self.config.min_rating)
            state["current_rating"] = lowered
            state["consecutive_fail"] = 0
            actions.append(
                AgentAction(
                    type="present_problem",
                    session_id=session_id,
                    data={"rating": lowered},
                    reason="adjust difficulty down",
                    cooldown_s=0,
                )
            )
            state["last_action_ts"] = now
            return actions

        # Rule: idle message
        if event.kind == "idle.tick":
            idle_ms = state.get("last_idle_ms", 0)
            if idle_ms >= self.config.idle_threshold_ms and self._cooldown_passed(now, state.get("last_message_ts", 0.0), self.config.chat_cooldown_s):
                state["last_message_ts"] = now
                state["last_action_ts"] = now
                return [
                    AgentAction(
                        type="send_message",
                        session_id=session_id,
                        data={"text": "Still with me? Let me know if you want a hint or a break.", "level": "plain"},
                        reason="idle threshold",
                        cooldown_s=self.config.chat_cooldown_s,
                    )
                ]

        if event.kind == "ui.editor.save":
            lines_changed = int(state.get("last_save_lines_changed", 0) or 0)
            if lines_changed >= 10 and self._cooldown_passed(now, state.get("last_message_ts", 0.0), self.config.chat_cooldown_s):
                state["last_message_ts"] = now
                state["last_action_ts"] = now
                return [
                    AgentAction(
                        type="send_message",
                        session_id=session_id,
                        data={
                            "text": "Great progress — consider running the tests when you feel ready.",
                            "level": "plain",
                        },
                        reason="significant save",
                        cooldown_s=self.config.chat_cooldown_s,
                    )
                ]

        return [
            AgentAction(
                type="stay_silent",
                session_id=session_id,
                data={},
                reason="no rule matched",
                cooldown_s=0,
            )
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_state(self, session_id: str, ts: float) -> Dict[str, Any]:
        if session_id not in self._state:
            self._state[session_id] = {
                "current_rating": self.config.base_rating,
                "last_action_ts": 0.0,
                "last_hint_ts": 0.0,
                "last_message_ts": 0.0,
                "current_problem_id": None,
                "verdict_score": 0.5,
                "consecutive_fail": 0,
                "last_activity_ts": ts,
                "speak_gate": "ok",
                "latest_tags": [],
                "last_idle_ms": 0,
            }
        return self._state[session_id]

    def _snapshot_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        snapshot = dict(state)
        tags = snapshot.get("latest_tags")
        if isinstance(tags, set):
            snapshot["latest_tags"] = sorted(tags)
        return snapshot

    def _log_state(self, session_id: str, snapshot: Dict[str, Any]) -> None:
        if not self._logger:
            return
        self._logger(
            session_id,
            {
                "ts": self._clock(),
                "type": "decision.state",
                "state": snapshot,
            },
        )

    def _cooldown_passed(self, now: float, last_ts: float, cooldown: int) -> bool:
        if cooldown <= 0:
            return True
        return (now - last_ts) >= cooldown

    def _select_next_rating(self, current: int, verdict_score: float) -> int:
        if verdict_score >= 0.9:
            next_rating = current + self.config.step_up
        elif verdict_score >= 0.6:
            next_rating = current + self.config.step_keep
        elif verdict_score >= 0.3:
            next_rating = current
        else:
            next_rating = current - self.config.step_down
        return max(self.config.min_rating, min(self.config.max_rating, next_rating))
