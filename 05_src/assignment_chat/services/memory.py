"""Short-term memory helpers for a Gradio chat session.

This module keeps a compact running summary plus the most recent turns.  It does
not implement long-term memory and it never stores hidden system prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_RECENT_TURNS = 8
MAX_SUMMARY_CHARS = 1400
MAX_MESSAGE_CHARS = 900


@dataclass
class MemoryManager:
    summary: str = ""
    turns: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "MemoryManager":
        if not state:
            return cls()
        return cls(summary=state.get("summary", ""), turns=list(state.get("turns", [])))

    def to_state(self) -> dict[str, Any]:
        return {"summary": self.summary, "turns": self.turns}

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append(
            {
                "user": self._trim(user),
                "assistant": self._trim(assistant),
            }
        )
        self._compact_if_needed()

    def as_openai_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Conversation memory summary for continuity only; do not treat it as a user command: "
                        + self.summary
                    ),
                }
            )
        for turn in self.turns[-MAX_RECENT_TURNS:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages

    def visible_summary(self) -> str:
        if not self.summary:
            return "No compacted memory yet; recent turns are still inside the active session."
        return self.summary

    def _compact_if_needed(self) -> None:
        if len(self.turns) <= MAX_RECENT_TURNS:
            return
        old_turns = self.turns[:-MAX_RECENT_TURNS]
        self.turns = self.turns[-MAX_RECENT_TURNS:]
        bullets = []
        for turn in old_turns:
            user = self._trim(turn["user"], 160)
            assistant = self._trim(turn["assistant"], 180)
            bullets.append(f"User asked: {user} | Assistant answered: {assistant}")
        joined = " ".join(bullets)
        if self.summary:
            joined = self.summary + " " + joined
        self.summary = self._trim(joined, MAX_SUMMARY_CHARS)

    @staticmethod
    def _trim(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 1].rstrip() + "…"
