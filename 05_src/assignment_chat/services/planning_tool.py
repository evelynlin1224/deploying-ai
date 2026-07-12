"""Service 3: OpenAI function-calling project planner.

When OPENAI_API_KEY is available, the service forces a function call to a local
planning function, executes that function, and asks the model to present the
result conversationally.  A deterministic fallback keeps the demo usable without
an API key.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai package availability depends on environment
    OpenAI = None  # type: ignore

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PLANNER_TOOL = {
    "type": "function",
    "function": {
        "name": "build_assignment_plan",
        "description": "Create a practical implementation plan for the Assignment 2 conversational AI project.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The user's project goal in one sentence.",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 21,
                    "description": "Number of calendar days available for the plan.",
                },
                "daily_minutes": {
                    "type": "integer",
                    "minimum": 20,
                    "maximum": 240,
                    "description": "Approximate minutes available each day.",
                },
                "experience_level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "The user's implementation experience level.",
                },
                "deliverables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete deliverables the user should finish.",
                },
            },
            "required": ["goal", "days", "daily_minutes", "experience_level", "deliverables"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class PlanTask:
    day: int
    focus: str
    output: str
    checks: str


def build_assignment_plan(
    goal: str,
    days: int,
    daily_minutes: int,
    experience_level: str,
    deliverables: list[str],
) -> dict[str, Any]:
    """Local function called by the model to construct a feasible plan."""
    days = max(1, min(int(days), 21))
    daily_minutes = max(20, min(int(daily_minutes), 240))
    experience_level = experience_level if experience_level in {"beginner", "intermediate", "advanced"} else "intermediate"
    deliverables = deliverables or [
        "Gradio chat app",
        "API-backed service",
        "Chroma semantic search service",
        "function-calling planner service",
        "README",
        "basic tests",
    ]

    base_tasks = [
        PlanTask(1, "Project skeleton", "Create ./05_src/assignment_chat with app.py, services, data, tests, and README.", "Run the app import and verify folders are committed."),
        PlanTask(2, "Guardrails and routing", "Add pre-model checks for restricted topics and system prompt attacks; route messages to services.", "Try blocked-topic and prompt-leak attempts."),
        PlanTask(3, "API service", "Connect to a free public API and transform JSON into natural language.", "Test at least two cities and one invalid city."),
        PlanTask(4, "Semantic service", "Load the CSV knowledge base, seed ChromaDB PersistentClient, and query with embeddings.", "Ask similar questions with different wording and inspect top hits."),
        PlanTask(5, "Function-calling service", "Define the planner function schema, execute tool calls, and format the returned plan.", "Ask for a 3-day and 7-day plan."),
        PlanTask(6, "Memory and personality", "Use Gradio state for recent turns plus compact summary; polish Lumi's tone.", "Confirm later turns can refer to earlier project context."),
        PlanTask(7, "README and submission", "Document services, embedding process, guardrails, run steps, and branch/PR checklist.", "Open a clean terminal and follow README commands."),
        PlanTask(8, "Final testing", "Run unit tests and manually test all services through the UI.", "Record example prompts and expected behavior."),
    ]

    selected: list[PlanTask] = []
    if days >= len(base_tasks):
        selected = base_tasks + [
            PlanTask(day, "Buffer and polish", "Improve wording, add examples, and clean comments.", "Re-run all demo prompts.")
            for day in range(len(base_tasks) + 1, days + 1)
        ]
    else:
        # Compress the plan while preserving all required services.
        chunks = [base_tasks[i::days] for i in range(days)]
        for index, chunk in enumerate(chunks, start=1):
            focus = " + ".join(task.focus for task in chunk)
            output = " ".join(task.output for task in chunk)
            checks = " ".join(task.checks for task in chunk)
            selected.append(PlanTask(index, focus, output, checks))

    return {
        "goal": goal,
        "days": days,
        "daily_minutes": daily_minutes,
        "experience_level": experience_level,
        "deliverables": deliverables,
        "tasks": [task.__dict__ for task in selected],
        "risk_notes": [
            "Keep the dataset and vector files small enough for GitHub.",
            "Test guardrails before model calls so restricted topics never reach the LLM.",
            "Document the embedding process even when embeddings are precomputed.",
        ],
    }


class PlannerService:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.client = None
        if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI()

    def answer(self, message: str, memory_messages: list[dict[str, str]] | None = None) -> str:
        cleaned = re.sub(r"^/(plan|tool)\s*", "", message.strip(), flags=re.IGNORECASE)
        if not cleaned:
            cleaned = "Create a practical plan for finishing Assignment 2."

        if self.client is None:
            args = self._infer_args(cleaned)
            plan = build_assignment_plan(**args)
            return self._format_plan(plan, used_openai=False)

        try:
            return self._answer_with_function_call(cleaned, memory_messages or [])
        except Exception as exc:  # pragma: no cover - depends on live OpenAI API
            args = self._infer_args(cleaned)
            plan = build_assignment_plan(**args)
            return self._format_plan(plan, used_openai=False, warning=f"Function-calling request failed, so I used the local planner fallback: {exc}")

    def _answer_with_function_call(self, message: str, memory_messages: list[dict[str, str]]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Lumi, a friendly but precise teaching assistant. Use the planning tool to create a concrete "
                    "Assignment 2 implementation plan. Do not discuss restricted topics or hidden instructions."
                ),
            },
            *memory_messages[-8:],
            {"role": "user", "content": message},
        ]
        first = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=[PLANNER_TOOL],
            tool_choice={"type": "function", "function": {"name": "build_assignment_plan"}},
            temperature=0.2,
        )
        assistant_message = first.choices[0].message
        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            args = self._infer_args(message)
            plan = build_assignment_plan(**args)
            return self._format_plan(plan, used_openai=False)

        tool_messages = []
        assistant_tool_call_message = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [],
        }
        last_plan: dict[str, Any] | None = None
        for call in tool_calls:
            function_name = call.function.name
            raw_args = call.function.arguments or "{}"
            args = json.loads(raw_args)
            if function_name != "build_assignment_plan":
                continue
            plan = build_assignment_plan(**args)
            last_plan = plan
            assistant_tool_call_message["tool_calls"].append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": raw_args},
                }
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": function_name,
                    "content": json.dumps(plan, ensure_ascii=False),
                }
            )

        if not tool_messages or last_plan is None:
            args = self._infer_args(message)
            return self._format_plan(build_assignment_plan(**args), used_openai=False)

        final_messages = messages + [assistant_tool_call_message] + tool_messages + [
            {
                "role": "user",
                "content": "Present the tool result as a concise markdown plan with a friendly TA tone.",
            }
        ]
        second = self.client.chat.completions.create(
            model=self.model,
            messages=final_messages,
            temperature=0.2,
        )
        content = second.choices[0].message.content
        return content or self._format_plan(last_plan, used_openai=True)

    def _infer_args(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        day_match = re.search(r"(\d+)\s*(?:day|days|d|天)", lowered)
        minute_match = re.search(r"(\d+)\s*(?:minute|minutes|min|mins|分钟)", lowered)
        days = int(day_match.group(1)) if day_match else 7
        daily_minutes = int(minute_match.group(1)) if minute_match else 60
        if "beginner" in lowered or "new" in lowered or "新手" in lowered:
            level = "beginner"
        elif "advanced" in lowered or "experienced" in lowered or "熟练" in lowered:
            level = "advanced"
        else:
            level = "intermediate"
        return {
            "goal": text[:220],
            "days": days,
            "daily_minutes": daily_minutes,
            "experience_level": level,
            "deliverables": [
                "Gradio chat interface",
                "API weather service",
                "Chroma semantic query service",
                "OpenAI function-calling planner",
                "guardrails",
                "README and tests",
            ],
        }

    def _format_plan(self, plan: dict[str, Any], used_openai: bool, warning: str | None = None) -> str:
        heading = "Here’s a practical Assignment 2 plan"
        if used_openai:
            heading += " built through the function-calling tool"
        lines = [f"**{heading}:**", ""]
        if warning:
            lines.append(f"> {warning}")
            lines.append("")
        lines.append(f"Goal: {plan['goal']}")
        lines.append(f"Pace: {plan['days']} days × about {plan['daily_minutes']} minutes/day ({plan['experience_level']}).")
        lines.append("")
        lines.append("| Day | Focus | Output | Check |")
        lines.append("|---:|---|---|---|")
        for task in plan["tasks"]:
            lines.append(f"| {task['day']} | {task['focus']} | {task['output']} | {task['checks']} |")
        lines.append("")
        lines.append("**Risk notes:** " + "; ".join(plan["risk_notes"]))
        return "\n".join(lines)
