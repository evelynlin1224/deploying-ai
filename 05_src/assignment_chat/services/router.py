"""Lightweight intent router for the conversational services."""
from __future__ import annotations

import re
from enum import Enum


class Route(str, Enum):
    API_WEATHER = "api_weather"
    SEMANTIC_KB = "semantic_kb"
    PLANNER_TOOL = "planner_tool"
    GENERAL = "general"


WEATHER_TERMS = {
    "weather", "forecast", "temperature", "rain", "wind", "humidity", "climate", "api weather",
    "天气", "气温", "下雨", "风速", "湿度", "预报",
}

KB_TERMS = {
    "assignment", "requirement", "requirements", "rubric", "service", "services", "semantic",
    "chroma", "chromadb", "embedding", "embeddings", "dataset", "gradio", "guardrail", "guardrails",
    "memory", "readme", "github", "branch", "pull request", "pr", "submission", "sqlite", "api call",
    "作业", "要求", "语义", "检索", "知识库", "向量", "记忆", "提交", "分支", "接口", "服务",
}

PLAN_TERMS = {
    "plan", "schedule", "roadmap", "timeline", "break down", "steps", "todo", "milestone",
    "规划", "计划", "日程", "步骤", "待办", "路线图", "安排",
}


def _contains_any(message: str, terms: set[str]) -> bool:
    lowered = message.lower()
    return any(term in lowered for term in terms)


def route_message(message: str) -> Route:
    """Choose the service that should handle a user message."""
    msg = message.strip()
    lowered = msg.lower()

    if lowered.startswith(("/weather", "/api")):
        return Route.API_WEATHER
    if lowered.startswith(("/kb", "/search", "/semantic")):
        return Route.SEMANTIC_KB
    if lowered.startswith(("/plan", "/tool")):
        return Route.PLANNER_TOOL

    # Prefer weather over knowledge-base routing when the user clearly asks for live conditions.
    if _contains_any(lowered, WEATHER_TERMS):
        return Route.API_WEATHER

    # Planning queries often mention assignment terms; route them to the function tool first.
    if _contains_any(lowered, PLAN_TERMS) and re.search(r"\b(day|days|week|weeks|hour|hours|minute|minutes|project|assignment|build|implement)\b|天|周|小时|分钟|作业|项目|实现", lowered):
        return Route.PLANNER_TOOL

    if _contains_any(lowered, KB_TERMS):
        return Route.SEMANTIC_KB

    return Route.GENERAL
