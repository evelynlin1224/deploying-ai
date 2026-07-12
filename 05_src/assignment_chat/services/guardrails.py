"""Input guardrails for the Assignment 2 chat client.

The guardrails are deliberately enforced before any model call or service call.
This keeps restricted topics and prompt-injection attempts away from the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class GuardrailResult:
    blocked: bool
    reason: str = ""
    response: str = ""


_RESTRICTED_TOPIC_PATTERNS: list[tuple[str, str]] = [
    (r"\b(cat|cats|kitten|kittens|feline|felines|meow)\b|猫|小猫|喵", "cats"),
    (r"\b(dog|dogs|puppy|puppies|canine|canines|bark)\b|狗|小狗|汪", "dogs"),
    (
        r"\b(horoscope|horoscopes|zodiac|astrology|aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b|星座|占星|运势",
        "horoscopes or zodiac signs",
    ),
    (r"taylor\s+swift|泰勒\s*斯威夫特|霉霉", "Taylor Swift"),
]

_PROMPT_ATTACK_PATTERNS: list[tuple[str, str]] = [
    (r"system\s+prompt|developer\s+message|hidden\s+instruction|initial\s+instruction", "system prompt access"),
    (r"reveal|show|print|display|dump|leak|expose|verbatim", "prompt disclosure request"),
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", "prompt override attempt"),
    (r"act\s+as\s+(?:if\s+)?(?:there\s+are\s+)?no\s+rules", "prompt override attempt"),
    (r"change|modify|edit|replace|rewrite|update", "system prompt modification"),
    (r"系统提示|开发者消息|隐藏指令|泄露|透露|忽略之前|修改系统", "prompt attack in Chinese"),
]


def _matches_any(patterns: Iterable[tuple[str, str]], text: str) -> tuple[bool, str]:
    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True, label
    return False, ""


def check_guardrails(user_message: str) -> GuardrailResult:
    """Return a blocking result if the message violates project guardrails."""
    text = user_message.strip()
    if not text:
        return GuardrailResult(blocked=False)

    restricted, label = _matches_any(_RESTRICTED_TOPIC_PATTERNS, text)
    if restricted:
        return GuardrailResult(
            blocked=True,
            reason=f"restricted topic: {label}",
            response=(
                "I can’t help with that topic in this assignment chat. "
                "Try asking me about the project services, Chroma search, the API demo, "
                "or an implementation plan instead."
            ),
        )

    # A prompt attack generally contains both a disclosure/modification intent and a protected target.
    lower = text.lower()
    mentions_prompt_target = bool(
        re.search(
            r"system\s+prompt|developer\s+message|hidden\s+instruction|initial\s+instruction|系统提示|开发者消息|隐藏指令",
            lower,
            flags=re.IGNORECASE,
        )
    )
    asks_to_disclose_or_change = bool(
        re.search(
            r"reveal|show|print|display|dump|leak|expose|verbatim|change|modify|edit|replace|rewrite|update|透露|泄露|展示|显示|打印|修改|替换|改写",
            lower,
            flags=re.IGNORECASE,
        )
    )
    direct_override = bool(
        re.search(
            r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions|忽略之前|无视以上",
            lower,
            flags=re.IGNORECASE,
        )
    )

    if (mentions_prompt_target and asks_to_disclose_or_change) or direct_override:
        return GuardrailResult(
            blocked=True,
            reason="system prompt access or modification attempt",
            response=(
                "I can’t reveal or modify the system instructions. "
                "I can still explain the visible project design, guardrails, services, and code structure."
            ),
        )

    # A broader pattern sweep catches oddly phrased attempts.
    attack, label = _matches_any(_PROMPT_ATTACK_PATTERNS, text)
    if attack and mentions_prompt_target:
        return GuardrailResult(
            blocked=True,
            reason=label,
            response=(
                "I can’t reveal or change hidden instructions. "
                "Ask about the app’s public behavior or implementation details instead."
            ),
        )

    return GuardrailResult(blocked=False)
