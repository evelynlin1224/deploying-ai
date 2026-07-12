"""Gradio entry point for Assignment 2: conversational AI system.

Run from this folder:
    python app.py
"""
from __future__ import annotations

from typing import Any

import gradio as gr

from services.api_weather import WeatherService
from services.guardrails import check_guardrails
from services.llm import GeneralLLMService
from services.memory import MemoryManager
from services.planning_tool import PlannerService
from services.router import Route, route_message
from services.semantic_search import KnowledgeBaseService

APP_TITLE = "Lumi Lab Companion"
APP_DESCRIPTION = "A small conversational AI assignment app with API calls, semantic search, function calling, guardrails, and session memory."

weather_service = WeatherService()
knowledge_service = KnowledgeBaseService()
planner_service = PlannerService()
general_service = GeneralLLMService()


def respond(user_message: str, chat_history: list[tuple[str, str]] | None, memory_state: dict[str, Any] | None):
    """Handle one Gradio chat turn."""
    chat_history = chat_history or []
    memory = MemoryManager.from_state(memory_state)
    user_message = (user_message or "").strip()

    if not user_message:
        return "", chat_history, memory.to_state()

    guardrail = check_guardrails(user_message)
    if guardrail.blocked:
        bot_message = guardrail.response
    else:
        route = route_message(user_message)
        memory_messages = memory.as_openai_messages()
        if route == Route.API_WEATHER:
            bot_message = weather_service.answer(user_message)
        elif route == Route.SEMANTIC_KB:
            bot_message = knowledge_service.answer(user_message)
        elif route == Route.PLANNER_TOOL:
            bot_message = planner_service.answer(user_message, memory_messages=memory_messages)
        else:
            bot_message = general_service.answer(user_message, memory_messages=memory_messages)

    memory.add_turn(user_message, bot_message)
    chat_history.append((user_message, bot_message))
    return "", chat_history, memory.to_state()


def clear_chat():
    return [], MemoryManager().to_state(), ""


def show_memory(memory_state: dict[str, Any] | None) -> str:
    memory = MemoryManager.from_state(memory_state)
    return memory.visible_summary()


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            f"{APP_DESCRIPTION}\n\n"
            "**Persona:** Lumi is a friendly campus-lab TA: concise, curious, and practical. "
            "Use `/weather`, `/kb`, and `/plan` to call the three assignment services directly."
        )
        memory_state = gr.State(MemoryManager().to_state())
        chatbot = gr.Chatbot(label="Chat with Lumi", height=520)
        user_box = gr.Textbox(
            label="Message",
            placeholder="Try: /kb What guardrails are implemented?",
            lines=2,
        )
        with gr.Row():
            send_button = gr.Button("Send", variant="primary")
            clear_button = gr.Button("Clear")
            memory_button = gr.Button("Show compacted memory")
        memory_view = gr.Markdown("No compacted memory yet.")

        gr.Examples(
            examples=[
                "/weather Toronto",
                "/kb How does the Chroma semantic search service work?",
                "/kb What do I need in the README?",
                "/plan Make a 5 day plan, 60 minutes per day, for finishing Assignment 2",
                "What can you help me test before submission?",
            ],
            inputs=user_box,
        )

        send_button.click(respond, inputs=[user_box, chatbot, memory_state], outputs=[user_box, chatbot, memory_state])
        user_box.submit(respond, inputs=[user_box, chatbot, memory_state], outputs=[user_box, chatbot, memory_state])
        clear_button.click(clear_chat, outputs=[chatbot, memory_state, memory_view])
        memory_button.click(show_memory, inputs=memory_state, outputs=memory_view)

    return demo


if __name__ == "__main__":
    build_demo().launch()
