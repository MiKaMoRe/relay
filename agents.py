import asyncio
import json
from typing import Any

import ollama
from fastapi import HTTPException

import config

# import obsidian

# MODEL = "qwen3:8b"
# "qwen3-abliterated:8b"
# VISION_MODEL = "qwen3.5"  # модель с поддержкой vision


def run_text_agent(category: str, task: str, history: list[dict]) -> str:
    """Одиночный вызов текстового агента (code / research / creative / general)."""
    cfg = config.AGENT_CONFIGS[category]
    resp = ollama.chat(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": cfg["system_prompt"]},
            *history,
            {"role": "user", "content": task},
        ],
    )
    return resp["message"]["content"]


def run_vision_agent(task: str, image_b64: str | None, history: list[dict]) -> str:
    """
    history — предыдущие сообщения диалога в формате
    [{"role": "user"/"assistant", "content": str, "images": [base64,...]?}, ...],
    благодаря чему модель помнит ранее показанные картинки и свои ответы
    даже в последующих чисто текстовых вопросах.
    """
    cfg = config.AGENT_CONFIGS[config.VISION_CATEGORY]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": cfg["system_prompt"]}
    ]
    messages.extend(history)

    user_msg: dict[str, Any] = {"role": "user", "content": task}
    if image_b64:
        user_msg["images"] = [image_b64]
    messages.append(user_msg)

    resp = ollama.chat(model=cfg["model"], messages=messages)
    return resp["message"]["content"]


def mcp_tools_to_ollama_format(mcp_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


def classify_category(user_prompt: str) -> str:
    """
    Быстрая модель-роутер определяет категорию запроса.
    При любой неудаче (модель ответила не JSON-ом, назвала неизвестную
    категорию и т.д.) — тихо откатываемся на DEFAULT_CATEGORY, чтобы
    пользователь в любом случае получил ответ.
    """
    try:
        response = ollama.chat(
            model=config.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": config.ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        data = json.loads(response["message"]["content"])
        category = str(data.get("category", "")).strip()
    except Exception:
        category = ""

    if category not in config.ROUTABLE_CATEGORIES:
        category = config.DEFAULT_CATEGORY
    return category


def trim_history(history: list[dict]) -> list[dict]:
    """Оставляет только последние MAX_HISTORY_MESSAGES сообщений, чтобы контекст не рос бесконечно."""
    if len(history) <= config.MAX_HISTORY_MESSAGES:
        return history
    return history[-config.MAX_HISTORY_MESSAGES :]


# async def run_notes_agent(task: str, session: ClientSession, hitory: list[dict], max_steps: int = 5) -> str:
#     """
#     В отличие от остальных текстовых агентов, notes может вызывать
#     MCP-инструменты Obsidian несколько раз подряд (найти -> прочитать ->
#     обновить), пока не даст финальный текстовый ответ без новых вызовов.
#     """
#     cfg = AGENT_CONFIGS["notes"]
#     tools_response = await session.list_tools()
#     ollama_tools = mcp_tools_to_ollama_format(tools_response.tools)
#
#     messages: list[dict[str, Any]] = [
#         {"role": "system", "content": cfg["system_prompt"]},
#         *history,
#         {"role": "user", "content": task},
#     ]
#
#     for _ in range(max_steps):
#         response = await asyncio.to_thread(
#             ollama.chat, model=cfg["model"], messages=messages, tools=ollama_tools
#         )
#         message = response["message"]
#
#         if not message.get("tool_calls"):
#             return message["content"]
#
#         messages.append(message)
#         for call in message["tool_calls"]:
#             tool_name = call["function"]["name"]
#             tool_args = call["function"]["arguments"]
#             result = await session.call_tool(tool_name, arguments=tool_args)
#             result_text = result.content[0].text if result.content else ""
#             messages.append({"role": "tool", "content": result_text})
#
#     return (
#         f"Не удалось завершить задачу за отведённое число шагов ({max_steps}). "
#         "Попробуйте сформулировать запрос точнее или разбить его на части."
#     )


async def dispatch(category: str, task: str, history: list[dict]) -> str:
    """Единая точка вызова любого текстового (не-vision) агента по категории."""
    # if category == "notes":
    #     return await run_notes_agent(task, app.state.notes_session, history)
    if category in config.AGENT_CONFIGS:
        return await asyncio.to_thread(run_text_agent, category, task, history)
    raise HTTPException(status_code=500, detail=f"Неизвестная категория: {category}")
