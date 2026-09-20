import asyncio
import json
from typing import Any

import ollama
from mcp import Client

MCP_URL = "http://127.0.0.1:27200/mcp"
MODEL = "qwen3:8b"


def _mcp_result_to_text(result: Any) -> str:
    parts = []

    for content in result.content:
        if hasattr(content, "text"):
            parts.append(content.text)
        else:
            parts.append(str(content))

    if parts:
        return "\n".join(parts)

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)

    return str(result)


def _mcp_tool_to_ollama(tool: Any) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def _chat_with_vault(user_prompt: str) -> str:
    # Isolated MCP connection.
    async with Client(MCP_URL) as mcp:
        # Get tools from Obsidian MCP.
        tools_result = await mcp.list_tools()

        tools = [_mcp_tool_to_ollama(tool) for tool in tools_result.tools]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an assistant for an Obsidian vault. "
                    "Use the available MCP tools to search and read "
                    "the vault when necessary. "
                    "Do not invent information. "
                    "Base answers about the vault on retrieved notes."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=tools,
            )

            message = response["message"]

            # Keep Qwen's response in the conversation.
            messages.append(message)

            tool_calls = message.get("tool_calls", [])

            # Qwen has finished.
            if not tool_calls:
                return message.get("content", "")

            # Execute requested MCP tools.
            for call in tool_calls:
                function = call["function"]

                name = function["name"]
                arguments = function.get("arguments", {})

                try:
                    result = await mcp.call_tool(
                        name,
                        arguments,
                    )

                    output = _mcp_result_to_text(result)

                except Exception as exc:
                    output = f"MCP tool error: {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "content": output,
                    }
                )


def chat_with_vault(user_prompt: str) -> str:
    """
    Synchronous public API.

    Example:
        answer = chat_with_vault("What are my notes about MCP?")
    """
    return asyncio.run(_chat_with_vault(user_prompt))
