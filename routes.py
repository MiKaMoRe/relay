import asyncio
import base64
import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import agents
import config
import scheme

app = FastAPI(title="Multi-Agent API", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/agents")
def list_agents():
    """Список доступных категорий/агентов и их описаний (для отладки)."""
    return {name: cfg["description"] for name, cfg in config.AGENT_CONFIGS.items()}


@app.post("/chat", response_model=scheme.ChatResponse)
async def chat(
    message: Annotated[str, Form()] = "",
    image: Annotated[
        UploadFile | None,
        File(description="Опционально: изображение для vision-агента"),
    ] = None,
    history: Annotated[
        str, Form(description="JSON-список предыдущих сообщений диалога")
    ] = "[]",
):
    """
    Единая точка входа для чата: текст, изображение или и то, и другое.
    Диспетчер сам решает, какой агент и какая модель нужны — фронтенду
    не нужно самому выбирать между "обычным чатом" и "vision".
    """
    if not message.strip() and image is None:
        raise HTTPException(status_code=400, detail="Пустое сообщение без изображения")

    try:
        history_list = json.loads(history) if history else []
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Некорректный формат history (ожидается JSON-массив)",
        )

    history_list = agents.trim_history(history_list)

    image_b64 = None
    if image is not None:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Загруженный файл не является изображением"
            )
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Файл пустой")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    if image_b64:
        category = config.VISION_CATEGORY
        result = await asyncio.to_thread(
            agents.run_vision_agent, message, image_b64, history_list
        )
    else:
        category = await asyncio.to_thread(agents.classify_category, message)
        result = await agents.dispatch(category, message, history_list)

    return scheme.ChatResponse(agent_used=category, response=result)


@app.get("/health")
def health():
    return {"status": "ok"}
