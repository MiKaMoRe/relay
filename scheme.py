from pydantic import BaseModel


class ChatResponse(BaseModel):
    agent_used: str
    response: str
