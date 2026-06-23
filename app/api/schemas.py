from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    messages: list[ChatMessage]


class HealthResponse(BaseModel):
    status: str
    postgres: bool
    qdrant: bool
