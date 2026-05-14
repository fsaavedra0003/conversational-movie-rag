from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class RecommendRequest(BaseModel):
    question: str
    history: list[Message] = Field(default_factory=list)
    user_id: str | None = None
    mode: Literal["rag", "agent"] = "rag"