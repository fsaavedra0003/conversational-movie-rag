from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    # Chat message used in conversation history
    role: Literal["user", "assistant"]
    content: str


class RecommendRequest(BaseModel):
    # Current user query
    question: str

    # Previous conversation context
    history: list[Message] = Field(default_factory=list)

    # Optional returning user identifier
    user_id: str | None = None

    # Recommendation strategy selector
    mode: Literal["rag", "agent"] = "rag"