from collections.abc import AsyncGenerator

from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
        streaming=streaming,
    )


async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    llm = get_llm(streaming=True)

    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield chunk.content