from collections.abc import AsyncGenerator

from langchain_openai import ChatOpenAI

from app.config import settings

 # Centralized LLM configuration used by RAG and agent flows.
def get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
        streaming=streaming,
    )

# Stream tokens so the UI can display the response progressively.
async def stream_llm_response(prompt: str) -> AsyncGenerator[str, None]:
    llm = get_llm(streaming=True)

    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield chunk.content