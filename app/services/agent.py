from collections.abc import AsyncGenerator

from app.services.intent import classify_intent
from app.services.llm import get_llm, stream_llm_response
from app.services.movie_api import get_movie_info
from app.services.rag import rag_answer


METADATA_INTENTS = {
    "recommend_with_metadata",
    "recommend_with_availability",
    "explain_movie",
}


async def extract_movie_title(text: str) -> str:
    prompt = f"""
Extract the main recommended or discussed movie title from this answer.

Rules:
- Return only the movie title.
- No explanation.
- No quotes.
- If no clear title exists, return UNKNOWN.

Answer:
{text}
"""

    llm = get_llm(streaming=False)
    response = await llm.ainvoke(prompt)

    title = response.content.strip()

    if not title or title.upper() == "UNKNOWN":
        return "Unknown"

    return title


def has_valid_movie_info(movie_info: dict) -> bool:
    return any(
        movie_info.get(field)
        and "not found" not in str(movie_info.get(field)).lower()
        for field in ["year", "actors", "imdb_url", "poster"]
    )


def should_call_metadata_tool(intent: str, question: str) -> bool:
    text = question.lower()

    metadata_keywords = [
        "details",
        "metadata",
        "actors",
        "actor",
        "year",
        "imdb",
        "poster",
        "info",
        "information",
        "tell me about",
        "explain",
    ]

    return intent in METADATA_INTENTS or any(
        keyword in text for keyword in metadata_keywords
    )


def build_agent_final_prompt(
    question: str,
    history: str,
    intent: str,
    rag_response: str,
    movie_info: dict | None = None,
    metadata_found: bool = False,
) -> str:
    metadata_section = ""

    if movie_info and metadata_found:
        metadata_section = f"""
IMDbOT metadata:
Title: {movie_info["title"]}
Year: {movie_info["year"]}
Actors: {movie_info["actors"]}
IMDb URL: {movie_info["imdb_url"]}
Poster URL: {movie_info["poster"]}
"""
    else:
        metadata_section = """
IMDbOT metadata:
External metadata was not available or incomplete.
Use only the RAG recommendation and do not invent missing metadata.
"""

    return f"""
You are a simple agentic conversational movie recommender.

The agent completed these steps:
1. Detected the user intent.
2. Retrieved a recommendation using RAG.
3. Decided whether external metadata was needed.
4. Used IMDbOT metadata only when available.

User intent:
{intent}

Current chat history:
{history}

User question:
{question}

RAG recommendation:
{rag_response}

{metadata_section}

Final answer rules:
- Recommend or explain one movie only.
- Use the RAG recommendation as the main source of truth.
- Explain briefly why the movie matches the user.
- If metadata is available, include title, year, actors, IMDb URL, and poster URL.
- If metadata is missing, do not mention fake metadata.
- Do not claim streaming availability.
- Do not claim Netflix, Prime, or Disney availability.
- Keep it concise and friendly.

Assistant:
"""


async def stream_text(text: str) -> AsyncGenerator[str, None]:
    yield text


async def agent_recommend(
    question: str,
    history: str,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    intent = classify_intent(question)

    rag_response = await rag_answer(
        question=question,
        history=history,
        user_id=user_id,
    )

    needs_metadata = should_call_metadata_tool(intent, question)

    if not needs_metadata:
        async for token in stream_text(rag_response):
            yield token
        return

    movie_title = await extract_movie_title(rag_response)

    movie_info = await get_movie_info(movie_title)
    metadata_found = has_valid_movie_info(movie_info)

    prompt = build_agent_final_prompt(
        question=question,
        history=history,
        intent=intent,
        rag_response=rag_response,
        movie_info=movie_info,
        metadata_found=metadata_found,
    )

    async for token in stream_llm_response(prompt):
        yield token