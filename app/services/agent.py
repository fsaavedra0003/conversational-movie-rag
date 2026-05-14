from collections.abc import AsyncGenerator

# Intent classifier used to detect what the user wants
from app.services.intent import classify_intent

# LLM helpers:
# - get_llm() creates the model instance
# - stream_llm_response() streams tokens back to the UI
from app.services.llm import get_llm, stream_llm_response

# External movie metadata provider (IMDbOT)
from app.services.movie_api import get_movie_info

# RAG-based recommendation pipeline
from app.services.rag import rag_answer


# Intents that may require extra movie metadata
# such as actors, year, IMDb link, or poster
METADATA_INTENTS = {
    "recommend_with_metadata",
    "recommend_with_availability",
    "explain_movie",
}


async def extract_movie_title(text: str) -> str:
    """
    Uses the LLM to extract the main movie title
    from the generated RAG response.
    """

    # Prompt asking the LLM to only return the movie title
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

    # Create a non-streaming LLM instance
    llm = get_llm(streaming=False)

    # Invoke the LLM asynchronously
    response = await llm.ainvoke(prompt)

    # Clean whitespace from the response
    title = response.content.strip()

    # Fallback if extraction failed
    if not title or title.upper() == "UNKNOWN":
        return "Unknown"

    return title


def has_valid_movie_info(movie_info: dict) -> bool:
    """
    Validates whether the metadata response
    contains useful movie information.
    """

    return any(
        movie_info.get(field)
        and "not found" not in str(movie_info.get(field)).lower()
        for field in ["year", "actors", "imdb_url", "poster"]
    )


def should_call_metadata_tool(intent: str, question: str) -> bool:
    """
    Decides whether the system should call
    the external movie metadata tool.
    """

    # Convert question to lowercase for keyword matching
    text = question.lower()

    # Keywords indicating the user wants movie details
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

    # Return True if:
    # - the detected intent requires metadata
    # - OR the user explicitly asks for details
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
    """
    Builds the final agent prompt used
    to generate the conversational answer.
    """

    metadata_section = ""

    # If valid metadata exists,
    # include it inside the final prompt
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
        # Prevent hallucinated metadata
        metadata_section = """
IMDbOT metadata:
External metadata was not available or incomplete.
Use only the RAG recommendation and do not invent missing metadata.
"""

    # Final system prompt for the LLM
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
    """
    Simple async generator used to stream plain text.
    """

    yield text


async def agent_recommend(
    question: str,
    history: str,
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Main agent workflow.

    Flow:
    1. Detect intent
    2. Generate RAG recommendation
    3. Decide if metadata is needed
    4. Optionally retrieve external movie metadata
    5. Generate final conversational answer
    """

    # Step 1: classify user intent
    intent = classify_intent(question)

    # Step 2: generate recommendation using RAG
    rag_response = await rag_answer(
        question=question,
        history=history,
        user_id=user_id,
    )

    # Step 3: determine whether metadata lookup is needed
    needs_metadata = should_call_metadata_tool(intent, question)

    # If metadata is not needed,
    # stream the raw RAG response directly
    if not needs_metadata:
        async for token in stream_text(rag_response):
            yield token
        return

    # Step 4: extract movie title from the RAG response
    movie_title = await extract_movie_title(rag_response)

    # Fetch external metadata for the extracted movie
    movie_info = await get_movie_info(movie_title)

    # Validate metadata quality
    metadata_found = has_valid_movie_info(movie_info)

    # Step 5: build the final agent prompt
    prompt = build_agent_final_prompt(
        question=question,
        history=history,
        intent=intent,
        rag_response=rag_response,
        movie_info=movie_info,
        metadata_found=metadata_found,
    )

    # Stream final LLM-generated response token by token
    async for token in stream_llm_response(prompt):
        yield token