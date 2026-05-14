from collections.abc import AsyncGenerator

# Request schema containing:
# - question
# - chat history
# - user_id
# - recommendation mode
from app.schemas import RecommendRequest

# Agent-based recommendation pipeline
from app.services.agent import agent_recommend

# Maps frontend profile IDs to dataset user IDs
from app.services.profile_mapper import map_profile_id

# Standard RAG recommendation pipeline
from app.services.rag import rag_recommend


def history_to_text(request: RecommendRequest) -> str:
    """
    Converts structured chat history into
    a plain text conversation format.
    """

    # Example output:
    # user: I like horror movies
    # assistant: Try The Conjuring
    return "\n".join(
        f"{message.role}: {message.content}"
        for message in request.history
    )


async def recommend(request: RecommendRequest) -> AsyncGenerator[str, None]:
    """
    Main recommendation entry point.

    Flow:
    1. Convert chat history to text
    2. Map profile ID
    3. Validate user profile
    4. Select recommendation mode
    5. Stream recommendation response
    """

    # Convert structured history into text
    history = history_to_text(request)

    # Map frontend profile ID
    # to internal dataset user ID
    mapped_user_id = map_profile_id(request.user_id)

    # Handle invalid profile IDs
    if request.user_id and not mapped_user_id:

        yield (
            "Profile ID not found. "
            "You can continue as a new user "
            "or try another profile ID."
        )

        return

    # Use agent-based recommendation system
    if request.mode == "agent":

        async for token in agent_recommend(
            question=request.question,
            history=history,
            user_id=mapped_user_id,
        ):
            yield token

        return

    # Default:
    # use standard RAG recommendation pipeline
    async for token in rag_recommend(
        question=request.question,
        history=history,
        user_id=mapped_user_id,
    ):
        yield token