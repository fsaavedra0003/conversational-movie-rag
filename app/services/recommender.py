from collections.abc import AsyncGenerator

from app.schemas import RecommendRequest
from app.services.agent import agent_recommend
from app.services.profile_mapper import map_profile_id
from app.services.rag import rag_recommend


def history_to_text(request: RecommendRequest) -> str:
    return "\n".join(
        f"{message.role}: {message.content}"
        for message in request.history
    )


async def recommend(request: RecommendRequest) -> AsyncGenerator[str, None]:
    history = history_to_text(request)

    mapped_user_id = map_profile_id(request.user_id)

    if request.user_id and not mapped_user_id:
        yield "Profile ID not found. You can continue as a new user or try another profile ID."
        return

    if request.mode == "agent":
        async for token in agent_recommend(
            question=request.question,
            history=history,
            user_id=mapped_user_id,
        ):
            yield token
        return

    async for token in rag_recommend(
        question=request.question,
        history=history,
        user_id=mapped_user_id,
    ):
        yield token