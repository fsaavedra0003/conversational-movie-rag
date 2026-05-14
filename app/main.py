from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schemas import RecommendRequest
from app.services.recommender import recommend

app = FastAPI(title="Seez Conversational Movie Recommender")


@app.get("/health")
def health() -> dict:
    # Simple endpoint used for service health checks
    return {"status": "ok"}


@app.post("/recommend/stream")
async def recommend_stream(request: RecommendRequest):
    # Stream responses progressively to improve chat UX
    return StreamingResponse(
        recommend(request),
        media_type="text/plain",
    )