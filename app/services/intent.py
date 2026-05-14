# Lightweight rule-based intent detection for routing agent behavior.

def classify_intent(question: str) -> str:
    text = question.lower()

    if any(
        word in text
        for word in [
            "available",
            "netflix",
            "prime",
            "disney",
            "streaming",
            "where can i watch",
            "watch it on",
        ]
    ):
        return "recommend_with_availability"

    if any(
        word in text
        for word in [
            "details",
            "metadata",
            "actors",
            "actor",
            "year",
            "imdb",
            "poster",
            "info",
            "information",
        ]
    ):
        return "recommend_with_metadata"

    if any(
        word in text
        for word in [
            "tell me about",
            "synopsis",
            "plot",
            "explain",
            "what is",
            "describe",
        ]
    ):
        return "explain_movie"

    return "recommend_movie"