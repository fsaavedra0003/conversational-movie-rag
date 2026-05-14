# this avoid bad recomedations
def remove_seen_or_disliked(
    candidates: list[str],
    seen: set[str],
    disliked: set[str],
) -> list[str]:
    blocked = seen | disliked
    return [movie for movie in candidates if movie not in blocked]


def format_history(history: list[dict]) -> str:
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in history
    )