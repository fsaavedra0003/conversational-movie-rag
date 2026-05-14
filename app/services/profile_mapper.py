import json
from pathlib import Path

from app.config import settings


def load_profile_id_map() -> dict[str, str]:
    path = Path(settings.data_dir) / "user_ids.json"

    with open(path, "r", encoding="utf-8") as file:
        user_ids = json.load(file)

    return {str(index): user_id for index, user_id in enumerate(user_ids)}


def map_profile_id(profile_id: str | None) -> str | None:
    if not profile_id:
        return None

    profile_map = load_profile_id_map()
    return profile_map.get(str(profile_id).strip())