import httpx

from app.config import settings


def empty_movie_info(title: str) -> dict:
    return {
        "title": title,
        "year": "Year not found.",
        "actors": "Actors not found.",
        "imdb_url": "IMDb URL not found.",
        "poster": "Poster URL not found.",
    }


async def safe_get(url: str, params: dict) -> dict | list | None:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, params=params)

        print("IMDbOT URL:", str(response.url))
        print("IMDbOT status:", response.status_code)
        print("IMDbOT content-type:", response.headers.get("content-type"))
        print("IMDbOT raw response:", response.text[:500])

        response.raise_for_status()

        if not response.text.strip():
            print("IMDbOT returned empty response.")
            return None

        try:
            return response.json()
        except ValueError:
            print("IMDbOT returned non-JSON response.")
            return None

    except httpx.RequestError as error:
        print(f"IMDbOT request failed: {error}")
        return None

    except httpx.HTTPStatusError as error:
        print(f"IMDbOT returned bad status: {error}")
        return None


def extract_first_result(data: dict | list | None) -> dict | None:
    if not data:
        return None

    if isinstance(data, list):
        return data[0] if data else None

    for key in ["description", "results", "data", "movies", "titles"]:
        value = data.get(key)
        if isinstance(value, list) and value:
            return value[0]

    return data if isinstance(data, dict) else None


def normalize_movie_info(title: str, data: dict | None) -> dict:
    if not data:
        return empty_movie_info(title)

    return {
        "title": data.get("#TITLE") or data.get("title") or title,
        "year": str(data.get("#YEAR") or data.get("year") or "Year not found."),
        "actors": data.get("#ACTORS") or data.get("actors") or "Actors not found.",
        "imdb_url": data.get("#IMDB_URL") or data.get("imdb_url") or "IMDb URL not found.",
        "poster": data.get("#IMG_POSTER") or data.get("poster") or "Poster URL not found.",
    }


async def get_movie_info(title: str) -> dict:
    if not title or title.lower() == "unknown":
        return empty_movie_info(title)

    base_url = settings.imdbot_base_url.rstrip("/")

    search_data = await safe_get(
        url=f"{base_url}/search",
        params={"q": title},
    )

    first_result = extract_first_result(search_data)

    if not first_result:
        return empty_movie_info(title)

    return normalize_movie_info(title, first_result)