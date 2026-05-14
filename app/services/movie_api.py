import httpx

# Application settings/configuration
from app.config import settings


def empty_movie_info(title: str) -> dict:
    """
    Returns a fallback movie metadata structure
    when no valid metadata is found.
    """

    return {
        "title": title,
        "year": "Year not found.",
        "actors": "Actors not found.",
        "imdb_url": "IMDb URL not found.",
        "poster": "Poster URL not found.",
    }


async def safe_get(url: str, params: dict) -> dict | list | None:
    """
    Safely performs an async HTTP GET request.

    Handles:
    - network errors
    - invalid responses
    - empty responses
    - non-JSON responses
    """

    try:
        # Create async HTTP client with timeout
        async with httpx.AsyncClient(timeout=10) as client:

            # Send GET request
            response = await client.get(url, params=params)

        # Debug logs for troubleshooting API responses
        print("IMDbOT URL:", str(response.url))
        print("IMDbOT status:", response.status_code)
        print("IMDbOT content-type:", response.headers.get("content-type"))
        print("IMDbOT raw response:", response.text[:500])

        # Raise exception for non-200 responses
        response.raise_for_status()

        # Handle empty responses
        if not response.text.strip():
            print("IMDbOT returned empty response.")
            return None

        try:
            # Convert JSON response into Python object
            return response.json()

        except ValueError:
            # Handle invalid JSON
            print("IMDbOT returned non-JSON response.")
            return None

    except httpx.RequestError as error:
        # Handle connection/network errors
        print(f"IMDbOT request failed: {error}")
        return None

    except httpx.HTTPStatusError as error:
        # Handle HTTP status errors
        print(f"IMDbOT returned bad status: {error}")
        return None


def extract_first_result(data: dict | list | None) -> dict | None:
    """
    Extracts the first movie result from
    different possible IMDbOT response formats.
    """

    # Return None if response is empty
    if not data:
        return None

    # If response is already a list,
    # return the first item
    if isinstance(data, list):
        return data[0] if data else None

    # Try common response keys used by APIs
    for key in ["description", "results", "data", "movies", "titles"]:

        value = data.get(key)

        # Return first valid item in the list
        if isinstance(value, list) and value:
            return value[0]

    # Fallback:
    # return data itself if it is already a dictionary
    return data if isinstance(data, dict) else None


def normalize_movie_info(title: str, data: dict | None) -> dict:
    """
    Normalizes IMDbOT response fields into
    a consistent internal structure.
    """

    # Return fallback metadata if response is empty
    if not data:
        return empty_movie_info(title)

    return {
        # Use fallback title if missing
        "title": data.get("#TITLE") or data.get("title") or title,

        # Convert year to string for consistency
        "year": str(
            data.get("#YEAR")
            or data.get("year")
            or "Year not found."
        ),

        # Actor list or fallback message
        "actors": (
            data.get("#ACTORS")
            or data.get("actors")
            or "Actors not found."
        ),

        # IMDb URL or fallback
        "imdb_url": (
            data.get("#IMDB_URL")
            or data.get("imdb_url")
            or "IMDb URL not found."
        ),

        # Poster image URL or fallback
        "poster": (
            data.get("#IMG_POSTER")
            or data.get("poster")
            or "Poster URL not found."
        ),
    }


async def get_movie_info(title: str) -> dict:
    """
    Retrieves movie metadata from IMDbOT API.
    """

    # Prevent invalid searches
    if not title or title.lower() == "unknown":
        return empty_movie_info(title)

    # Remove trailing slash from base URL
    base_url = settings.imdbot_base_url.rstrip("/")

    # Perform movie search request
    search_data = await safe_get(
        url=f"{base_url}/search",
        params={"q": title},
    )

    # Extract first matching movie result
    first_result = extract_first_result(search_data)

    # Return fallback metadata if nothing was found
    if not first_result:
        return empty_movie_info(title)

    # Normalize and return movie metadata
    return normalize_movie_info(title, first_result)