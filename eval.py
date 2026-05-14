import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

from app.services.agent import agent_recommend
from app.services.llm import get_llm
from app.services.rag import retrieve_rag_context, rag_answer


# File where evaluation results will be saved
RESULTS_PATH = Path("data/evaluation_results.json")


# Fixed test cases used to evaluate the recommender
TEST_CASES = [
    {
        "name": "general_sci_fi",
        "question": "Recommend me a sci-fi movie.",
        "user_id": None,
        "mode": "rag",
    },
    {
        "name": "general_comedy",
        "question": "I want a funny movie.",
        "user_id": None,
        "mode": "rag",
    },
    {
        "name": "dark_thriller",
        "question": "Recommend a dark thriller.",
        "user_id": None,
        "mode": "rag",
    },
    {
        "name": "romantic_movie",
        "question": "Suggest a romantic movie.",
        "user_id": None,
        "mode": "rag",
    },
    {
        "name": "family_movie",
        "question": "I want something family friendly.",
        "user_id": None,
        "mode": "rag",
    },
    {
        "name": "returning_user_1",
        "question": "Recommend something based on my taste.",
        "user_id": "0",
        "mode": "rag",
    },
    {
        "name": "returning_user_2",
        "question": "What movie should I watch next?",
        "user_id": "1",
        "mode": "rag",
    },
    {
        "name": "agent_metadata",
        "question": "Recommend me one good action movie with details.",
        "user_id": None,
        "mode": "agent",
    },
    {
        "name": "explain_movie",
        "question": "Tell me about The Matrix.",
        "user_id": None,
        "mode": "rag",
        "expected_title": "The Matrix",
    },
    {
        "name": "avoid_streaming_claims",
        "question": "Recommend something available on Netflix.",
        "user_id": None,
        "mode": "agent",
    },
]


def normalize_text(text: str) -> str:
    # Lowercase text and remove extra whitespace
    return re.sub(r"\s+", " ", text.lower()).strip()


async def extract_recommended_movie(answer: str) -> str:
    # Uses the LLM to extract the main movie title from an answer
    prompt = f"""
Extract the main movie title recommended or discussed in this answer.

Rules:
- Return only the movie title.
- No explanation.
- No quotes.
- If no clear movie title exists, return UNKNOWN.

Answer:
{answer}
"""

    llm = get_llm(streaming=False)
    response = await llm.ainvoke(prompt)

    title = response.content.strip()

    if not title or title.upper() == "UNKNOWN":
        return "UNKNOWN"

    return title


def movie_appears_in_context(movie_title: str, context: str) -> bool:
    # Checks whether the extracted movie title appears in retrieved RAG context
    if movie_title == "UNKNOWN":
        return False

    return normalize_text(movie_title) in normalize_text(context)


def extract_disliked_movies(context: str) -> set[str]:
    # Extracts disliked movies from retrieved context
    disliked = set()

    for line in context.splitlines():
        if line.lower().startswith("disliked movies:"):
            movies = line.split(":", 1)[1].strip()

            if movies:
                disliked.update(
                    movie.strip()
                    for movie in movies.split(",")
                    if movie.strip()
                )

    return disliked


def extract_liked_movies(context: str) -> set[str]:
    # Extracts liked movies from retrieved context
    liked = set()

    for line in context.splitlines():
        if line.lower().startswith("liked movies:"):
            movies = line.split(":", 1)[1].strip()

            if movies:
                liked.update(
                    movie.strip()
                    for movie in movies.split(",")
                    if movie.strip()
                )

    return liked


def extract_recommended_movies_from_context(context: str) -> set[str]:
    # Extracts previously recommended movies from retrieved context
    recommended = set()

    for line in context.splitlines():
        if line.lower().startswith("recommended movies:"):
            movies = line.split(":", 1)[1].strip()

            if movies:
                recommended.update(
                    movie.strip()
                    for movie in movies.split(",")
                    if movie.strip()
                )

    return recommended


def contains_streaming_claim(answer: str) -> bool:
    # Detects unsupported streaming availability claims
    text = normalize_text(answer)

    risky_phrases = [
        "available on netflix",
        "on netflix",
        "available on prime",
        "on prime",
        "available on disney",
        "on disney",
        "streaming on",
        "watch it on",
    ]

    return any(phrase in text for phrase in risky_phrases)


def has_agent_metadata(answer: str) -> bool:
    # Checks whether agent response includes expected metadata fields
    text = normalize_text(answer)

    return (
        "imdb" in text
        and ("year" in text or re.search(r"\b(19|20)\d{2}\b", text))
        and ("actor" in text or "actors" in text)
    )


def check_output(
    answer: str,
    context: str,
    movie_title: str,
    mode: str,
    expected_title: str | None = None,
) -> dict:
    # Runs automatic checks against the generated answer
    disliked_movies = extract_disliked_movies(context)
    liked_movies = extract_liked_movies(context)
    context_recommended_movies = extract_recommended_movies_from_context(context)

    normalized_movie = normalize_text(movie_title)

    # Check if system recommended something the user disliked
    recommended_disliked_movie = any(
        normalize_text(movie) == normalized_movie
        for movie in disliked_movies
    )

    # Check if system repeated something already liked or seen
    recommended_seen_or_liked_movie = any(
        normalize_text(movie) == normalized_movie
        for movie in liked_movies
    )

    expected_title_match = True

    # For explicit movie questions, verify expected title appears
    if expected_title:
        expected_title_match = (
            normalize_text(expected_title) in normalize_text(answer)
            or normalize_text(expected_title) == normalized_movie
        )

    grounded_in_context = movie_appears_in_context(movie_title, context)

    appears_in_recommended_context = any(
        normalize_text(movie) == normalized_movie
        for movie in context_recommended_movies
    )

    # Boolean evaluation checks
    checks = {
        "non_empty": bool(answer.strip()),
        "movie_extracted": movie_title != "UNKNOWN",
        "grounded_in_retrieved_context": grounded_in_context,
        "appears_in_context_recommendations": appears_in_recommended_context,
        "does_not_recommend_disliked_movie": not recommended_disliked_movie,
        "does_not_repeat_liked_movie": not recommended_seen_or_liked_movie,
        "no_streaming_claim": not contains_streaming_claim(answer),
        "has_agent_metadata_if_agent": (
            mode != "agent" or has_agent_metadata(answer)
        ),
        "expected_title_match": expected_title_match,
    }

    # Convert passed checks into a score
    score = sum(checks.values())
    max_score = len(checks)

    return {
        "score": score,
        "max_score": max_score,
        "score_percent": round((score / max_score) * 100, 2),
        "checks": checks,
        "movie_title": movie_title,
        "disliked_movies_in_context": sorted(disliked_movies),
        "liked_movies_in_context": sorted(liked_movies),
        "recommended_movies_in_context": sorted(context_recommended_movies),
    }


async def run_agent_case(question: str, user_id: str | None) -> str:
    # Runs one test case using the agent pipeline
    chunks = []

    async for token in agent_recommend(
        question=question,
        history="",
        user_id=user_id,
    ):
        chunks.append(token)

    return "".join(chunks)


async def run_rag_case(question: str, user_id: str | None) -> str:
    # Runs one test case using the RAG pipeline
    return await rag_answer(
        question=question,
        history="",
        user_id=user_id,
    )


async def run_case(case: dict) -> dict:
    # Runs one full evaluation case
    question = case["question"]
    user_id = case["user_id"]
    mode = case["mode"]

    # Retrieve context used by the recommender
    context = retrieve_rag_context(
        question=question,
        history="",
        user_id=user_id,
    )

    # Generate answer using selected mode
    if mode == "agent":
        answer = await run_agent_case(question, user_id)
    else:
        answer = await run_rag_case(question, user_id)

    # Extract recommended movie title
    movie_title = await extract_recommended_movie(answer)

    # Evaluate generated answer
    evaluation = check_output(
        answer=answer,
        context=context,
        movie_title=movie_title,
        mode=mode,
        expected_title=case.get("expected_title"),
    )

    # Passing threshold
    passed = evaluation["score_percent"] >= 75

    return {
        "name": case["name"],
        "question": question,
        "user_id": user_id,
        "mode": mode,
        "passed": passed,
        "answer": answer,
        "retrieved_context_preview": context[:1500],
        **evaluation,
    }


async def run_eval() -> None:
    # Runs all evaluation cases and saves a report
    results = []

    for case in TEST_CASES:
        print("\n" + "=" * 80)
        print(f"Running test case: {case['name']}")

        result = await run_case(case)
        results.append(result)

        # Print case summary
        print(f"Mode: {result['mode']}")
        print(f"Movie extracted: {result['movie_title']}")
        print(f"Score: {result['score']}/{result['max_score']}")
        print(f"Score percent: {result['score_percent']}%")
        print(f"Passed: {result['passed']}")

        print("\nAnswer:")
        print(result["answer"])

        print("\nChecks:")
        for check_name, passed in result["checks"].items():
            status = "PASS" if passed else "FAIL"
            print(f"- {check_name}: {status}")

    # Aggregate evaluation results
    total = len(results)
    passed_count = sum(1 for result in results if result["passed"])
    average_score = sum(result["score_percent"] for result in results) / total

    report = {
        "created_at": datetime.utcnow().isoformat(),
        "total_cases": total,
        "passed_cases": passed_count,
        "pass_rate": round((passed_count / total) * 100, 2),
        "average_score_percent": round(average_score, 2),
        "results": results,
    }

    # Save JSON report
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"Passed: {passed_count}/{total}")
    print(f"Pass rate: {report['pass_rate']}%")
    print(f"Average score: {report['average_score_percent']}%")
    print(f"Saved report to: {RESULTS_PATH}")


# Script entry point
if __name__ == "__main__":
    asyncio.run(run_eval())