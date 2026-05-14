from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.services.llm import get_llm, stream_llm_response


COLLECTION_NAME = "movies"


def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    return Chroma(
        persist_directory=settings.vectorstore_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve_rag_context(
    question: str,
    history: str,
    user_id: str | None = None,
    k: int = 5,
) -> str:
    vectorstore = load_vectorstore()
    query = f"{history}\n{question}"

    if user_id:
        docs = vectorstore.similarity_search(
            query,
            k=k,
            filter={"user_id": user_id},
        )

        if not docs:
            print(f"No docs found for user_id={user_id}. Using general RAG.")
            docs = vectorstore.similarity_search(query, k=k)
    else:
        docs = vectorstore.similarity_search(query, k=k)

    print("\n===== RETRIEVED DOCS =====")
    for index, doc in enumerate(docs, start=1):
        print(f"\nDOC {index}")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:1000])
        print("-------------------------")
    print("==========================\n")

    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_prompt(
    question: str,
    history: str,
    context: str,
    user_id: str | None = None,
) -> str:
    profile_text = ""

    if user_id:
        profile_text = f"""
Returning user profile:
User ID: {user_id}

Use this user's retrieved memory as the main source for personalization.
"""

    return f"""
You are a conversational movie recommender system.

Important rules:
- Continue the conversation naturally.
- Use the retrieved context as your main evidence.
- Recommend one movie only.
- Prefer a movie that appears in the retrieved context.
- Do not invent unrelated movie titles.
- Do not recommend movies the user already watched.
- Do not recommend movies the user disliked or rejected.
- If the user asks about a movie, explain that movie instead of recommending a new one.
- Mention briefly why the movie matches the user's taste.
- Keep the answer concise and friendly.

{profile_text}

Retrieved context:
{context}

Current chat history:
{history}

Latest user message:
{question}

Assistant:
"""


async def rag_answer(
    question: str,
    history: str,
    user_id: str | None = None,
) -> str:
    context = retrieve_rag_context(
        question=question,
        history=history,
        user_id=user_id,
    )

    prompt = build_rag_prompt(
        question=question,
        history=history,
        context=context,
        user_id=user_id,
    )

    llm = get_llm(streaming=False)
    response = await llm.ainvoke(prompt)

    return response.content


async def rag_recommend(
    question: str,
    history: str,
    user_id: str | None = None,
):
    context = retrieve_rag_context(
        question=question,
        history=history,
        user_id=user_id,
    )

    prompt = build_rag_prompt(
        question=question,
        history=history,
        context=context,
        user_id=user_id,
    )

    async for token in stream_llm_response(prompt):
        yield token