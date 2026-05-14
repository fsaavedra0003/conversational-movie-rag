# Chroma vector database used for semantic retrieval
from langchain_community.vectorstores import Chroma

# OpenAI embedding model used to convert text into vectors
from langchain_openai import OpenAIEmbeddings

# Application configuration/settings
from app.config import settings

# LLM utilities:
# - get_llm() creates the language model instance
# - stream_llm_response() streams responses token by token
from app.services.llm import get_llm, stream_llm_response


# Name of the ChromaDB collection
COLLECTION_NAME = "movies"


def load_vectorstore() -> Chroma:
    """
    Loads the Chroma vector database
    with OpenAI embeddings.
    """

    # Create embedding model
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    # Load persistent Chroma vector store
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
    """
    Retrieves the most relevant documents
    from the vector database using semantic search.
    """

    # Load vector database
    vectorstore = load_vectorstore()

    # Combine chat history and latest question
    # into a single retrieval query
    query = f"{history}\n{question}"

    # If user_id exists,
    # perform personalized retrieval
    if user_id:

        docs = vectorstore.similarity_search(
            query,
            k=k,
            filter={"user_id": user_id},
        )

        # Fallback to general retrieval
        # if no personalized docs were found
        if not docs:
            print(f"No docs found for user_id={user_id}. Using general RAG.")

            docs = vectorstore.similarity_search(query, k=k)

    else:
        # General retrieval for new users
        docs = vectorstore.similarity_search(query, k=k)

    # Debug output:
    # print retrieved documents and metadata
    print("\n===== RETRIEVED DOCS =====")

    for index, doc in enumerate(docs, start=1):

        print(f"\nDOC {index}")
        print("Metadata:", doc.metadata)

        # Print first 1000 characters only
        print(doc.page_content[:1000])

        print("-------------------------")

    print("==========================\n")

    # Merge retrieved documents into one context string
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_prompt(
    question: str,
    history: str,
    context: str,
    user_id: str | None = None,
) -> str:
    """
    Builds the final RAG prompt
    sent to the LLM.
    """

    profile_text = ""

    # Add personalization instructions
    # for returning users
    if user_id:
        profile_text = f"""
Returning user profile:
User ID: {user_id}

Use this user's retrieved memory as the main source for personalization.
"""

    # Final RAG system prompt
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
    """
    Generates a complete RAG-based response
    using a non-streaming LLM call.
    """

    # Retrieve relevant context from vector database
    context = retrieve_rag_context(
        question=question,
        history=history,
        user_id=user_id,
    )

    # Build final RAG prompt
    prompt = build_rag_prompt(
        question=question,
        history=history,
        context=context,
        user_id=user_id,
    )

    # Create non-streaming LLM instance
    llm = get_llm(streaming=False)

    # Generate final response
    response = await llm.ainvoke(prompt)

    return response.content


async def rag_recommend(
    question: str,
    history: str,
    user_id: str | None = None,
):
    """
    Streams a RAG-generated recommendation
    token by token.
    """

    # Retrieve relevant retrieval context
    context = retrieve_rag_context(
        question=question,
        history=history,
        user_id=user_id,
    )

    # Build final RAG prompt
    prompt = build_rag_prompt(
        question=question,
        history=history,
        context=context,
        user_id=user_id,
    )

    # Stream generated response tokens
    async for token in stream_llm_response(prompt):
        yield token