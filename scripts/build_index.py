import json
import shutil
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

FINAL_DATA_PATH = DATA_DIR / "final_data.jsonl"
ITEM_MAP_PATH = DATA_DIR / "item_map.json"
CONVERSATION_PATH = DATA_DIR / "Conversation.txt"

COLLECTION_NAME = "movies"
BATCH_SIZE = 50
MAX_DOCS = 1000
EMBEDDING_MODEL = "text-embedding-3-small"

tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def load_item_map() -> dict:
    with open(ITEM_MAP_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_conversations() -> str:
    with open(CONVERSATION_PATH, "r", encoding="utf-8") as file:
        return file.read()


def get_conversation_by_id(content: str, conversation_id: int) -> str:
    blocks = content.strip().split("\n\n")
    current_id = None
    conversation = []

    for block in blocks:
        if block.strip().isdigit():
            if current_id == conversation_id:
                return "\n\n".join(conversation)

            current_id = int(block.strip())
            conversation = []
        else:
            conversation.append(block)

    if current_id == conversation_id:
        return "\n\n".join(conversation)

    return ""


def build_documents() -> list[Document]:
    item_map = load_item_map()
    conversations = load_conversations()
    docs = []
    total_tokens = 0

    with open(FINAL_DATA_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if len(docs) >= MAX_DOCS:
                break

            user_data = json.loads(line)
            user_id, info = next(iter(user_data.items()))

            for conv_group in info.get("Conversation", []):
                for _, conv in conv_group.items():
                    conversation_id = conv.get("conversation_id")
                    rec_items = conv.get("rec_item", [])
                    likes = conv.get("user_likes", [])
                    dislikes = conv.get("user_dislikes", [])

                    dialogue = get_conversation_by_id(
                        conversations,
                        int(conversation_id),
                    )

                    recommended_movies = [
                        item_map.get(item_id, item_id)
                        for item_id in rec_items
                    ]

                    liked_movies = [
                        item_map.get(item_id, item_id)
                        for item_id in likes
                    ]

                    disliked_movies = [
                        item_map.get(item_id, item_id)
                        for item_id in dislikes
                    ]

                    text = f"""
User ID: {user_id}

Dialogue:
{dialogue}

Liked movies:
{", ".join(liked_movies)}

Disliked movies:
{", ".join(disliked_movies)}

Recommended movies:
{", ".join(recommended_movies)}
""".strip()

                    token_count = count_tokens(text)
                    total_tokens += token_count

                    print(
                        f"Conversation {conversation_id} | "
                        f"Tokens: {token_count} | "
                        f"Total tokens: {total_tokens}"
                    )

                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "user_id": user_id,
                                "conversation_id": str(conversation_id),
                                "recommended_movies": ", ".join(
                                    recommended_movies
                                ),
                                "token_count": token_count,
                            },
                        )
                    )

                    if len(docs) >= MAX_DOCS:
                        break

    print(f"Total documents: {len(docs)}")
    print(f"Estimated total tokens: {total_tokens}")

    return docs


def main() -> None:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    docs = build_documents()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    indexed_tokens = 0

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        batch_tokens = sum(
            int(doc.metadata.get("token_count", 0))
            for doc in batch
        )

        vectorstore.add_documents(batch)

        indexed_tokens += batch_tokens

        print(
            f"Indexed {min(i + BATCH_SIZE, len(docs))}/{len(docs)} "
            f"| Batch tokens: {batch_tokens} "
            f"| Indexed tokens: {indexed_tokens}"
        )

    print("ChromaDB index created successfully")


if __name__ == "__main__":
    main()