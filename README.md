# Conversational Movie Recommender System (CRS)

LLM-based conversational movie recommender system built using:

- FastAPI
- Streamlit
- OpenAI
- LangChain
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- Agent-based recommendation flow

The system uses the Movie subset of the LLM-Redial dataset to generate personalized conversational movie recommendations.

---

# Features

## Implemented CRS Approaches

### 1. RAG-based Conversational Recommender

The RAG pipeline retrieves relevant historical movie conversations and user preferences from ChromaDB and uses them as grounding context for the LLM.

Features:
- Retrieval-Augmented Generation
- Personalized retrieval using user profiles
- Conversational memory
- Streaming responses
- Context-grounded recommendations
- Hallucination reduction constraints

---

### 2. Agent-based Conversational Recommender

The agent-based system extends the RAG pipeline by:
- detecting user intent
- generating a recommendation with RAG
- extracting the recommended movie
- calling an external movie metadata API (IMDbOT)
- enriching the final recommendation with metadata

Features:
- Tool/API usage
- Intent classification
- Metadata enrichment
- Conversational recommendation generation

---

# Project Architecture

```text
Streamlit UI
    ↓
FastAPI /recommend/stream
    ↓
RAG or Agent Mode
    ↓
ChromaDB Retrieval
    ↓
OpenAI LLM Generation
    ↓
Streaming Response
```

---

# Tech Stack

- Python 3.10+
- FastAPI
- Streamlit
- LangChain
- OpenAI API
- ChromaDB
- OpenAI Embeddings
- IMDbOT API
- Pytest

---

# Dataset

Dataset used:
- LLM-Redial (Movie category)

The indexing pipeline uses:
- `final_data.jsonl`
- `Conversation.txt`
- `item_map.json`
- `user_ids.json`

---

# Retrieval Pipeline

## Indexing

The system builds vector documents containing:
- dialogue history
- liked movies
- disliked movies
- recommended movies
- user_id metadata

Each conversation is embedded using:
- `text-embedding-3-small`

Stored in:
- ChromaDB vector database

---

## Retrieval

At inference time:
1. user question + chat history are combined
2. similar CRS conversations are retrieved
3. retrieved context is injected into the LLM prompt
4. the LLM generates a grounded recommendation

Returning users are retrieved using:

```python
filter={"user_id": user_id}
```

---

# Prompt Engineering Improvements

## Prompt Change 1: Retrieval-grounded recommendation constraint

Initial problem:
The LLM could recommend plausible movie titles even when they were not supported by the retrieved LLM-Redial context. This increased hallucination risk and made the recommender less faithful to the dataset.

Change added in `build_rag_prompt()`:

```text
- Use the retrieved context as your main evidence.
- Prefer a movie that appears in the retrieved context.
- Do not invent unrelated movie titles.
```

Why this improves accuracy:
This makes the model ground its recommendation in retrieved historical CRS conversations instead of relying only on its general movie knowledge. It improves faithfulness, reduces hallucinated titles, and makes the output more aligned with the RAG pipeline.

Expected effect:
The recommendation is more likely to come from the retrieved user/dialogue context and therefore better reflects the dataset evidence.

---

## Prompt Change 2: Personalization and negative-preference constraint

Initial problem:
The LLM could recommend movies that the user had already watched, disliked, or rejected in previous conversations. This creates poor personalization and inconsistent recommendations.

Change added in `build_rag_prompt()`:

```text
- Do not recommend movies the user already watched.
- Do not recommend movies the user disliked or rejected.
- Use this user's retrieved memory as the main source for personalization.
```

Why this improves accuracy:
This forces the model to use retrieved user memory as a personalization signal rather than generic conversational context. It improves recommendation consistency and user alignment.

Expected effect:
The recommender avoids repeated or disliked movies and produces recommendations more aligned with the user's historical preferences.

---

# API Endpoints

## Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

## Streaming Recommendation Endpoint

```http
POST /recommend/stream
```

Request body:

```json
{
  "question": "Recommend a sci-fi movie",
  "history": [],
  "user_id": "12",
  "mode": "rag"
}
```

Modes:
- `rag`
- `agent`

---

# Running the Project

## 1. Clone Repository

```bash
git clone <repo_url>
cd seez-crs
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create `.env`

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

---

## 5. Build ChromaDB Index

```bash
python app/scripts/build_index.py
```

---

## 6. Run FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Backend:
```text
http://127.0.0.1:8000
```

---

## 7. Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

# Example User Flows

## New User

1. User selects new user flow
2. User asks for movie recommendation
3. System retrieves similar CRS conversations
4. LLM generates grounded recommendation

---

## Returning User

1. User enters profile ID
2. System maps profile to dataset user
3. Personalized conversations are retrieved
4. Recommendations are generated using user memory

---

# Agent Flow

The agent system performs:

1. Intent classification
2. RAG recommendation generation
3. Movie title extraction
4. External IMDbOT API call
5. Final recommendation generation with metadata

Returned metadata:
- title
- year
- actors
- IMDb URL
- poster URL

---

# Evaluation Notes

The system was evaluated manually using multiple conversational recommendation scenarios.

Key evaluation goals:
- grounded recommendations
- hallucination reduction
- conversational consistency
- personalization quality
- avoidance of disliked/watched movies

---

# Current Limitations

- Retrieval currently uses similarity search only
- No reranking stage
- No automatic quantitative evaluation framework
- Limited intent classification
- Agent system is intentionally lightweight
- Metadata API quality depends on IMDbOT availability

---

# Future Improvements

Possible future improvements:
- hybrid retrieval
- reranking models
- structured user preference memory
- query rewriting
- evaluation with RAGAS
- async vectorstore caching
- advanced agent orchestration
- LangSmith tracing
- multi-agent recommendation workflows

---

# Tests

Run tests:

```bash
pytest
```

Current tests:
- health endpoint
- filtering utility

---

# Author

Francisco Saavedra
