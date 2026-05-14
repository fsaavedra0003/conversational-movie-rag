# Conversational Movie Recommender System (CRS)

LLM-based Conversational Recommender System built for the Seez Generative AI Engineering technical assessment.

The project implements multiple LLM-based conversational recommendation approaches on top of the LLM-REDIAL dataset and exposes the system through a FastAPI streaming API with a Streamlit interface.

The implementation focuses on:
- Retrieval-Augmented Generation (RAG)
- Agent-based recommendation orchestration
- Personalized retrieval
- Streaming conversational inference
- Prompt engineering
- Evaluation and recommendation grounding

---

# Assignment Coverage

| Requirement | Implementation |
|---|---|
| Implement at least two LLM-based CRS | ✅ RAG-based CRS + Agent-based CRS |
| Conversational recommendation | ✅ Multi-turn conversational recommendations |
| API serving | ✅ FastAPI streaming endpoint |
| Streaming responses | ✅ StreamingResponse with async generators |
| Performance considerations | ✅ Async architecture with streaming inference |
| Python 3.10+ | ✅ Implemented |
| requirements.txt | ✅ Included |
| Prompt engineering improvements | ✅ Documented |
| Evaluation methodology | ✅ Implemented in `eval.py` |
| User personalization | ✅ User-profile retrieval filtering |
| Modern LLM stack | ✅ OpenAI + LangChain + ChromaDB |

---

# Project Overview

This system is a conversational movie recommender powered by Large Language Models and Retrieval-Augmented Generation.

The recommender uses the Movie subset of the LLM-REDIAL dataset to:
- retrieve semantically similar movie conversations
- ground recommendations in historical dialogue evidence
- personalize recommendations for returning users
- reduce hallucinations
- generate conversational responses

The project additionally implements a lightweight agent workflow capable of:
- intent classification
- recommendation generation
- metadata retrieval
- response orchestration

---

# Implemented CRS Approaches

## 1. RAG-based Conversational Recommender

The RAG system retrieves relevant historical movie conversations from ChromaDB and injects them into the LLM prompt as grounding context.

### Features
- Retrieval-Augmented Generation
- Conversational grounding
- Personalized retrieval
- User memory filtering
- Hallucination reduction
- Streaming inference
- Context-aware recommendations

---

## 2. Agent-based Conversational Recommender

The agent system extends the RAG pipeline with lightweight orchestration logic.

The agent:
1. Classifies user intent
2. Generates a recommendation using RAG
3. Extracts the movie title
4. Calls an external metadata API
5. Produces a final enriched conversational response

### Agent Features
- Intent classification
- Tool/API usage
- Metadata enrichment
- Recommendation explanation
- Multi-step reasoning flow
- External movie information retrieval

---

# Architecture

```text
Streamlit UI
    ↓
FastAPI API Layer
    ↓
Recommendation Orchestrator
    ↓
RAG or Agent Mode
    ↓
ChromaDB Retrieval
    ↓
OpenAI LLM
    ↓
Streaming Response
```

---

# Tech Stack

## Backend
- FastAPI
- Python 3.10+
- AsyncIO
- StreamingResponse

## LLM & Retrieval
- OpenAI GPT-4o-mini
- LangChain
- OpenAI Embeddings
- ChromaDB

## Frontend
- Streamlit

## Additional Components
- IMDbOT API
- Pytest
- Pydantic
- httpx

---

# Dataset

This project uses the Movie subset of the LLM-REDIAL dataset.

The implementation specifically uses:
- `Conversation.txt`
- `final_data.jsonl`
- `item_map.json`
- `user_ids.json`

---

# Prompt Engineering Improvements

## Prompt Improvement 1: Retrieval Grounding Constraint

### Added Prompt Rules

```text
- Use the retrieved context as your main evidence.
- Prefer a movie that appears in the retrieved context.
- Do not invent unrelated movie titles.
```

### Expected Impact
- Reduced hallucinations
- Better recommendation grounding
- More faithful RAG behavior

---

## Prompt Improvement 2: Negative Preference Awareness

### Added Prompt Rules

```text
- Do not recommend movies the user already watched.
- Do not recommend movies the user disliked or rejected.
- Use this user's retrieved memory as the main source for personalization.
```

### Expected Impact
- Better personalization
- Reduced repeated recommendations
- More realistic conversational behavior

---

# Evaluation Methodology

The project includes a dedicated evaluation pipeline in `eval.py`.

The evaluation validates:
- recommendation grounding
- hallucination reduction
- disliked movie filtering
- metadata generation
- recommendation extraction
- unsupported streaming-availability claim prevention
- personalized recommendation behavior for returning users

The evaluation pipeline runs fixed conversational recommendation scenarios across both RAG mode and agent mode. Each generated response is automatically checked against a set of deterministic quality criteria.

## Evaluation Results

The evaluation was executed on 10 fixed conversational recommendation scenarios covering general recommendations, personalized returning-user recommendations, agent metadata enrichment, explicit movie explanation, and unsupported streaming-availability requests.

| Metric | Result |
|---|---:|
| Total evaluation cases | 10 |
| Passed cases | 10 |
| Pass rate | 100% |
| Average score | 88.89% |
| Passing threshold | 75% |
| Checks per case | 9 |

## Evaluation Checks

| Evaluation Check | Purpose |
|---|---|
| `non_empty` | Ensures the model returns an actual response |
| `movie_extracted` | Verifies that a movie title can be extracted from the answer |
| `grounded_in_retrieved_context` | Checks that the recommended movie appears in retrieved RAG context |
| `appears_in_context_recommendations` | Checks whether the title appears in prior structured recommendation fields |
| `does_not_recommend_disliked_movie` | Prevents recommending movies the user disliked |
| `does_not_repeat_liked_movie` | Prevents recommending movies already liked or watched |
| `no_streaming_claim` | Avoids unsupported claims such as saying a movie is available on Netflix |
| `has_agent_metadata_if_agent` | Ensures agent responses include metadata such as year, actors, and IMDb link |
| `expected_title_match` | Verifies expected title alignment for explicit movie questions |

## Evaluation Summary

All 10 evaluation scenarios passed the configured threshold of 75%.

The average score was **88.89%**, meaning each response passed 8 out of 9 automatic checks on average.

The only recurring failed check was:

```text
appears_in_context_recommendations: false
```

This is acceptable because the recommended movie was still grounded in the retrieved context, but it did not always appear specifically inside the structured `Recommended movies:` field.

## Example Evaluation Output

```text
Test case: agent_metadata
Mode: agent
Movie extracted: Iron Monkey
Score: 8/9
Score percent: 88.89%
Passed: True
```

The agent response also included external movie metadata:

```text
Title: Iron Monkey
Year: 1993
Actors: Rongguang Yu, Donnie Yen
IMDb URL: included
Poster: included
```

## Run Evaluation

```bash
python eval.py
```

The script saves a full JSON report to:

```text
evaluation_results.json
```

## Performance Considerations

The system uses:
- async FastAPI endpoints
- StreamingResponse for lower perceived latency
- precomputed embeddings
- lightweight ChromaDB retrieval
- non-blocking external API calls

These choices improve scalability and conversational responsiveness.



## Architectural Decisions

- RAG was chosen to ground recommendations and reduce hallucinations.
- The agent workflow enables tool usage and metadata enrichment.
- ChromaDB was selected for lightweight local semantic retrieval and easy reproducibility.



## Current Limitations

- Recommendation quality depends on retrieval quality.
- No reranking model is implemented yet.
- Evaluation is deterministic and not human-rated.
- External metadata depends on third-party APIs.




# API Endpoint

## Streaming Recommendation Endpoint

```http
POST /recommend/stream
```

### Example Request

```json
{
  "question": "Recommend me a sci-fi movie",
  "history": [],
  "user_id": "12",
  "mode": "rag"
}
```

---

# Running the Project

## 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Create .env

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

## 4. Build the Vector Index

```bash
python app/scripts/build_index.py
```

## 5. Start FastAPI Server

```bash
uvicorn app.main:app --reload
```

## 6. Launch Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

# Testing

Run unit tests:

```bash
pytest
```

---

# Future Improvements

Potential future extensions:
- Multi-agent recommendation systems
- Hybrid retrieval pipelines
- Reranking models
- LangGraph orchestration
- Long-term conversational memory
- RAGAS-based evaluation
- LangSmith tracing

---

# Conclusion

This project demonstrates:
- modern LLM application engineering
- Retrieval-Augmented Generation
- conversational recommendation systems
- streaming API serving
- personalization
- prompt engineering
- lightweight agent orchestration
- evaluation-aware AI system design

The implementation prioritizes:
- grounding
- modularity
- reproducibility
- async scalability
- conversational quality
- engineering clarity
