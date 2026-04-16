# RAG Chatbot

A full-stack document intelligence application built on Retrieval-Augmented Generation. Upload PDFs, ask questions in plain language, and get answers cited directly from the source — with multi-turn conversation memory and built-in hallucination guardrails.

---

Standard LLM integrations hallucinate — the model fills gaps in its knowledge with plausible-sounding fiction. For any application where accuracy matters (legal, medical, internal knowledge bases, research), that's a non-starter.

RAG fixes this by separating *retrieval* from *generation*. The model is only allowed to answer from documents you provide. If the answer isn't there, it says so. This project implements that pattern end to end — from PDF ingestion through a browser-based chat UI — with a clean API boundary between the frontend and backend.

---

## Architecture

The system is split into two independently deployed services that communicate over HTTP.

```
┌─────────────────────────────────────┐
│         Streamlit Frontend          │
│  PDF upload · Chat UI · Session     │
│  state · Source citation display    │
└────────────────┬────────────────────┘
                 │ REST API (JSON)
                 ▼
┌─────────────────────────────────────┐
│          FastAPI Backend            │
│                                     │
│  /upload          /chat             │
│     │                │              │
│     ▼                ▼              │
│  ingest.py      retriever.py        │
│     │                │              │
│     └──────┬─────────┘              │
│            ▼                        │
│       ChromaDB                      │
│  (cosine similarity index)          │
│            │                        │
│            ▼                        │
│        chat.py                      │
│   prompt assembly → GPT-4o          │
└─────────────────────────────────────┘
```

### Two pipelines

**Ingestion** (happens once per document)
```
PDF → pdfplumber → raw text per page
    → sliding window chunker (500 chars, 100 overlap)
    → OpenAI text-embedding-3-small → 1536-dim vectors
    → ChromaDB stores {vector, text, filename, page}
```

**Retrieval + Generation** (happens per query)
```
Question → embedding → ChromaDB cosine search → top 5 chunks
         → chunks + conversation history → GPT-4o prompt
         → cited answer with source filename and page number
```

---

## Frontend

Built with Streamlit. 

**Session state for conversation memory** — Streamlit re-renders the entire page on every interaction. Conversation history is stored in `st.session_state` as a list of `{question, answer}` pairs and sent with every API call. This keeps the backend completely stateless while preserving multi-turn context on the client side — no server-side session management needed.

**Optimistic duplicate prevention** — once a filename is uploaded in a session, it's tracked in `st.session_state.uploaded_docs` so the user doesn't accidentally re-upload and trigger a redundant API call. The backend also deduplicates, but catching it on the frontend saves a round trip.

**Environment-driven API URL** — `API_URL` is read from an environment variable with `localhost:8000` as the default. This means the same frontend code runs identically in local development (pointing to localhost) and in production (pointing to the deployed backend URL) with no code changes.

---

## Backend

Built with FastAPI. Four endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/upload` | POST | Ingests a PDF — parse, chunk, embed, store |
| `/chat` | POST | Answers a question from uploaded document context |
| `/documents` | GET | Lists all indexed filenames |
| `/health` | GET | Health check |

**Validation layer** — FastAPI + Pydantic validate request shapes before any business logic runs. Invalid file types return 400, empty questions return 400, ingestion failures return 422 or 500 with a descriptive message. The frontend never has to guess why a request failed.

**Temp file pattern for uploads** — uploaded files are written to a `tempfile.NamedTemporaryFile`, passed to the ingestion pipeline, then deleted in a `finally` block regardless of success or failure. This prevents disk accumulation on the server without requiring a separate cleanup job.

---

## Specific design decisions

**Why text-embedding-3-small over larger models** — embedding quality scales with model size, but so does cost and latency. `text-embedding-3-small` produces 1536-dimensional vectors at $0.02/million tokens with retrieval quality that's sufficient for document Q&A. The marginal gain from `text-embedding-3-large` doesn't justify 5× the cost for this use case.

**Why cosine similarity over L2** — ChromaDB defaults to L2 (Euclidean) distance. We explicitly configure `hnsw:space: cosine` because cosine similarity measures the angle between vectors, not their magnitude. Two text chunks discussing the same concept at different lengths will have similar angles but different magnitudes — cosine similarity correctly identifies them as semantically related, L2 would penalize the length difference.

**Chunk overlap prevents boundary information loss** — a sentence that falls at the boundary of two 500-character chunks appears in both. Without overlap, a question about content spanning a boundary would retrieve only half the relevant context. The 100-character overlap is approximately one sentence — enough to preserve continuity without significantly increasing the number of chunks.

**Deduplication before ingestion** — re-uploading the same PDF without checking creates duplicate embedding IDs in ChromaDB, which corrupts cosine similarity scores (scores go negative, retrieval breaks entirely). The check is a metadata query — `collection.get(where={"filename": filename})` — before any embedding computation happens.

**Prompt ordering: context before history** — the GPT-4o prompt is assembled as: system instructions → retrieved chunks → conversation history → current question. Context deliberately comes before history. If history came first, the model might anchor on previous answers rather than re-grounding in the retrieved chunks for the new question. Ordering matters.

**Temperature 0.2** — low enough to keep answers factual and grounded in the source material, high enough to avoid robotic phrasing. Temperature 0 produces deterministic but sometimes terse answers. 0.2 is the practical sweet spot for document Q&A.

**Hallucination guardrail** — the system prompt instructs GPT-4o to respond with a fixed phrase when the answer isn't in the context. This is explicitly tested — off-topic questions like "what is the capital of France?" return the guardrail response, not a hallucinated answer.

---

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── main.py         # FastAPI app — routing, validation, error handling
│   ├── ingest.py       # Ingestion pipeline — parse, chunk, embed, store
│   ├── retriever.py    # Retrieval — embed query, cosine search, rank chunks
│   ├── chat.py         # Generation — prompt assembly, GPT-4o, citations
│   └── config.py       # Centralized config — models, chunk sizes, paths
├── frontend/
│   └── app.py          # Streamlit UI — upload, chat, session state
├── .env.example
├── .python-version     # Pins Python 3.11 for reproducible deploys
├── render.yaml         # Infrastructure as code — both services declared
└── requirements.txt    # Pinned dependencies
```

---

## Local Setup

Requires Python 3.11+ and an OpenAI API key.

```bash
git clone git@github.com:Hesyet000/rag-chatbot.git
cd rag-chatbot

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# paste your OpenAI API key into .env

# Terminal 1
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2
streamlit run frontend/app.py
```

`http://localhost:8501` — frontend  
`http://localhost:8000/docs` — interactive API docs

---

## Deployment

Both services deploy to Render from the same GitHub repository. The `render.yaml` at the repo root declares both services — runtime, build command, start command, and environment variables. Render detects the file automatically on connect.

The frontend service receives the backend's public URL as `API_URL` at deploy time. No hardcoded URLs anywhere in the code.

**Free tier caveat** — Render's free tier uses an ephemeral filesystem. ChromaDB data doesn't survive service restarts. For a persistent production deployment, ChromaDB local would be replaced with a managed vector database. The `get_chroma_collection()` abstraction in `retriever.py` and `ingest.py` makes this swap localized to one function in each file.

---

## Stack

FastAPI · Streamlit · OpenAI GPT-4o · text-embedding-3-small · ChromaDB · pdfplumber · Render
