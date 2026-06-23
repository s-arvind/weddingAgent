# WeddingAgent — Production RAG + Agentic AI System

A production-grade AI agent that helps users discover, compare, and book wedding vendors across India using a corpus of **70,000+ real vendor records**.

## What It Does

Users ask natural-language questions; the agent autonomously chains tools to answer them:

> "Find me a banquet hall in South Delhi for 200 guests with veg catering under ₹1800/head, and compare the top 3"

The agent will: search by city + category → filter by capacity + price → retrieve full details → synthesize a comparison — all without the user constructing a single filter.

---

## Why This Project

| Signal | Resume Value |
|---|---|
| 70k+ real documents | Demonstrates working at realistic scale |
| RAG pipeline | Core demanded skill in AI/ML roles (2024–2026) |
| Agentic tool-use | Differentiates from basic LLM wrappers |
| Evaluation suite | Shows engineering discipline, not just vibes |
| End-to-end system | Proves you can ship, not just prototype |

---

## System Overview

```
User Query (natural language)
        │
        ▼
┌───────────────────┐
│   FastAPI Gateway │  ← auth, rate limiting, request tracing
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Agent Layer     │  ← LLM-agnostic tool-use (default: Groq)
│   (Orchestrator)  │    decides which tools to call, in what order
└────────┬──────────┘
         │
    ┌────┴─────────────────────────────────────┐
    │              Tool Registry               │
    ├──────────────┬───────────┬───────────────┤
    │ search_      │ filter_   │ get_vendor_   │
    │ vendors      │ by_       │ details /     │
    │ (RAG)        │ criteria  │ compare       │
    └──────┬───────┴─────┬─────┴───────────────┘
           │             │
           ▼             ▼
    ┌─────────────┐  ┌──────────────┐
    │  Vector DB  │  │  PostgreSQL  │
    │  (Qdrant)   │  │  (Metadata   │
    │  Embeddings │  │   + Filters) │
    └─────────────┘  └──────────────┘
```

---

## Documentation Index

| Document | Contents |
|---|---|
| [Architecture](docs/01-architecture.md) | Full system design, component decisions, data flow |
| [RAG Pipeline](docs/02-rag-pipeline.md) | Chunking strategy, embeddings, vector DB, retrieval |
| [Agent Design](docs/03-agent-design.md) | Tool definitions, orchestration, multi-step reasoning |
| [Data Ingestion](docs/04-data-ingestion.md) | Parsing vendor data, cleaning, indexing pipeline |
| [Evaluation](docs/05-evaluation.md) | Metrics, test sets, quality gates, monitoring |
| [Implementation Roadmap](docs/06-implementation-roadmap.md) | Phase-by-phase build plan with milestones |

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Embedding | `BAAI/bge-m3` (local) | Free, multilingual (Hinglish queries), 1024-dim |
| Vector DB | Qdrant | Production-ready, filtering + ANN, Docker-native |
| Structured DB | PostgreSQL | Structured metadata, complex filters |
| Cache | Valkey | Drop-in Redis replacement, BSD licensed |
| Agent LLM | Groq (LLM-agnostic) | Free tier, fast, OpenAI-compatible API |
| Backend | FastAPI + Python | Async, fast, great AI ecosystem |
| Frontend | Streamlit (MVP) → Next.js | Fast demo → production upgrade |
| Observability | LangSmith + Prometheus | RAG tracing + infra metrics |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/s-arvind/weddingAgent
cd weddingAgent && pip install -r requirements.txt

# 2. Copy env and fill in keys
cp .env.example .env

# 3. Start infrastructure
docker compose up -d

# 4. Run migrations
alembic upgrade head

# 5. Ingest vendor data
PYTHONPATH=. python scripts/ingest.py

# 6. Start the API
PYTHONPATH=. uvicorn app.main:app --reload --port 8000

# 7. Start the UI
PYTHONPATH=. streamlit run frontend/app.py
```

---

## Project Structure

```
weddingAgent/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── agent/
│   │   ├── orchestrator.py  # Agent loop
│   │   └── tools.py         # Tool implementations
│   ├── rag/
│   │   ├── embedder.py      # Embedding generation
│   │   └── retriever.py     # Hybrid search (dense + BM25 + RRF)
│   ├── db/
│   │   ├── qdrant.py        # Vector DB client
│   │   └── postgres.py      # Structured DB client
│   ├── models/
│   │   ├── vendor.py        # Vendor ORM model
│   │   └── vendor_collection.py  # Qdrant collection ops
│   └── api/
│       ├── routes.py        # REST endpoints
│       ├── schemas.py       # Pydantic request/response models
│       └── session.py       # Valkey session store
├── scripts/
│   └── ingest.py            # Data ingestion + embedding pipeline
├── migrations/              # Alembic migrations
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
