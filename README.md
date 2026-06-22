# WeddingAgent — Production RAG + Agentic AI System

A production-grade AI agent that helps users discover, compare, and book wedding vendors across India using a corpus of **70,000+ real vendor records** scraped from WeddingWire India.

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
| [Data Ingestion](docs/04-data-ingestion.md) | Parsing scrape_wire data, cleaning, indexing pipeline |
| [Evaluation](docs/05-evaluation.md) | Metrics, test sets, quality gates, monitoring |
| [Implementation Roadmap](docs/06-implementation-roadmap.md) | Phase-by-phase build plan with milestones |

---

## Tech Stack (Summary)

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

## Quick Start (After Building)

```bash
# 1. Clone and install
git clone https://github.com/you/weddingagent
cd weddingagent && pip install -r requirements.txt

# 2. Start infrastructure
docker-compose up -d  # Qdrant, PostgreSQL, Valkey

# 3. Ingest data
python scripts/ingest.py --data-dir ~/Documents/scrape_wire/data

# 4. Run the agent
python -m uvicorn app.main:app --reload

# 5. Open the UI
streamlit run frontend/app.py
```

---

## Project Structure

```
weddingagent/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── agent/
│   │   ├── orchestrator.py  # Agent loop
│   │   └── tools.py         # Tool implementations
│   ├── rag/
│   │   ├── embedder.py      # Embedding generation
│   │   ├── retriever.py     # Vector + hybrid search
│   │   └── reranker.py      # Cross-encoder reranking
│   ├── db/
│   │   ├── qdrant.py        # Vector DB client
│   │   └── postgres.py      # Structured DB client
│   └── api/
│       └── routes.py        # REST endpoints
├── scripts/
│   ├── ingest.py            # Data ingestion pipeline
│   ├── embed_batch.py       # Batch embedding with checkpointing
│   └── evaluate.py          # Evaluation runner
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/
│   ├── test_retrieval.py
│   ├── test_agent.py
│   └── eval_dataset.json    # Ground-truth query set
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
