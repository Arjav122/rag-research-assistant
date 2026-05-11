# RAG Research Assistant

Production-grade, modular AI research intelligence platform built with FastAPI, Streamlit, Qdrant, PostgreSQL, LangChain, and LangGraph.

## Core Capabilities

- Semantic research paper search
- Upload + chat workflow scaffold for papers
- Literature review generation
- Research paper comparison scaffold
- Recommendation engine
- Citation-aware answers
- Conversational memory (session-level)
- Agentic orchestration via LangGraph

## Project Structure

```text
backend/
  api/routes
  agents
  services
  db
  retrieval
  ingestion
  prompts
  utils
  middleware
  tests
frontend/
  pages
  components
  services
  utils
storage/
scripts/
notebooks/
```

## Architecture Overview

### Ingestion Pipeline
arXiv API -> PDF download -> PyMuPDF text extraction -> semantic chunking -> OpenAI embeddings (`text-embedding-3-large`) -> Qdrant storage with metadata.

### Retrieval Pipeline
Query -> Embedding -> Qdrant vector search + BM25 keyword retrieval -> BAAI reranker -> context builder -> LLM response generation.

### Agentic Workflow
LangGraph nodes:
- Retrieval Agent
- Citation Agent
- Summarization Agent

Extensible service layer for:
- Literature Review Agent
- Comparison Agent
- Recommendation Agent

## Environment Variables (Fill Manually in `.env`)

Create a `.env` file in project root and add:

- `OPENAI_API_KEY=...` (your OpenAI API key)
- `QDRANT_URL=http://qdrant:6333` (or your custom Qdrant URL)
- `DATABASE_URL=postgresql+psycopg2://user:password@host:5432/dbname`
- `SECRET_KEY=...` (long random secret)
- `QDRANT_COLLECTION=ai_research_chunks`
- `OPENAI_EMBEDDING_MODEL=text-embedding-3-large`
- `OPENAI_CHAT_MODEL=gpt-4.1-mini`
- `BACKEND_URL=http://backend:8000` (for Docker frontend) or `http://localhost:8000` (local)

## Run with Docker

```bash
docker compose up --build
```

Services:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`
- Qdrant: `http://localhost:6335`

## API Endpoints

- `POST /api/v1/ingestion/papers`
- `POST /api/v1/search/`
- `POST /api/v1/chat/`
- `POST /api/v1/recommendation/`
- `POST /api/v1/literature/review`
- `POST /api/v1/comparison/`
- `GET /api/v1/auth/health`

## Data Ingestion Script

```bash
python scripts/run_ingestion.py
```

## Notes for Production Hardening

- Replace in-memory chat memory with Redis or Postgres-backed memory.
- Add robust arXiv XML parsing (e.g., feedparser or lxml).
- Add async batching + retry/backoff in ingestion.
- Add auth token issuance and user RBAC.
- Add observability (structured logging, tracing, metrics).
- Add CI tests and quality gates.

## Future Enhancements

- Semantic Scholar integration
- Graph RAG
- Multimodal RAG
- Research trend analytics
