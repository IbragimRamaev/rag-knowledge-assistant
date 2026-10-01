# RAG Knowledge Assistant

AI-powered assistant that answers employee questions using a company's internal knowledge base (RAG — Retrieval-Augmented Generation).

## Status

🚧 Work in progress — MVP under active development. Core pipeline (ingestion, retrieval, generation), the API, a basic web chat, Docker and CI are in place; conversation memory and streaming are not yet.

## How it works

1. Company documents are chunked and converted into embeddings.
2. Embeddings are stored in a vector database (pgvector).
3. On a user question, the system retrieves the most relevant chunks.
4. An LLM (Claude Sonnet 5) generates an answer grounded in the retrieved context.

## Tech stack

- **Backend:** Python, FastAPI
- **Vector store:** PostgreSQL + pgvector
- **LLM:** Claude Sonnet 5 (Anthropic API)
- **Embeddings:** OpenAI text-embedding-3-small
- **Frontend:** vanilla HTML/CSS/JS, no framework
- **Infra:** Docker, docker-compose

## Architecture

Five independent, replaceable layers:

1. **Data sources** — company documents (`.txt`/`.md`) in `data/documents/`
2. **Ingestion pipeline** (`app/ingestion/`) — chunking + embedding generation
3. **Vector DB** (`app/storage/`) — PostgreSQL + pgvector, behind a `VectorStore` interface
4. **RAG engine** (`app/rag/`) — retrieval (vector search) + generation (Claude), with a relevance threshold that skips the LLM call entirely for off-topic questions
5. **API + chat UI** — FastAPI (`app/api/`, `POST /ask`, `POST /ingest`) + a static chat frontend (`frontend/`)

Each layer sits behind an interface (`VectorStore`, `LLMProvider`, `EmbeddingProvider`), so the underlying implementation can be swapped without touching the rest of the system.

## Roadmap / Known limitations

Only `.txt` and `.md` documents are supported right now. This is a deliberate MVP scope, not a technical constraint — the loader sits behind an interface, isolated from the rest of the pipeline, so PDF, Word (`.docx`) and Excel support can be added as new modules under `app/ingestion/` without touching the chunker, embedder or storage layer.

Tables are a separate problem. Reading the file is the easy part — tabular data doesn't fit sentence-based chunking well, so it'll need its own row-based processing path rather than reusing the current chunker.

Conversation memory (sending prior turns back to the LLM) and streaming responses are planned but not implemented — every `/ask` call today is a single, independent question.

New formats and features will be added as real needs come up, not preemptively.

## Setup

### Option A: everything in Docker (recommended)

1. Create a `.env` file in the project root with the required variables — see `app/config.py` for the full list (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, ...). For Docker, `DATABASE_URL` is overridden automatically inside the containers (see the comment in `docker/docker-compose.yml`); the `.env` value is only what runs outside Docker.

2. Build and start Postgres + the API:

   ```bash
   docker compose -f docker/docker-compose.yml up -d --build
   ```

   The `app` service mounts `./app` as a volume and runs `uvicorn --reload`, so local code edits are picked up without rebuilding the image.

3. Apply the DB schema (first run only):

   ```bash
   docker exec -i rag-postgres psql -U postgres -d rag_db < app/db/schema.sql
   ```

4. Ingest the knowledge base and ask a question:

   ```bash
   curl -X POST http://localhost:8000/ingest
   curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Сколько дней отпуска положено сотруднику?"}'
   ```

5. Open the chat UI — serve `frontend/` with any static server and open it in a browser:

   ```bash
   cd frontend && python3 -m http.server 5500
   ```

   Then visit <http://localhost:5500>.

### Option B: API running locally (Postgres still via Docker)

1. Create a virtualenv and install dependencies (`requirements.txt` covers both the runtime app and test/lint tooling):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create `.env` as in Option A (`DATABASE_URL` should point at `localhost`, which is already the default in `app/config.py`).

3. Start just Postgres and apply the schema:

   ```bash
   docker compose -f docker/docker-compose.yml up -d postgres
   docker exec -i rag-postgres psql -U postgres -d rag_db < app/db/schema.sql
   ```

4. Run the API:

   ```bash
   uvicorn app.main:app --reload
   ```

### Pre-commit hooks

The hooks config lives in `.github/pre-commit/` rather than the repo root, so `git commit` keeps working automatically (the installed hook has the path baked in), but manual invocations need an explicit `-c`:

```bash
pre-commit install --config .github/pre-commit/.pre-commit-config.yaml   # one-time
pre-commit run --all-files -c .github/pre-commit/.pre-commit-config.yaml
```

### Tests

```bash
pytest tests/
```

`tests/` holds pure-logic unit tests (chunking) that run without credentials, including in CI. `scripts/` has integration/smoke scripts that hit the real OpenAI/Anthropic APIs and Postgres — run them manually against a configured `.env` (e.g. `python scripts/smoke_test_rag.py`).

## License

MIT
