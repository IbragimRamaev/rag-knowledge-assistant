# RAG Knowledge Assistant

AI-powered assistant that answers employee questions using a company's internal knowledge base (RAG — Retrieval-Augmented Generation).

## Status

🚧 Work in progress — MVP under active development.

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
- **Infra:** Docker, docker-compose

## Roadmap / Known limitations

Only `.txt` and `.md` documents are supported right now. This is a deliberate MVP scope, not a technical constraint — the loader sits behind an interface, isolated from the rest of the pipeline, so PDF, Word (`.docx`) and Excel support can be added as new modules under `app/ingestion/` without touching the chunker, embedder or storage layer.

Tables are a separate problem. Reading the file is the easy part — tabular data doesn't fit sentence-based chunking well, so it'll need its own row-based processing path rather than reusing the current chunker.

New formats will be added as real client needs come up, not preemptively.

## Architecture


Each layer sits behind an interface (`VectorStore`, `LLMProvider`, `EmbeddingProvider`), so the underlying implementation can be swapped without touching the rest of the system.

## Setup

1. Create a virtualenv and install dependencies (`requirements-dev.txt` pulls in `requirements.txt` plus test/lint tooling):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   ```

2. Create a `.env` file with the required variables — see `app/config.py` for the full list (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, ...).

3. Start Postgres + pgvector and apply the schema:

   ```bash
   docker compose up -d
   docker exec -i rag-postgres psql -U postgres -d rag_db < app/db/schema.sql
   ```

4. Install the git hooks (ruff, detect-secrets, pytest run automatically on every commit), then run them once against the whole repo to check the current state:

   ```bash
   pre-commit install
   pre-commit run --all-files
   ```

## License

MIT
