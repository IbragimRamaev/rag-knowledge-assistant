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

## Architecture


Each layer sits behind an interface (`VectorStore`, `LLMProvider`, `EmbeddingProvider`), so the underlying implementation can be swapped without touching the rest of the system.

## Setup

Setup instructions will be added once the initial pipeline is working.

## License

MIT
