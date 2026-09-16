"""
Smoke test for Step 4 (RAG engine): ingest the full knowledge base, then ask
RAGEngine real questions and print its full text answers - including one
question with no answer anywhere in the knowledge base, to check that Claude
says so honestly instead of inventing one.

Requires OPENAI_API_KEY, ANTHROPIC_API_KEY and DATABASE_URL to be set (via
.env), and the schema from app/db/schema.sql to already be applied.

Usage:
    python scripts/smoke_test_rag.py
"""

import asyncio

from app.config import settings
from app.ingestion.pipeline import ingest_documents
from app.providers.claude_llm import ClaudeLLMProvider
from app.providers.openai_embedding import OpenAIEmbeddingProvider
from app.rag.rag_engine import RAGEngine
from app.rag.retriever import Retriever
from app.storage.pg_vector_store import PgVectorStore

DOCUMENTS_DIR = "data/documents"

QUESTIONS = [
    "Сколько дней отпуска положено сотруднику?",
    "Какие расходы компенсируются в командировке?",
    "Что делать, если я получил подозрительное фишинговое письмо?",
    # Edge case: not covered anywhere in the knowledge base.
    "Какая зарплата у CEO?",
]


async def main() -> None:
    embedding_provider = OpenAIEmbeddingProvider()

    chunks = ingest_documents(DOCUMENTS_DIR, embedding_provider)
    print(f"Ingested {len(chunks)} chunks from {DOCUMENTS_DIR}")

    store = PgVectorStore(settings.database_url)
    try:
        await store.save_chunks(chunks)
        print(f"Saved {len(chunks)} chunks (company_id={settings.default_company_id})\n")

        engine = RAGEngine(
            retriever=Retriever(embedding_provider, store),
            llm_provider=ClaudeLLMProvider(),
        )

        for question in QUESTIONS:
            answer = await engine.answer(question)
            print(f"Q: {question}")
            print(f"A: {answer}\n")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
