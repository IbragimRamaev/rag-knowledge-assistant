from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.providers.claude_llm import ClaudeLLMProvider
from app.providers.openai_embedding import OpenAIEmbeddingProvider
from app.rag.rag_engine import RAGEngine
from app.rag.retriever import Retriever
from app.storage.pg_vector_store import PgVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embedding_provider = OpenAIEmbeddingProvider()
    vector_store = PgVectorStore()

    app.state.embedding_provider = embedding_provider
    app.state.vector_store = vector_store
    app.state.rag_engine = RAGEngine(
        retriever=Retriever(embedding_provider, vector_store),
        llm_provider=ClaudeLLMProvider(),
    )

    yield

    await vector_store.close()


app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)
app.include_router(router)
