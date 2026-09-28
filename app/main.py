from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Wide open on purpose: the frontend is a static file that can be opened as file://
# or served from any local port, so there's no single origin to allowlist, and the
# API has no cookie-based auth for a wildcard to put at risk. Tighten this to a real
# origin allowlist before this ever serves actual users.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
