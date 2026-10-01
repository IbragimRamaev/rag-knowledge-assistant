from typing import Annotated

from fastapi import Depends, Request

from app.core.interfaces import EmbeddingProvider, VectorStore
from app.rag.rag_engine import RAGEngine


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
RAGEngineDep = Annotated[RAGEngine, Depends(get_rag_engine)]
