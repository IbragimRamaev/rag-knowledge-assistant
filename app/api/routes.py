import logging

import asyncpg
from fastapi import APIRouter, HTTPException

from app.api.dependencies import EmbeddingProviderDep, RAGEngineDep, VectorStoreDep
from app.api.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.ingestion.pipeline import ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_QUESTION_LENGTH = 2000

_DB_UNAVAILABLE_ERRORS = (OSError, asyncpg.PostgresError, asyncpg.InterfaceError)
_DB_UNAVAILABLE_DETAIL = "Vector database is unavailable, try again later."


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, rag_engine: RAGEngineDep) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long ({len(question)} chars, max {MAX_QUESTION_LENGTH}).",
        )

    try:
        answer = await rag_engine.answer(question)
    except _DB_UNAVAILABLE_ERRORS as exc:
        logger.exception("Vector database unavailable while answering a question")
        raise HTTPException(status_code=503, detail=_DB_UNAVAILABLE_DETAIL) from exc

    return AskResponse(answer=answer)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    payload: IngestRequest,
    vector_store: VectorStoreDep,
    embedding_provider: EmbeddingProviderDep,
) -> IngestResponse:
    try:
        chunks = await ingest_documents(payload.documents_dir, embedding_provider)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid documents directory: {exc}") from exc

    try:
        await vector_store.save_chunks(chunks)
    except _DB_UNAVAILABLE_ERRORS as exc:
        logger.exception("Vector database unavailable while saving ingested chunks")
        raise HTTPException(status_code=503, detail=_DB_UNAVAILABLE_DETAIL) from exc

    return IngestResponse(chunks_ingested=len(chunks), documents_dir=payload.documents_dir)
