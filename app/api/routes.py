import json
import logging
from collections.abc import AsyncIterator

import asyncpg
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.dependencies import EmbeddingProviderDep, RAGEngineDep, VectorStoreDep
from app.api.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from app.core.models import HistoryTurn, StreamTextChunk
from app.ingestion.pipeline import ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_QUESTION_LENGTH = 2000
MAX_HISTORY_MESSAGES = 20

_DB_UNAVAILABLE_ERRORS = (OSError, asyncpg.PostgresError, asyncpg.InterfaceError)
_DB_UNAVAILABLE_DETAIL = "Vector database is unavailable, try again later."


def _validate_ask_payload(payload: AskRequest) -> tuple[str, list[HistoryTurn]]:
    """Shared by /ask and /ask/stream: validate the question, cap history. Raises
    HTTPException(400) - safe to call before a StreamingResponse even starts, since
    FastAPI handles the exception before the generator is ever touched."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long ({len(question)} chars, max {MAX_QUESTION_LENGTH}).",
        )

    # MVP simplification: the client resends the full history on every request and
    # nothing is persisted server-side. For production this should move to DB-backed
    # session storage keyed by session_id/user_id instead - this trusts whatever the
    # client sends and caps it defensively, but doesn't survive a page reload.
    history = payload.history[-MAX_HISTORY_MESSAGES:]
    return question, history


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, rag_engine: RAGEngineDep) -> AskResponse:
    question, history = _validate_ask_payload(payload)

    try:
        result = await rag_engine.answer(question, history=history)
    except _DB_UNAVAILABLE_ERRORS as exc:
        logger.exception("Vector database unavailable while answering a question")
        raise HTTPException(status_code=503, detail=_DB_UNAVAILABLE_DETAIL) from exc

    return AskResponse(answer=result.answer, source_documents=result.source_documents)


@router.post("/ask/stream")
async def ask_stream(payload: AskRequest, rag_engine: RAGEngineDep) -> StreamingResponse:
    question, history = _validate_ask_payload(payload)

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in rag_engine.answer_stream(question, history=history):
                if isinstance(event, StreamTextChunk):
                    data = {"type": "text", "content": event.text}
                else:
                    data = {"type": "sources", "content": event.source_documents}
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        except _DB_UNAVAILABLE_ERRORS:
            # The 200 + SSE headers are already committed by the time this can fire
            # (StreamingResponse sends them before the first chunk), so a real 503
            # isn't possible here - the client sees a normal stream carrying one
            # error-typed event instead.
            logger.exception("Vector database unavailable while streaming an answer")
            data = {"type": "error", "content": _DB_UNAVAILABLE_DETAIL}
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
