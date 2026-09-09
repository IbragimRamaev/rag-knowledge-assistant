from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import settings
from app.core.interfaces import VectorStore
from app.core.models import ChunkData, RetrievedChunk

_UPSERT_CHUNK = """
    INSERT INTO chunks (company_id, source_document, chunk_index, text, embedding)
    VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (company_id, source_document, chunk_index)
    DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""

_SEARCH_CHUNKS = """
    SELECT text, source_document, chunk_index, 1 - (embedding <=> $1) AS score
    FROM chunks
    WHERE company_id = $2
    ORDER BY embedding <=> $1
    LIMIT $3
"""


class PgVectorStore(VectorStore):
    """VectorStore backed by PostgreSQL + pgvector, accessed via asyncpg."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or settings.database_url
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, init=register_vector)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def save_chunks(self, chunks: list[ChunkData], company_id: str | None = None) -> None:
        if not chunks:
            return
        company_id = company_id or settings.default_company_id
        rows = [
            (company_id, chunk.source_document, chunk.chunk_index, chunk.text, chunk.embedding)
            for chunk in chunks
        ]
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.executemany(_UPSERT_CHUNK, rows)

    async def search(
        self, query_embedding: list[float], top_k: int = 5, company_id: str | None = None
    ) -> list[RetrievedChunk]:
        company_id = company_id or settings.default_company_id
        pool = await self._get_pool()
        records = await pool.fetch(_SEARCH_CHUNKS, query_embedding, company_id, top_k)
        return [
            RetrievedChunk(
                text=record["text"],
                source_document=record["source_document"],
                chunk_index=record["chunk_index"],
                score=record["score"],
            )
            for record in records
        ]
