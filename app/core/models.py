from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    """A raw document loaded from a data source, before chunking."""

    source_document: str
    text: str


@dataclass
class ChunkData:
    """A single chunk produced by the ingestion pipeline, with an optional embedding."""

    text: str
    source_document: str
    chunk_index: int
    embedding: Optional[list[float]] = None


@dataclass
class RetrievedChunk:
    """A chunk returned by VectorStore.search, ranked by similarity to the query."""

    text: str
    source_document: str
    chunk_index: int
    score: float
