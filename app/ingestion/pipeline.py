from pathlib import Path

from app.core.interfaces import EmbeddingProvider
from app.core.models import ChunkData
from app.ingestion.chunker import chunk_document
from app.ingestion.embedder import embed_chunks
from app.ingestion.loader import load_documents


def ingest_documents(
    documents_dir: Path | str, embedding_provider: EmbeddingProvider
) -> list[ChunkData]:
    """Load documents from a directory, chunk them, and attach embeddings. No persistence."""
    chunks: list[ChunkData] = []
    for document in load_documents(documents_dir):
        chunks.extend(chunk_document(document))
    return embed_chunks(chunks, embedding_provider)
