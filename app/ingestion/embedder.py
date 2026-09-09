from app.core.interfaces import EmbeddingProvider
from app.core.models import ChunkData


def embed_chunks(chunks: list[ChunkData], provider: EmbeddingProvider) -> list[ChunkData]:
    """Compute and attach an embedding to each chunk, preserving order."""
    if not chunks:
        return chunks

    vectors = provider.embed([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector
    return chunks
