import logging
import re

import tiktoken

from app.config import settings
from app.core.models import ChunkData, Document

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9«\"])")

DEFAULT_MAX_TOKENS = 800
DEFAULT_OVERLAP_RATIO = 0.15


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!' or '?' followed by whitespace and a capital/digit."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(normalized) if s.strip()]


def chunk_document(
    document: Document,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
    encoding: tiktoken.Encoding | None = None,
) -> list[ChunkData]:
    """
    Split a document into chunks of up to `max_tokens` tokens (~500-800 for typical
    prose), packing whole sentences greedily so a sentence is never split across a
    chunk boundary. Consecutive chunks share a tail/head overlap worth roughly
    `overlap_ratio` of the preceding chunk's tokens.
    """
    encoding = encoding or tiktoken.encoding_for_model(settings.embedding_model)
    sentences = split_into_sentences(document.text)
    if not sentences:
        return []

    sentence_tokens = [len(encoding.encode(s)) for s in sentences]
    total_sentences = len(sentences)
    chunks: list[ChunkData] = []
    start = 0

    while start < total_sentences:
        end = start
        token_count = 0
        while end < total_sentences and (
            token_count + sentence_tokens[end] <= max_tokens or end == start
        ):
            token_count += sentence_tokens[end]
            end += 1

        if end == start + 1 and sentence_tokens[start] > max_tokens:
            logger.warning(
                "Sentence %d of %r has %d tokens, exceeding max_tokens=%d on its own; "
                "keeping it as a single oversized chunk instead of splitting it mid-sentence.",
                start,
                document.source_document,
                sentence_tokens[start],
                max_tokens,
            )

        chunk_text = " ".join(sentences[start:end])
        chunks.append(
            ChunkData(
                text=chunk_text, source_document=document.source_document, chunk_index=len(chunks)
            )
        )

        if end >= total_sentences:
            break

        overlap_target = token_count * overlap_ratio
        back = end
        overlap_tokens = 0
        while back > start and overlap_tokens < overlap_target:
            back -= 1
            overlap_tokens += sentence_tokens[back]

        start = back if back > start else end

    return chunks
