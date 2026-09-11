import logging

import tiktoken

from app.core.models import Document
from app.ingestion.chunker import chunk_document, split_into_sentences

ENCODING = tiktoken.get_encoding("cl100k_base")

SENTENCES = [
    "The quarterly report shows strong growth in the northern region.",
    "Sales increased by fifteen percent compared to last year.",
    "Customer satisfaction scores also improved significantly.",
    "The support team reduced average response time by half.",
    "New hires in engineering doubled the team's capacity.",
    "Product launches are now scheduled for every quarter.",
    "Marketing spend was reallocated toward digital channels.",
    "Overall the company is well positioned for next year.",
]


def _make_document() -> Document:
    return Document(source_document="report.md", text=" ".join(SENTENCES))


def test_split_into_sentences_matches_source():
    assert split_into_sentences(" ".join(SENTENCES)) == SENTENCES


def test_chunks_never_split_a_sentence_mid_way():
    chunks = chunk_document(_make_document(), max_tokens=25, overlap_ratio=0.15, encoding=ENCODING)

    assert len(chunks) > 1
    for chunk in chunks:
        for sentence in split_into_sentences(chunk.text):
            assert sentence in SENTENCES


def test_chunk_index_and_source_are_set():
    chunks = chunk_document(_make_document(), max_tokens=25, overlap_ratio=0.15, encoding=ENCODING)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.source_document == "report.md" for c in chunks)
    assert all(c.embedding is None for c in chunks)


def test_overlap_between_adjacent_chunks():
    chunks = chunk_document(_make_document(), max_tokens=25, overlap_ratio=0.15, encoding=ENCODING)

    for previous, current in zip(chunks, chunks[1:]):
        previous_sentences = set(split_into_sentences(previous.text))
        current_sentences = set(split_into_sentences(current.text))
        assert previous_sentences & current_sentences, (
            "adjacent chunks should share at least one sentence"
        )


def test_empty_document_produces_no_chunks():
    document = Document(source_document="empty.txt", text="   ")
    assert chunk_document(document, encoding=ENCODING) == []


def test_document_shorter_than_max_tokens_is_a_single_chunk():
    chunks = chunk_document(_make_document(), max_tokens=10_000, encoding=ENCODING)
    assert len(chunks) == 1
    assert chunks[0].text == " ".join(SENTENCES)


def test_sentence_longer_than_max_tokens_becomes_its_own_chunk_with_warning(caplog):
    # No internal '.'/'!'/'?', so split_into_sentences keeps it as a single sentence.
    long_sentence = "The system continuously processes " + "data " * 1200 + "without interruption."
    document = Document(
        source_document="oversized.md",
        text=f"{SENTENCES[0]} {long_sentence} {SENTENCES[1]}",
    )

    long_sentence_tokens = len(ENCODING.encode(long_sentence))
    assert long_sentence_tokens > 800, "fixture sentence must actually exceed max_tokens"

    with caplog.at_level(logging.WARNING, logger="app.ingestion.chunker"):
        chunks = chunk_document(document, max_tokens=800, overlap_ratio=0.15, encoding=ENCODING)

    # The oversized sentence must not be split - it becomes a single chunk on its own,
    # even though that chunk exceeds max_tokens.
    oversized_chunks = [c for c in chunks if c.text == long_sentence]
    assert len(oversized_chunks) == 1
    assert len(ENCODING.encode(oversized_chunks[0].text)) > 800

    # Its neighbours are still chunked normally.
    assert chunks[0].text == SENTENCES[0]
    assert chunks[-1].text == SENTENCES[1]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "exceeding max_tokens" in warnings[0].message
