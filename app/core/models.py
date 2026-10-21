from dataclasses import dataclass


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
    embedding: list[float] | None = None


@dataclass
class RetrievedChunk:
    """A chunk returned by VectorStore.search, ranked by similarity to the query."""

    text: str
    source_document: str
    chunk_index: int
    score: float


@dataclass
class HistoryTurn:
    """One prior question/answer pair from the conversation, as sent by the client."""

    question: str
    answer: str


NO_INFORMATION_ANSWER = "У меня нет этой информации в базе знаний."


@dataclass
class AnswerResult:
    """The result of RAGEngine.answer(): the generated answer plus which documents grounded it."""

    answer: str
    source_documents: list[str]


@dataclass
class StreamTextChunk:
    """One incremental piece of text from RAGEngine.answer_stream()."""

    text: str


@dataclass
class StreamSources:
    """Final event from RAGEngine.answer_stream(): which documents grounded the answer."""

    source_documents: list[str]
