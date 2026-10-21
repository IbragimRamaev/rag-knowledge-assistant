from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.core.models import RetrievedChunk, StreamSources, StreamTextChunk
from app.providers.claude_llm import ClaudeLLMProvider


def _fake_text_block(text: str, citations: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text, citations=citations)


def _fake_citation(source: str) -> SimpleNamespace:
    return SimpleNamespace(source=source)


def _fake_create_client(content: list) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


class _FakeMessageStream:
    """Minimal stand-in for anthropic's MessageStream: async context manager whose
    __aiter__ replays canned events, plus get_final_message() for the usage log."""

    def __init__(self, events: list, final_message) -> None:
        self._events = events
        self._final_message = final_message

    async def __aenter__(self) -> "_FakeMessageStream":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def __aiter__(self):
        for event in self._events:
            yield event

    async def get_final_message(self):
        return self._final_message


def _fake_stream_client(events: list) -> MagicMock:
    final_message = MagicMock()
    final_message.usage.input_tokens = 10
    final_message.usage.output_tokens = 5

    client = MagicMock()
    client.messages.stream = MagicMock(return_value=_FakeMessageStream(events, final_message))
    return client


@pytest.mark.asyncio
async def test_generate_extracts_cited_documents_deduped_in_order():
    client = _fake_create_client(
        [
            _fake_text_block("Первая часть. "),
            _fake_text_block("Вторая часть.", citations=[_fake_citation("doc_a.md")]),
            _fake_text_block(
                " Третья часть.",
                citations=[_fake_citation("doc_b.md"), _fake_citation("doc_a.md")],
            ),
        ]
    )

    provider = ClaudeLLMProvider(client=client)
    result = await provider.generate("question", context_chunks=[])

    assert result.text == "Первая часть. Вторая часть. Третья часть."
    # doc_a cited twice, doc_b once - deduped, kept in first-appearance order.
    assert result.cited_documents == ["doc_a.md", "doc_b.md"]


@pytest.mark.asyncio
async def test_generate_returns_no_cited_documents_when_model_cites_nothing():
    client = _fake_create_client([_fake_text_block("Просто ответ без цитат.")])

    provider = ClaudeLLMProvider(client=client)
    result = await provider.generate("question", context_chunks=[])

    assert result.cited_documents == []


@pytest.mark.asyncio
async def test_generate_stream_yields_text_chunks_then_one_sources_event():
    events = [
        SimpleNamespace(type="text", text="Привет, "),
        SimpleNamespace(type="citation", citation=_fake_citation("doc_a.md")),
        SimpleNamespace(type="text", text="мир."),
        SimpleNamespace(type="citation", citation=_fake_citation("doc_a.md")),
    ]
    client = _fake_stream_client(events)

    provider = ClaudeLLMProvider(client=client)
    received = [event async for event in provider.generate_stream("question", context_chunks=[])]

    assert received == [
        StreamTextChunk(text="Привет, "),
        StreamTextChunk(text="мир."),
        StreamSources(source_documents=["doc_a.md"]),  # deduped across the two citation events
    ]


@pytest.mark.asyncio
async def test_generate_stream_sources_event_is_empty_when_no_citations():
    events = [SimpleNamespace(type="text", text="Ответ без цитат.")]
    client = _fake_stream_client(events)

    provider = ClaudeLLMProvider(client=client)
    received = [event async for event in provider.generate_stream("question", context_chunks=[])]

    assert received == [
        StreamTextChunk(text="Ответ без цитат."),
        StreamSources(source_documents=[]),
    ]


_REAL_CHUNKS = [
    RetrievedChunk(
        text="Сотрудникам полагается 28 календарных дней оплачиваемого отпуска в год.",
        source_document="company_policy.md",
        chunk_index=0,
        score=0.69,
    ),
    RetrievedChunk(
        text="Компания предоставляет ноутбук и необходимую периферию для удалённой работы.",
        source_document="remote_work_policy.md",
        chunk_index=0,
        score=0.4,
    ),
]


@pytest.mark.integration
@pytest.mark.skipif(
    not settings.anthropic_api_key,
    reason="requires a real ANTHROPIC_API_KEY - hits the live Sonnet API",
)
@pytest.mark.asyncio
async def test_generate_real_api_cites_only_the_relevant_chunk():
    provider = ClaudeLLMProvider()
    result = await provider.generate("Сколько дней отпуска положено сотруднику?", _REAL_CHUNKS)

    assert "28" in result.text
    assert "company_policy.md" in result.cited_documents
    assert "remote_work_policy.md" not in result.cited_documents
