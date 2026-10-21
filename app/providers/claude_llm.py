import logging
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings
from app.core.interfaces import LLMProvider
from app.core.models import (
    NO_INFORMATION_ANSWER,
    GenerationResult,
    HistoryTurn,
    RetrievedChunk,
    StreamSources,
    StreamTextChunk,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""\
You are a company knowledge-base assistant. You answer employee questions using \
ONLY the information given in the provided search results.

Rules:
- Base your answer strictly on the search results. Never use outside knowledge, \
even if you know the answer.
- If the search results do not contain enough information to answer the question, \
say so honestly (e.g. "{NO_INFORMATION_ANSWER}") instead of guessing or inventing \
an answer.
- Respond in Russian, concisely - a few sentences at most.
- Formatting: you may use **bold** for emphasis and "- " bullet lists where a list \
genuinely reads better than prose (e.g. several distinct items). Never use headers, \
tables, links, code blocks, or any HTML. Most answers are a few sentences and don't \
need either device - reach for them only when they actually help, not by default.\
"""

_MAX_TOKENS = 1024


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


class ClaudeLLMProvider(LLMProvider):
    """LLMProvider backed by the Anthropic Messages API. Context chunks are passed
    as search_result blocks with citations enabled, so the model's citations (not
    just retrieval score) can drive which documents get reported as sources."""

    def __init__(
        self, client: AsyncAnthropic | None = None, model: str = settings.llm_model
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model

    def _build_messages(
        self, question: str, context_chunks: list[RetrievedChunk], history: list[HistoryTurn] | None
    ) -> list[dict]:
        search_results = [
            {
                "type": "search_result",
                "source": chunk.source_document,
                "title": chunk.source_document,
                "content": [{"type": "text", "text": chunk.text}],
                "citations": {"enabled": True},
            }
            for chunk in context_chunks
        ]
        user_content = [*search_results, {"type": "text", "text": question}]

        # Prior turns as plain dialogue (for tone only) - search results are scoped
        # to the current turn alone, not re-sent for earlier ones.
        messages = []
        for turn in history or []:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        messages.append({"role": "user", "content": user_content})
        return messages

    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> GenerationResult:
        messages = self._build_messages(question, context_chunks, history)

        # No `temperature` here on purpose: it's removed for claude-sonnet-5 - passing
        # it (even via extra_body) fails with 400 "temperature is deprecated for this
        # model." Grounding comes from _SYSTEM_PROMPT + disabled thinking instead.
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            messages=messages,
        )
        logger.info(
            "Claude usage: model=%s input_tokens=%d output_tokens=%d",
            self._model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        text_parts = []
        cited_documents = []
        for block in response.content:
            if block.type != "text":
                continue
            text_parts.append(block.text)
            for citation in block.citations or []:
                cited_documents.append(citation.source)

        return GenerationResult(
            text="".join(text_parts), cited_documents=_dedupe_preserve_order(cited_documents)
        )

    async def generate_stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> AsyncIterator[StreamTextChunk | StreamSources]:
        messages = self._build_messages(question, context_chunks, history)
        cited_documents = []

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            messages=messages,
        ) as stream:
            # Raw event iteration, not stream.text_stream: text_stream only ever
            # yields text deltas, with no way to also see citations as they arrive.
            async for event in stream:
                if event.type == "text":
                    yield StreamTextChunk(text=event.text)
                elif event.type == "citation":
                    cited_documents.append(event.citation.source)

            final_message = await stream.get_final_message()
            logger.info(
                "Claude usage (stream): model=%s input_tokens=%d output_tokens=%d",
                self._model,
                final_message.usage.input_tokens,
                final_message.usage.output_tokens,
            )

        yield StreamSources(source_documents=_dedupe_preserve_order(cited_documents))
