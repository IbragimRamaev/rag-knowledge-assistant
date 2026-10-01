import logging
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from app.config import settings
from app.core.interfaces import LLMProvider
from app.core.models import NO_INFORMATION_ANSWER, HistoryTurn, RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""\
You are a company knowledge-base assistant. You answer employee questions using \
ONLY the information given inside <context> tags.

Rules:
- Base your answer strictly on <context>. Never use outside knowledge, even if you know the answer.
- If <context> does not contain enough information to answer <question>, say so honestly \
(e.g. "{NO_INFORMATION_ANSWER}") instead of guessing or inventing an answer.
- Respond in Russian, concisely - a few sentences at most.
- Plain text only - no markdown (no **bold**, no bullet/numbered lists, no headers). \
The answer is rendered as-is in a chat UI that does not interpret markdown.\
"""

_MAX_TOKENS = 1024


class ClaudeLLMProvider(LLMProvider):
    """LLMProvider backed by the Anthropic Messages API."""

    def __init__(
        self, client: AsyncAnthropic | None = None, model: str = settings.llm_model
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model

    def _build_messages(
        self, question: str, context_chunks: list[RetrievedChunk], history: list[HistoryTurn] | None
    ) -> list[dict]:
        context = "\n\n".join(chunk.text for chunk in context_chunks)
        user_message = f"<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"

        # Prior turns as plain dialogue (for tone only) - the <context> grounding is
        # scoped to the current turn alone, not re-sent for earlier ones.
        messages = []
        for turn in history or []:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        messages.append({"role": "user", "content": user_message})
        return messages

    async def generate(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> str:
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
        return "".join(block.text for block in response.content if block.type == "text")

    async def generate_stream(
        self,
        question: str,
        context_chunks: list[RetrievedChunk],
        history: list[HistoryTurn] | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(question, context_chunks, history)

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

            final_message = await stream.get_final_message()
            logger.info(
                "Claude usage (stream): model=%s input_tokens=%d output_tokens=%d",
                self._model,
                final_message.usage.input_tokens,
                final_message.usage.output_tokens,
            )
