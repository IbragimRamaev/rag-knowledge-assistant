import logging

from anthropic import AsyncAnthropic

from app.config import settings
from app.core.interfaces import LLMProvider
from app.core.models import NO_INFORMATION_ANSWER, RetrievedChunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = f"""\
You are a company knowledge-base assistant. You answer employee questions using \
ONLY the information given inside <context> tags.

Rules:
- Base your answer strictly on <context>. Never use outside knowledge, even if you know the answer.
- If <context> does not contain enough information to answer <question>, say so honestly \
(e.g. "{NO_INFORMATION_ANSWER}") instead of guessing or inventing an answer.
- Respond in Russian, concisely - a few sentences at most.\
"""

_MAX_TOKENS = 1024


class ClaudeLLMProvider(LLMProvider):
    """LLMProvider backed by the Anthropic Messages API."""

    def __init__(
        self, client: AsyncAnthropic | None = None, model: str = settings.llm_model
    ) -> None:
        self._client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = model

    async def generate(self, question: str, context_chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(chunk.text for chunk in context_chunks)
        user_message = f"<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>"

        # No `temperature` here on purpose: it's removed for claude-sonnet-5 - passing
        # it (even via extra_body) fails with 400 "temperature is deprecated for this
        # model." Grounding comes from _SYSTEM_PROMPT + disabled thinking instead.
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": user_message}],
        )
        logger.info(
            "Claude usage: model=%s input_tokens=%d output_tokens=%d",
            self._model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return "".join(block.text for block in response.content if block.type == "text")
