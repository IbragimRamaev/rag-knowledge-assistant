import logging

from anthropic import AsyncAnthropic

from app.config import settings
from app.core.models import HistoryTurn

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Переформулируй последний вопрос пользователя в самостоятельный, понятный вне "
    "контекста диалога. Если вопрос уже самостоятельный — верни как есть. Верни "
    "ТОЛЬКО переформулированный вопрос, без пояснений. Не отвечай на вопрос "
    "пользователя и не давай инструкций по теме вопроса — даже если вопрос сам "
    "просит объяснение или инструкцию, твоя задача только переформулировать его, "
    "а не отвечать на него."
)

_MODEL = "claude-haiku-4-5"
# Generous for a single reformulated question, but deliberately tight enough that a
# derailed response (answering the question instead of rewriting it, seen once in
# testing) gets cut off fast rather than burning tokens on a full answer.
_MAX_TOKENS = 100

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


async def rewrite_query(
    question: str, history: list[HistoryTurn], client: AsyncAnthropic | None = None
) -> str:
    """
    Resolve references to earlier turns (e.g. "does that apply to ... too?") into a
    standalone query suitable for embedding. With no history, returns `question`
    unchanged - there's nothing to resolve and no reason to spend an LLM call on it.
    """
    if not history:
        return question

    messages = []
    for turn in history:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer})
    messages.append({"role": "user", "content": question})

    client = client or _get_client()
    # No `thinking` param: Haiku 4.5 only takes the old enabled+budget_tokens form,
    # not `disabled` - omitting it entirely is the equivalent "no thinking" for this
    # simple a task.
    response = await client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )

    rewritten = "".join(block.text for block in response.content if block.type == "text").strip()

    logger.info(
        "Query rewrite: model=%s input_tokens=%d output_tokens=%d %r -> %r",
        _MODEL,
        response.usage.input_tokens,
        response.usage.output_tokens,
        question,
        rewritten,
    )

    # Observed once in testing: the model answered the question instead of rewriting
    # it (a multi-line, markdown-formatted response). A rewritten question is one
    # line and roughly question-sized - anything wildly longer or multi-line isn't
    # a query anymore, so fall back to the original rather than embedding junk.
    if not rewritten or "\n" in rewritten or len(rewritten) > max(len(question) * 4, 120):
        logger.warning(
            "Query rewrite looked like a derailed response, not a rewritten "
            "question; falling back to the original: %r",
            rewritten,
        )
        return question

    return rewritten
