import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.config import settings
from app.core.models import HistoryTurn
from app.rag.query_contextualizer import rewrite_query

QUESTION = "Что делать, если я получил подозрительное фишинговое письмо?"
HISTORY = [
    HistoryTurn(
        question="Сколько дней отпуска положено сотруднику?",
        answer="Сотруднику полагается 28 календарных дней оплачиваемого отпуска в год.",
    )
]


def _fake_client(response_text: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(type="text", text=response_text)]
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_rewrite_query_returns_question_unchanged_when_history_empty():
    # No client passed - if this tried to call the API it would blow up with no
    # credentials, proving the empty-history short circuit runs before any request.
    result = await rewrite_query(QUESTION, [], client=None)
    assert result == QUESTION


@pytest.mark.asyncio
async def test_rewrite_query_uses_the_model_rewrite_when_it_looks_like_a_question():
    rewritten = "Что нужно делать при получении подозрительного фишингового письма?"
    client = _fake_client(rewritten)
    result = await rewrite_query(QUESTION, HISTORY, client=client)
    assert result == rewritten


@pytest.mark.asyncio
async def test_rewrite_query_falls_back_when_the_model_answers_instead_of_rewriting(caplog):
    """
    Deterministic regression test for a real failure mode found in manual testing
    (Day 17): with a short prior history, Haiku would sometimes answer this exact
    question instead of rewriting it - a multi-line, markdown-formatted response -
    which would otherwise get embedded as the retrieval query verbatim. This
    reproduces that exact response via a fake client so the fallback is verified
    every run, not only on the rare occasions the real model happens to misbehave.
    """
    derailed_response = (
        "Если вы получили подозрительное фишинговое письмо, вот что нужно делать:\n\n"
        "1. **Не открывайте вложения и не переходите по ссылкам**\n"
        "2. **Сообщите в IT-отдел**\n"
    )
    client = _fake_client(derailed_response)

    with caplog.at_level(logging.WARNING, logger="app.rag.query_contextualizer"):
        result = await rewrite_query(QUESTION, HISTORY, client=client)

    assert result == QUESTION  # fell back to the original, not the derailed response
    assert any("derailed" in record.message for record in caplog.records)


@pytest.mark.skipif(
    not settings.anthropic_api_key,
    reason="requires a real ANTHROPIC_API_KEY - hits the live Haiku API",
)
@pytest.mark.asyncio
async def test_rewrite_query_real_api_stays_question_shaped_across_repeated_calls(caplog):
    """
    Real-API regression test for the same failure mode (see the mocked test above),
    run repeatedly against the live model - not mocked - to track the actual derail
    rate over time. Skipped wherever ANTHROPIC_API_KEY isn't set (e.g. CI); run
    locally with a real .env to exercise it for real.
    """
    attempts = 15
    fallback_triggers = 0

    for _ in range(attempts):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="app.rag.query_contextualizer"):
            result = await rewrite_query(QUESTION, HISTORY)

        # The contract rewrite_query must uphold no matter what the model does
        # internally: a single line, roughly question-sized, never empty. This is
        # what actually protects retrieval quality - not the model behaving.
        assert result
        assert "\n" not in result
        assert len(result) <= max(len(QUESTION) * 4, 120)

        if any("derailed" in record.message for record in caplog.records):
            fallback_triggers += 1

    print(f"\nreal-API derail rate this run: {fallback_triggers}/{attempts} calls fell back")
