import logging
from collections.abc import AsyncIterator

from app.core.interfaces import LLMProvider
from app.core.models import (
    NO_INFORMATION_ANSWER,
    AnswerResult,
    HistoryTurn,
    RetrievedChunk,
    StreamSources,
    StreamTextChunk,
)
from app.rag.query_contextualizer import rewrite_query
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# How many recent turns generation sees for conversational tone. Deliberately small -
# this is not what grounds the answer (retrieval + source_documents is), just what
# keeps replies from sounding like each question lands in a vacuum.
GENERATION_HISTORY_TURNS = 3

# Picked from scripts/smoke_test_rag.py's real off-topic vs. on-topic scores, not guessed:
# off-topic questions scored up to 0.192 ("2+2"), the weakest genuinely relevant match
# scored 0.3442. 0.25 sits roughly at the midpoint, biased slightly toward the on-topic
# side (a real question slipping through still costs one LLM call; a real question wrongly
# rejected costs the answer entirely).
DEFAULT_MIN_SCORE_THRESHOLD = 0.25

# A chunk is only listed in source_documents if its score is within 30% of the top
# chunk's score - retrieval returns top_k chunks regardless of whether they actually
# contributed to the answer, and reporting all of them as "sources" overstates what
# was actually used.
DEFAULT_SOURCE_RELEVANCE_MARGIN = 0.3


class RAGEngine:
    """Orchestrates retrieval + generation to answer a question grounded in the knowledge base."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
        source_relevance_margin: float = DEFAULT_SOURCE_RELEVANCE_MARGIN,
    ) -> None:
        self._retriever = retriever
        self._llm_provider = llm_provider
        self.min_score_threshold = min_score_threshold
        self.source_relevance_margin = source_relevance_margin

    async def _retrieve_grounded_chunks(
        self, question: str, history: list[HistoryTurn], top_k: int, company_id: str | None
    ) -> tuple[list[RetrievedChunk], float] | None:
        """Rewrite + retrieve + threshold check, shared by answer() and answer_stream().
        Returns (chunks, best_score) if retrieval cleared the threshold, None if the
        caller should short-circuit to NO_INFORMATION_ANSWER without calling the LLM."""
        retrieval_query = await rewrite_query(question, history)
        if retrieval_query != question:
            logger.info("Rewrote query for retrieval: %r -> %r", question, retrieval_query)

        chunks = await self._retriever.retrieve(retrieval_query, top_k=top_k, company_id=company_id)

        best_score = chunks[0].score if chunks else None
        if best_score is None or best_score < self.min_score_threshold:
            # Technical safety net on top of the prompt-based one in ClaudeLLMProvider:
            # skip the LLM call entirely for retrieval that isn't even in the right
            # neighborhood, saving tokens on questions we already know we can't answer.
            logger.info(
                "Best retrieval score %s below threshold %.4f for %r; skipping the LLM call",
                f"{best_score:.4f}" if best_score is not None else "n/a",
                self.min_score_threshold,
                retrieval_query,
            )
            return None

        return chunks, best_score

    def _filter_source_documents(
        self, chunks: list[RetrievedChunk], best_score: float
    ) -> list[str]:
        relevance_cutoff = best_score * (1 - self.source_relevance_margin)
        return list(
            dict.fromkeys(
                chunk.source_document for chunk in chunks if chunk.score >= relevance_cutoff
            )
        )

    def _resolve_source_documents(
        self,
        answer_text: str,
        cited_documents: list[str],
        chunks: list[RetrievedChunk],
        best_score: float,
    ) -> list[str]:
        """Decide what to report as source_documents, in priority order:
        1. The model said it doesn't know -> nothing actually grounded it.
        2. The model cited specific documents -> trust that over score alone; it
           reflects what was actually used, not just what was structurally similar.
        3. The model answered but cited nothing (citations are a Sonnet-side
           feature that can simply not fire) -> fall back to the score-margin
           heuristic so a real answer still gets a best-effort attribution."""
        if answer_text.strip() == NO_INFORMATION_ANSWER:
            return []
        if cited_documents:
            return cited_documents

        # Measures how often citations don't fire for a real answer - if this logs
        # often, the margin heuristic is still doing real work, not just covering
        # a rare gap.
        logger.info("No citations for a real answer; falling back to score-margin filtering")
        return self._filter_source_documents(chunks, best_score)

    async def answer(
        self,
        question: str,
        history: list[HistoryTurn] | None = None,
        top_k: int = 5,
        company_id: str | None = None,
    ) -> AnswerResult:
        history = history or []

        retrieved = await self._retrieve_grounded_chunks(question, history, top_k, company_id)
        if retrieved is None:
            return AnswerResult(answer=NO_INFORMATION_ANSWER, source_documents=[])
        chunks, best_score = retrieved

        recent_history = history[-GENERATION_HISTORY_TURNS:]
        result = await self._llm_provider.generate(question, chunks, history=recent_history)

        source_documents = self._resolve_source_documents(
            result.text, result.cited_documents, chunks, best_score
        )
        return AnswerResult(answer=result.text, source_documents=source_documents)

    async def answer_stream(
        self,
        question: str,
        history: list[HistoryTurn] | None = None,
        top_k: int = 5,
        company_id: str | None = None,
    ) -> AsyncIterator[StreamTextChunk | StreamSources]:
        """Same semantics as answer(), yielded incrementally: retrieval happens
        synchronously up front (exactly like answer()), then generation streams as
        text chunks, and source_documents arrives as one final event."""
        history = history or []

        retrieved = await self._retrieve_grounded_chunks(question, history, top_k, company_id)
        if retrieved is None:
            yield StreamTextChunk(text=NO_INFORMATION_ANSWER)
            yield StreamSources(source_documents=[])
            return
        chunks, best_score = retrieved

        recent_history = history[-GENERATION_HISTORY_TURNS:]
        answer_parts = []
        cited_documents: list[str] = []
        async for event in self._llm_provider.generate_stream(
            question, chunks, history=recent_history
        ):
            if isinstance(event, StreamTextChunk):
                answer_parts.append(event.text)
                yield event
            else:
                cited_documents = event.source_documents

        source_documents = self._resolve_source_documents(
            "".join(answer_parts), cited_documents, chunks, best_score
        )
        yield StreamSources(source_documents=source_documents)
