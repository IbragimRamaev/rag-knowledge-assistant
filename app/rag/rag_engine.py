import logging

from app.core.interfaces import LLMProvider
from app.core.models import NO_INFORMATION_ANSWER, AnswerResult, HistoryTurn
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

    async def answer(
        self,
        question: str,
        history: list[HistoryTurn] | None = None,
        top_k: int = 5,
        company_id: str | None = None,
    ) -> AnswerResult:
        history = history or []

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
            return AnswerResult(answer=NO_INFORMATION_ANSWER, source_documents=[])

        recent_history = history[-GENERATION_HISTORY_TURNS:]
        answer_text = await self._llm_provider.generate(question, chunks, history=recent_history)

        relevance_cutoff = best_score * (1 - self.source_relevance_margin)
        source_documents = list(
            dict.fromkeys(
                chunk.source_document for chunk in chunks if chunk.score >= relevance_cutoff
            )
        )
        return AnswerResult(answer=answer_text, source_documents=source_documents)
