import logging

from app.core.interfaces import LLMProvider
from app.core.models import NO_INFORMATION_ANSWER, AnswerResult
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# Picked from scripts/smoke_test_rag.py's real off-topic vs. on-topic scores, not guessed:
# off-topic questions scored up to 0.192 ("2+2"), the weakest genuinely relevant match
# scored 0.3442. 0.25 sits roughly at the midpoint, biased slightly toward the on-topic
# side (a real question slipping through still costs one LLM call; a real question wrongly
# rejected costs the answer entirely).
DEFAULT_MIN_SCORE_THRESHOLD = 0.25


class RAGEngine:
    """Orchestrates retrieval + generation to answer a question grounded in the knowledge base."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        min_score_threshold: float = DEFAULT_MIN_SCORE_THRESHOLD,
    ) -> None:
        self._retriever = retriever
        self._llm_provider = llm_provider
        self.min_score_threshold = min_score_threshold

    async def answer(
        self, question: str, top_k: int = 5, company_id: str | None = None
    ) -> AnswerResult:
        chunks = await self._retriever.retrieve(question, top_k=top_k, company_id=company_id)

        best_score = chunks[0].score if chunks else None
        if best_score is None or best_score < self.min_score_threshold:
            # Technical safety net on top of the prompt-based one in ClaudeLLMProvider:
            # skip the LLM call entirely for retrieval that isn't even in the right
            # neighborhood, saving tokens on questions we already know we can't answer.
            logger.info(
                "Best retrieval score %s below threshold %.4f for %r; skipping the LLM call",
                f"{best_score:.4f}" if best_score is not None else "n/a",
                self.min_score_threshold,
                question,
            )
            return AnswerResult(answer=NO_INFORMATION_ANSWER, source_documents=[])

        answer_text = await self._llm_provider.generate(question, chunks)
        source_documents = list(dict.fromkeys(chunk.source_document for chunk in chunks))
        return AnswerResult(answer=answer_text, source_documents=source_documents)
