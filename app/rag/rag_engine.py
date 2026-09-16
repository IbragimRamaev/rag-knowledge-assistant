from app.core.interfaces import LLMProvider
from app.rag.retriever import Retriever


class RAGEngine:
    """Orchestrates retrieval + generation to answer a question grounded in the knowledge base."""

    def __init__(self, retriever: Retriever, llm_provider: LLMProvider) -> None:
        self._retriever = retriever
        self._llm_provider = llm_provider

    async def answer(self, question: str, top_k: int = 5, company_id: str | None = None) -> str:
        chunks = await self._retriever.retrieve(question, top_k=top_k, company_id=company_id)
        return await self._llm_provider.generate(question, chunks)
