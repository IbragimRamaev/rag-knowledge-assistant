"""
Integration smoke test for the full RAG pipeline (Week 2 wrap-up): ingest the
full knowledge base, then ask RAGEngine real questions covering all 7
documents plus a few off-topic ones, and print a results table - what
document each question expected, what was actually retrieved, the top
similarity score, and whether the score threshold short-circuited the LLM
call.

Requires OPENAI_API_KEY, ANTHROPIC_API_KEY and DATABASE_URL to be set (via
.env), and the schema from app/db/schema.sql to already be applied.

Usage:
    python scripts/smoke_test_rag.py
"""

import asyncio
from dataclasses import dataclass

from app.config import settings
from app.ingestion.pipeline import ingest_documents
from app.providers.claude_llm import ClaudeLLMProvider
from app.providers.openai_embedding import OpenAIEmbeddingProvider
from app.rag.rag_engine import RAGEngine
from app.rag.retriever import Retriever
from app.storage.pg_vector_store import PgVectorStore

DOCUMENTS_DIR = "data/documents"


@dataclass
class Case:
    question: str
    expected_document: str | None  # None marks an off-topic question


CASES = [
    # --- company_policy.md ---
    Case("Сколько дней отпуска положено сотруднику?", "company_policy.md"),
    Case("Что нужно вернуть при увольнении и в какой срок?", "company_policy.md"),
    # --- benefits_overview.md ---
    Case("Компания компенсирует абонемент в спортзал?", "benefits_overview.md"),
    Case(
        "Сколько оплачиваемых дней даётся в случае свадьбы или рождения ребёнка?",
        "benefits_overview.md",
    ),
    # --- code_of_conduct.md ---
    Case("Куда обращаться, если я стал свидетелем домогательств на работе?", "code_of_conduct.md"),
    Case("Что считается конфликтом интересов и как о нём сообщить?", "code_of_conduct.md"),
    # --- expense_reimbursement.md ---
    Case("Какие расходы компенсируются в командировке?", "expense_reimbursement.md"),
    Case(
        "В течение какого срока нужно подать заявку на возмещение расходов?",
        "expense_reimbursement.md",
    ),
    # --- it_security_policy.md ---
    Case("Что делать, если я получил подозрительное фишинговое письмо?", "it_security_policy.md"),
    Case("Как часто нужно менять пароль от корпоративных систем?", "it_security_policy.md"),
    # --- onboarding_guide.md ---
    Case("Кто такой buddy и чем он занимается в первые недели новичка?", "onboarding_guide.md"),
    # --- remote_work_policy.md ---
    Case("Можно ли работать удалённо через публичный Wi-Fi без VPN?", "remote_work_policy.md"),
    # --- off-topic: nothing in the knowledge base should match these ---
    Case("Какая столица Франции?", None),
    Case("Сколько будет 2+2?", None),
    Case("Какая погода завтра в Бишкеке?", None),
]


async def main() -> None:
    embedding_provider = OpenAIEmbeddingProvider()

    chunks = await ingest_documents(DOCUMENTS_DIR, embedding_provider)
    print(f"Ingested {len(chunks)} chunks from {DOCUMENTS_DIR}")

    store = PgVectorStore(settings.database_url)
    try:
        await store.save_chunks(chunks)
        print(f"Saved {len(chunks)} chunks (company_id={settings.default_company_id})\n")

        retriever = Retriever(embedding_provider, store)
        engine = RAGEngine(retriever=retriever, llm_provider=ClaudeLLMProvider())
        print(f"min_score_threshold = {engine.min_score_threshold}\n")

        rows = []
        for case in CASES:
            # Retrieve separately (in addition to engine.answer()'s own internal
            # retrieval) purely to report the top score/document in the table below.
            retrieved = await retriever.retrieve(case.question, top_k=3)
            top_score = retrieved[0].score if retrieved else None
            top_document = retrieved[0].source_document if retrieved else "-"

            result = await engine.answer(case.question)
            # Ground truth for "did the technical safety net skip the LLM call": the same
            # condition RAGEngine.answer() checks internally. Comparing result.answer to
            # NO_INFORMATION_ANSWER would be wrong - Claude's prompt-based refusal can
            # produce that exact same string *after* being called, which isn't the same thing.
            threshold_fired = top_score is None or top_score < engine.min_score_threshold
            rows.append((case, top_score, top_document, result, threshold_fired))

            print(f"Q: {case.question}")
            print(f"   expected: {case.expected_document or '(off-topic)'}")
            if top_score is not None:
                print(f"   top match: {top_document} (score={top_score:.4f})")
            else:
                print("   top match: none")
            print(f"   A: {result.answer}")
            print(f"   sources: {result.source_documents}\n")

        # --- summary table ---
        columns = ("question", "expected", "found", "score", "threshold?")
        widths = (55, 22, 22, 7, 10)

        def row_line(question: str, expected: str, found: str, score: str, fired: str) -> str:
            question = question if len(question) <= widths[0] else question[: widths[0] - 3] + "..."
            cells = (question, expected, found, score, fired)
            return " | ".join(cell.ljust(w) for cell, w in zip(cells, widths))

        header = row_line(*columns)
        print(header)
        print("-" * len(header))
        for case, top_score, top_document, _result, threshold_fired in rows:
            expected = case.expected_document or "(off-topic)"
            score_str = f"{top_score:.4f}" if top_score is not None else "n/a"
            fired = "YES" if threshold_fired else "no"
            print(row_line(case.question, expected, top_document, score_str, fired))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
