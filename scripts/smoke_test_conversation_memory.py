"""
Smoke test for Day 17 (conversation memory + LLM-based query rewriting): walks a
real multi-turn conversation through POST /ask, exactly as the frontend would,
ending in a pronoun-referencing follow-up several turns after the topic it refers
to. A last-message heuristic would rewrite it against the wrong (most recent)
topic; LLM-based rewriting with the full history should resolve it correctly.

The actual rewritten query and Haiku token usage aren't in the HTTP response (by
design - that's an internal retrieval detail, not part of the public API contract),
so this script also prints the relevant `docker compose logs app` lines for the
final turn.

Requires the API running (docker compose -f docker/docker-compose.yml up -d) with
the knowledge base already ingested.

Usage:
    python scripts/smoke_test_conversation_memory.py
"""

import subprocess

import httpx2 as httpx

BASE_URL = "http://localhost:8000"

TURNS = [
    "Сколько дней отпуска положено сотруднику?",
    "Что делать, если я получил подозрительное фишинговое письмо?",
    "Какие расходы компенсируются в командировке?",
    # Deliberately anchored to turn 1's specific wording ("частями", "дней подряд"),
    # not to the immediately preceding topic - a last-message-only heuristic has
    # nothing in the expense-reimbursement turn to tie this to, and would either
    # misfire against it or fail outright.
    "А можно взять все эти дни сразу, одним блоком, а не частями?",
]


def main() -> None:
    history: list[dict] = []

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        for question in TURNS:
            response = client.post("/ask", json={"question": question, "history": history})
            response.raise_for_status()
            data = response.json()

            print(f"Q: {question}")
            print(f"A: {data['answer']}")
            print(f"   sources: {data['source_documents']}\n")

            history.append({"question": question, "answer": data["answer"]})

    print("--- relevant container log lines for the final (follow-up) turn ---")
    logs = subprocess.run(
        ["docker", "compose", "-f", "docker/docker-compose.yml", "logs", "app", "--tail", "50"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in logs.splitlines():
        if "Query rewrite" in line or "Best retrieval score" in line:
            print(line)


if __name__ == "__main__":
    main()
