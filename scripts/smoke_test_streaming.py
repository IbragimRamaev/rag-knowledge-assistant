"""
Smoke test for Day 18 (streaming responses): hits POST /ask/stream with a real
question and shows the raw SSE events as they arrive - proving the text actually
comes incrementally, not as one chunk at the end - plus the final sources event.

Requires the API running (docker compose -f docker/docker-compose.yml up -d) with
the knowledge base already ingested.

Usage:
    python scripts/smoke_test_streaming.py
"""

import json
import time

import httpx2 as httpx

BASE_URL = "http://localhost:8000"
QUESTION = "Какие расходы компенсируются в командировке?"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        start = time.monotonic()
        first_chunk_at = None
        text_chunks: list[str] = []
        sources: list[str] | None = None

        with client.stream(
            "POST", "/ask/stream", json={"question": QUESTION, "history": []}
        ) as response:
            response.raise_for_status()
            print(f"Q: {QUESTION}\n")
            print("Raw SSE events as they arrive:\n")

            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue

                event = json.loads(line[len("data:") :].strip())

                if event["type"] == "text":
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic() - start
                    text_chunks.append(event["content"])
                    print(f"  [{len(text_chunks):>3}] text  {event['content']!r}")
                elif event["type"] == "sources":
                    sources = event["content"]
                    print(f"  [final] sources  {sources}")
                elif event["type"] == "error":
                    print(f"  [error] {event['content']}")

        elapsed = time.monotonic() - start

        print(f"\nTotal text chunks: {len(text_chunks)}")
        print(
            f"Time to first chunk: {first_chunk_at:.2f}s (retrieval + rewrite happen before this)"
        )
        print(f"Total time: {elapsed:.2f}s")
        print(f"\nFull reassembled answer:\n{''.join(text_chunks)}")
        print(f"\nSources: {sources}")


if __name__ == "__main__":
    main()
