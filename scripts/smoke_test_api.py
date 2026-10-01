"""
Smoke test for Step 5 (API): boot the real FastAPI app under uvicorn as a
subprocess and hit it with real HTTP requests - golden path plus the edge
cases the endpoints are supposed to reject cleanly (empty/too-long question,
DB unavailable).

Requires OPENAI_API_KEY, ANTHROPIC_API_KEY and DATABASE_URL to be set (via
.env), the schema from app/db/schema.sql already applied, and Docker
available on PATH (used to stop/start the Postgres container for the 503
check).

Usage:
    python scripts/smoke_test_api.py
"""

import subprocess
import sys
import time

import httpx2 as httpx

BASE_URL = "http://127.0.0.1:8000"
POSTGRES_CONTAINER = "rag-postgres"

failures: list[str] = []


def check(label: str, condition: bool, extra: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}{(' - ' + extra) if extra else ''}")
    if not condition:
        failures.append(label)


def wait_until_ready(client: httpx.Client, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if client.get("/docs").status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    raise RuntimeError("uvicorn did not become ready in time")


def wait_until_postgres(target_up: bool, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", POSTGRES_CONTAINER],
            capture_output=True,
            text=True,
        )
        running = result.stdout.strip() == "true"
        if running == target_up:
            time.sleep(1.0)  # let asyncpg's pool actually notice the state change
            return
        time.sleep(0.5)
    raise RuntimeError(f"Postgres did not reach running={target_up} in time")


def main() -> None:
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
            wait_until_ready(client)
            print("uvicorn is up\n")

            # --- golden path: /ingest ---
            response = client.post("/ingest", json={"documents_dir": "data/documents"})
            check("/ingest returns 200", response.status_code == 200, f"got {response.status_code}")
            if response.status_code == 200:
                body = response.json()
                print(f"    ingested {body['chunks_ingested']} chunks from {body['documents_dir']}")

            # --- golden path: /ask ---
            question = "Сколько дней отпуска положено сотруднику?"
            response = client.post("/ask", json={"question": question})
            check(
                "/ask returns 200 for a real question",
                response.status_code == 200,
                f"got {response.status_code}",
            )
            if response.status_code == 200:
                body = response.json()
                print(f"    Q: {question}")
                print(f"    A: {body['answer']}")
                print(f"    sources: {body['source_documents']}\n")

            # --- edge case: empty question -> 400 ---
            response = client.post("/ask", json={"question": ""})
            check(
                "empty question returns 400",
                response.status_code == 400,
                f"got {response.status_code}",
            )
            print(f"    detail: {response.json().get('detail')}\n")

            # --- edge case: whitespace-only question -> 400 ---
            response = client.post("/ask", json={"question": "   "})
            check(
                "whitespace-only question returns 400",
                response.status_code == 400,
                f"got {response.status_code}",
            )

            # --- edge case: question over the length limit -> 400 ---
            response = client.post("/ask", json={"question": "a" * 2001})
            check(
                "over-length question returns 400",
                response.status_code == 400,
                f"got {response.status_code}",
            )
            print(f"    detail: {response.json().get('detail')}\n")

            # --- edge case: bad ingest directory -> 400, not 500 ---
            response = client.post("/ingest", json={"documents_dir": "data/does-not-exist"})
            check(
                "ingest on a missing directory returns 400",
                response.status_code == 400,
                f"got {response.status_code}",
            )
            print(f"    detail: {response.json().get('detail')}\n")

            # --- edge case: DB unavailable -> 503, not 500 ---
            print("Stopping Postgres container to test the 503 path...")
            subprocess.run(["docker", "stop", POSTGRES_CONTAINER], check=True, capture_output=True)
            try:
                wait_until_postgres(target_up=False)
                response = client.post("/ask", json={"question": question})
                check(
                    "/ask returns 503 when Postgres is down",
                    response.status_code == 503,
                    f"got {response.status_code}",
                )
                print(f"    detail: {response.json().get('detail')}\n")
            finally:
                print("Restarting Postgres container...")
                subprocess.run(
                    ["docker", "start", POSTGRES_CONTAINER], check=True, capture_output=True
                )
                wait_until_postgres(target_up=True)
                print("Postgres is back up\n")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        output = server.stdout.read() if server.stdout else ""
        if failures:
            print("--- uvicorn output (for debugging failures) ---")
            print(output)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
