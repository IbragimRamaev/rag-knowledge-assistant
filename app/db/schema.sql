-- Step 3 (Storage) schema for the `chunks` table. No Alembic yet — run this
-- by hand whenever the schema changes.
--
-- How to run (from the project root, against the local docker-compose Postgres):
--   docker exec -i rag-postgres psql -U postgres -d rag_db < app/db/schema.sql
--
-- Or interactively, once connected via `docker exec -it rag-postgres psql -U postgres -d rag_db`,
-- paste the statements below directly into the psql prompt.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    -- Multi-tenancy from day one: every row is scoped to a company, even though
    -- there is a single tenant today. Defaults to the current DEFAULT_COMPANY_ID.
    company_id TEXT NOT NULL DEFAULT 'demo-company',
    source_document TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Re-ingesting the same document should update chunks in place, not duplicate them.
    UNIQUE (company_id, source_document, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_company_id_idx ON chunks (company_id);

-- No vector similarity index (ivfflat/hnsw) yet: the dataset is tiny and a plain
-- sequential scan over embedding <=> query is fast enough. Add one once retrieval
-- latency actually becomes a problem.
