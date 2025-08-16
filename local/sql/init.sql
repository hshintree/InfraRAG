-- Initialize database schema for InfraRAG local dev
-- Requires pgvector image; vector extension will be available

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
	document_id TEXT PRIMARY KEY,
	title TEXT,
	document_type TEXT,
	jurisdiction TEXT,
	governing_law TEXT,
	industry TEXT
);

CREATE TABLE IF NOT EXISTS clauses (
	id BIGSERIAL PRIMARY KEY,
	document_id TEXT REFERENCES documents(document_id),
	section_id TEXT,
	title TEXT,
	content TEXT,
	tags TEXT[],
	embedding VECTOR(384)
);

CREATE INDEX IF NOT EXISTS idx_clauses_document_id ON clauses(document_id);
CREATE INDEX IF NOT EXISTS idx_clauses_section_id ON clauses(section_id);
CREATE INDEX IF NOT EXISTS idx_clauses_embedding_ivfflat ON clauses USING ivfflat (embedding vector_l2_ops) WITH (lists = 200); 