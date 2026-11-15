-- pgvector bootstrap for ClinicalNoteRAG.
-- The vector column is intentionally un-dimensioned here so the same table can
-- hold either 384-d MiniLM vectors or 512-d hashing-fallback vectors. The
-- application creates typed indexes when the dimension is known.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    text TEXT NOT NULL,
    patient_id TEXT,
    encounter_id TEXT,
    resource_type TEXT,
    code_system TEXT,
    code TEXT,
    doc_type TEXT,
    service_date DATE,
    claim_id TEXT,
    denied BOOLEAN,
    denial_code TEXT,
    title TEXT,
    section TEXT,
    chunk_index INTEGER,
    metadata JSONB NOT NULL,
    embedding vector NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_patient_idx ON chunks (patient_id);
CREATE INDEX IF NOT EXISTS chunks_encounter_idx ON chunks (encounter_id);
CREATE INDEX IF NOT EXISTS chunks_doc_type_idx ON chunks (doc_type);
CREATE INDEX IF NOT EXISTS chunks_code_idx ON chunks (code_system, code);
CREATE INDEX IF NOT EXISTS chunks_service_date_idx ON chunks (service_date);
