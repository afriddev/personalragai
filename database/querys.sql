CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE 
        files(
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
            data TEXT NOT NULL
            created_at TIMESTAMP DEFAULT NOW (),
            created_by TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW (),
            updated_by TEXT NOT NULL
        )

CREATE TABLE
    documents (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
        file_id UUID NOT NULL,
        name TEXT NOT NULL,
        url TEXT NOT NULL,
        content_type TEXT NOT NULL,
        extracted_text TEXT DEFAULT NULL,
        description TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW (),
        created_by TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT NOW (),
        updated_by TEXT NOT NULL
    
    )


CREATE TABLE 
        images(
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
            file_id UUID NOT NULL,
            document_id UUID DEFAULT NULL,
            url TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW (),
            created_by TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW (),
            updated_by TEXT NOT NULL

        )










    
--     CREATE INDEX idx_chunkq_vec_hnsw
--     ON chunk_questions USING hnsw (embedding vector_cosine_ops)
-- WITH
--     (m = 16, ef_construction = 300
--     );