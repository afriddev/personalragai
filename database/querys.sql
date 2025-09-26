CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE EXTENSION IF NOT EXISTS vector;




CREATE TABLE 
        chunks(
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
            text TEXT NOT NULL
)

CREATE TABLE 
        nodes(            
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
            summary TEXT NOT NULL
)

CREATE TABLE 
        claims(
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
            chunk_id UUID NOT NULL,
            node_id UUID DEFAULT NULL,
            embedding VECTOR(1024) NOT NULL
)








async with psqlDb.pool.acquire() as conn:
                await conn.set_type_codec(
                    "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
                )
                rows = await conn.fetch(
                    searchRagQuery, cast(Any, queryVector)[0].embedding, 40
                )





-- CREATE TABLE
--     documents (
--         id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
--         data UUID NOT NULL,
--         name TEXT NOT NULL,
--         url TEXT NOT NULL,
--         content_type TEXT NOT NULL,
--         extracted_text TEXT DEFAULT NULL,
--         description TEXT NOT NULL,
--         created_at TIMESTAMP DEFAULT NOW (),
--         created_by TEXT NOT NULL,
--         updated_at TIMESTAMP DEFAULT NOW (),
--         updated_by TEXT NOT NULL    
--     )

-- CREATE TABLE 
--         images(
--             id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
--             document_id UUID DEFAULT NULL,
--             url TEXT NOT NULL,
--             created_at TIMESTAMP DEFAULT NOW (),
--             created_by TEXT NOT NULL,
--             updated_at TIMESTAMP DEFAULT NOW (),
--             updated_by TEXT NOT NULL
--         )

-- CREATE TABLE 
--         chunks(
--             id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
--             document_id UUID NOT NULL,
--             chunk_text TEXT NOT NULL,
--             chunk_index INT NOT NULL,
--             created_at TIMESTAMP DEFAULT NOW (),
--             created_by TEXT NOT NULL,
--             updated_at TIMESTAMP DEFAULT NOW (),
--             updated_by TEXT NOT NULL
-- )

-- CREATE TABLE 
--         nodes(            
--             id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
--             summary TEXT NOT NULL,
--             created_at TIMESTAMP DEFAULT NOW (),
--             created_by TEXT NOT NULL,
--             updated_at TIMESTAMP DEFAULT NOW (),
--             updated_by TEXT NOT NULL
-- )

-- CREATE TABLE 
--         chunk_entities(
--             id UUID PRIMARY KEY DEFAULT uuid_generate_v4 (),
--             chunk_id UUID NOT NULL,
--             node_id UUID DEFAULT NULL,
--             entity_description TEXT NOT NULL,
--             embedding VECTOR(1024) NOT NULL,
--             created_at TIMESTAMP DEFAULT NOW (),
--             created_by TEXT NOT NULL,
--             updated_at TIMESTAMP DEFAULT NOW (),
--             updated_by TEXT NOT NULL
-- )


    
--     CREATE INDEX idx_chunkq_vec_hnsw
--     ON chunk_questions USING hnsw (embedding vector_cosine_ops)
-- WITH
--     (m = 16, ef_construction = 300
--     );