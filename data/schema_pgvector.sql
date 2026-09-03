-- Diyetisyen Klinik Karar Destek Sistemi (CDSS)
-- PostgreSQL + pgvector Veri Tabanı Şeması ve Geçiş Planı (Faz 2)

-- 1. pgvector eklentisini aktif et
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Klinik Kılavuzlar ve Makaleler Üst Tablosu
CREATE TABLE IF NOT EXISTS clinical_guidelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    organization VARCHAR(100) NOT NULL, -- Örn: ADA, ESPEN, EASD, WHO
    publication_year INT NOT NULL,
    domain VARCHAR(100) NOT NULL,       -- Örn: endocrinology, hematology, nephrology
    source_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Parçalanmış ve Vektörleştirilmiş Pasajlar (Chunks) Tablosu
CREATE TABLE IF NOT EXISTS guideline_chunks (
    id VARCHAR(100) PRIMARY KEY,        -- Örn: ADA_2024_P14, WHO_IDA_2023_C4
    guideline_id UUID REFERENCES clinical_guidelines(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    evidence_level VARCHAR(10),         -- Örn: Level A, Level B, Expert Consensus
    disease_tag VARCHAR(50) NOT NULL,   -- Örn: t2dm, anemia, ckd, hypertension
    tags TEXT[],                        -- Örn: ARRAY['insulin_resistance', 'fiber', 'glycemic_index']
    embedding vector(384),              -- all-MiniLM-L6-v2 için 384, BAAI/bge-m3 için 1024 boyutlu vektör
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Hızlı Vektör Benzerlik Arama İndeksi (HNSW - Cosine Distance)
CREATE INDEX IF NOT EXISTS idx_guideline_chunks_embedding 
ON guideline_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5. Hibrit Arama için PostgreSQL Full-Text Search (Sparse BM25 Dengi) İndeksi
CREATE INDEX IF NOT EXISTS idx_guideline_chunks_fts 
ON guideline_chunks 
USING gin (to_tsvector('turkish', content));

-- 6. Hasta Değerlendirme ve Diyetisyen Karar Geçmişi Tablosu
CREATE TABLE IF NOT EXISTS patient_evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(50) NOT NULL,
    lab_data JSONB NOT NULL,
    active_constraints JSONB NOT NULL,
    retrieved_citations JSONB NOT NULL,
    llm_recommendations JSONB NOT NULL,
    dietitian_notes TEXT,
    status VARCHAR(50) DEFAULT 'PENDING_DIETITIAN_REVIEW', -- PENDING, APPROVED, EDITED_AND_APPROVED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    approved_at TIMESTAMP WITH TIME ZONE
);
