-- Setup script for Cost-Based Optimization for Hybrid SQL + Vector Queries
-- PostgreSQL 16+ with pgvector extension

-- Drop existing database if exists (for clean setup)
DROP DATABASE IF EXISTS hybrid_query_db;

-- Create database
CREATE DATABASE hybrid_query_db;

-- Connect to the database
\c hybrid_query_db

-- Install pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create products table (E-commerce dataset)
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    price DECIMAL(10, 2),
    brand TEXT,
    category TEXT,
    rating DECIMAL(3, 2),
    num_reviews INTEGER,
    in_stock BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(768)  -- BERT embedding dimension
);

-- Create research papers table (Academic dataset)
CREATE TABLE research_papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT[],
    year INTEGER,
    venue TEXT,
    citations INTEGER,
    keywords TEXT[],
    arxiv_id TEXT,
    embedding vector(768)  -- SciBERT embedding dimension
);

-- Create images table (Image search dataset)
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    description TEXT,
    tags TEXT[],
    width INTEGER,
    height INTEGER,
    format TEXT,
    file_size INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(512)  -- CLIP embedding dimension
);

-- Create B-tree indexes (for SQL filters)
-- Products indexes
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_rating ON products(rating);

-- Research papers indexes
CREATE INDEX idx_papers_year ON research_papers(year);
CREATE INDEX idx_papers_venue ON research_papers(venue);
CREATE INDEX idx_papers_citations ON research_papers(citations);

-- Images indexes
CREATE INDEX idx_images_format ON images(format);
CREATE INDEX idx_images_tags ON images USING GIN(tags);

-- Create vector indexes (for similarity search)
-- HNSW indexes (more accurate, better for high-dimensional vectors)
CREATE INDEX idx_products_embedding_hnsw ON products 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_papers_embedding_hnsw ON research_papers 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_images_embedding_hnsw ON images 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Alternative: IVFFlat indexes (faster build, approximate)
-- Uncomment if you want to test IVFFlat instead of HNSW
/*
CREATE INDEX idx_products_embedding_ivf ON products 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_papers_embedding_ivf ON research_papers 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_images_embedding_ivf ON images 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
*/

-- Create statistics table (for cost model calibration)
CREATE TABLE query_statistics (
    id SERIAL PRIMARY KEY,
    table_name TEXT,
    index_name TEXT,
    index_type TEXT,  -- 'btree', 'hnsw', 'ivfflat'
    num_rows INTEGER,
    num_distinct_values INTEGER,
    avg_row_size INTEGER,
    index_size_bytes BIGINT,
    avg_search_time_ms DECIMAL(10, 3),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create query execution log (for optimizer learning)
CREATE TABLE execution_log (
    id SERIAL PRIMARY KEY,
    query_hash TEXT,
    query_text TEXT,
    plan_type TEXT,  -- 'filter_first', 'vector_first', 'hybrid'
    estimated_cost DECIMAL(12, 3),
    actual_runtime_ms DECIMAL(10, 3),
    rows_returned INTEGER,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions (adjust user as needed)
GRANT ALL PRIVILEGES ON DATABASE hybrid_query_db TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Enable timing for query analysis
\timing on

-- Display setup summary
SELECT 
    'Setup Complete!' as status,
    version() as postgres_version;

SELECT 
    extname as extension,
    extversion as version
FROM pg_extension 
WHERE extname = 'vector';

\echo 'Database setup complete!'
\echo 'Next steps:'
\echo '  1. Run: python scripts/load_data.py'
\echo '  2. Run: python benchmarks/run_experiments.py'
