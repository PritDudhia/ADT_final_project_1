-- Multi-table Schema for E-commerce Database with Hybrid SQL+Vector Queries
-- Database Architect: Complete schema with joins, foreign keys, and vector columns

-- Drop existing tables
DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS sellers CASCADE;

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Categories table
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_category_id INTEGER REFERENCES categories(category_id),
    -- Vector embedding for category similarity
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sellers table
CREATE TABLE sellers (
    seller_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    rating DECIMAL(3, 2) CHECK (rating >= 0 AND rating <= 5),
    location VARCHAR(100),
    -- Vector embedding for seller profile
    profile_embedding vector(384),
    joined_date DATE,
    total_sales INTEGER DEFAULT 0,
    verified BOOLEAN DEFAULT FALSE
);

-- Products table (main table)
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    brand VARCHAR(100),
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    category_id INTEGER REFERENCES categories(category_id),
    seller_id INTEGER REFERENCES sellers(seller_id),
    -- Vector embeddings for semantic search
    embedding vector(384),
    image_embedding vector(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    weight_kg DECIMAL(8, 2),
    dimensions VARCHAR(50)
);

-- Customers table
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    location VARCHAR(100),
    -- Vector embedding for customer preferences
    preference_embedding vector(384),
    join_date DATE,
    lifetime_value DECIMAL(12, 2) DEFAULT 0,
    segment VARCHAR(50)
);

-- Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(12, 2),
    status VARCHAR(50) CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled')),
    shipping_address TEXT,
    payment_method VARCHAR(50)
);

-- Order items (junction table with additional attributes)
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL,
    discount_percent DECIMAL(5, 2) DEFAULT 0
);

-- Reviews table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id) ON DELETE CASCADE,
    customer_id INTEGER REFERENCES customers(customer_id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    title VARCHAR(200),
    content TEXT,
    -- Vector embedding for review semantic search
    review_embedding vector(384),
    helpful_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified_purchase BOOLEAN DEFAULT FALSE
);

-- ============================================================================
-- INDEXES FOR SQL OPTIMIZATION
-- ============================================================================

-- B-tree indexes for common SQL queries
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_price ON products(price);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_seller ON products(seller_id);
CREATE INDEX idx_products_stock ON products(stock_quantity);

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(status);

CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);

CREATE INDEX idx_reviews_product ON reviews(product_id);
CREATE INDEX idx_reviews_customer ON reviews(customer_id);
CREATE INDEX idx_reviews_rating ON reviews(rating);

CREATE INDEX idx_customers_segment ON customers(segment);
CREATE INDEX idx_sellers_rating ON sellers(rating);

-- Composite indexes for common join patterns
CREATE INDEX idx_products_brand_price ON products(brand, price);
CREATE INDEX idx_products_category_price ON products(category_id, price);
CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);

-- ============================================================================
-- VECTOR INDEXES FOR SIMILARITY SEARCH
-- ============================================================================

-- HNSW indexes for approximate nearest neighbor search
CREATE INDEX idx_products_embedding_hnsw 
    ON products USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_products_image_embedding_hnsw 
    ON products USING hnsw (image_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_reviews_embedding_hnsw 
    ON reviews USING hnsw (review_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_categories_embedding_hnsw 
    ON categories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_customers_preference_hnsw 
    ON customers USING hnsw (preference_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_sellers_profile_hnsw 
    ON sellers USING hnsw (profile_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- IVFFlat indexes (alternative for larger datasets)
CREATE INDEX idx_products_embedding_ivfflat 
    ON products USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX idx_reviews_embedding_ivfflat 
    ON reviews USING ivfflat (review_embedding vector_cosine_ops)
    WITH (lists = 100);

-- ============================================================================
-- MATERIALIZED VIEWS FOR COMMON AGGREGATIONS
-- ============================================================================

-- Product statistics with review aggregates
CREATE MATERIALIZED VIEW product_stats AS
SELECT 
    p.product_id,
    p.name,
    p.brand,
    p.price,
    p.category_id,
    p.seller_id,
    COUNT(DISTINCT r.review_id) as review_count,
    AVG(r.rating) as avg_rating,
    COUNT(DISTINCT oi.order_id) as order_count,
    SUM(oi.quantity) as total_sold,
    p.embedding,
    p.image_embedding
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id;

CREATE INDEX idx_product_stats_brand ON product_stats(brand);
CREATE INDEX idx_product_stats_category ON product_stats(category_id);
CREATE INDEX idx_product_stats_rating ON product_stats(avg_rating);
CREATE INDEX idx_product_stats_embedding_hnsw 
    ON product_stats USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Customer purchase history summary
CREATE MATERIALIZED VIEW customer_summary AS
SELECT 
    c.customer_id,
    c.name,
    c.segment,
    c.preference_embedding,
    COUNT(DISTINCT o.order_id) as order_count,
    SUM(o.total_amount) as total_spent,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.order_date) as last_order_date,
    COUNT(DISTINCT oi.product_id) as unique_products_bought
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY c.customer_id;

CREATE INDEX idx_customer_summary_segment ON customer_summary(segment);
CREATE INDEX idx_customer_summary_spent ON customer_summary(total_spent);

-- ============================================================================
-- PARTITIONING EXAMPLES (for distributed systems component)
-- ============================================================================

-- Partition orders by date (range partitioning)
-- This would be used in distributed deployment
/*
CREATE TABLE orders_partitioned (
    LIKE orders INCLUDING ALL
) PARTITION BY RANGE (order_date);

CREATE TABLE orders_2024 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE orders_2025 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE orders_2026 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
*/

-- ============================================================================
-- STATISTICS AND CONFIGURATION
-- ============================================================================

-- Analyze tables for query planner
ANALYZE categories;
ANALYZE sellers;
ANALYZE products;
ANALYZE customers;
ANALYZE orders;
ANALYZE order_items;
ANALYZE reviews;

-- Set statistics targets for better cost estimation
ALTER TABLE products ALTER COLUMN brand SET STATISTICS 1000;
ALTER TABLE products ALTER COLUMN price SET STATISTICS 1000;
ALTER TABLE products ALTER COLUMN category_id SET STATISTICS 1000;
ALTER TABLE orders ALTER COLUMN order_date SET STATISTICS 1000;

-- Refresh materialized views
REFRESH MATERIALIZED VIEW product_stats;
REFRESH MATERIALIZED VIEW customer_summary;
