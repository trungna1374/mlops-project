-- Create database schema for e-commerce MLOps

-- Create MLflow schema for experiment tracking
CREATE SCHEMA IF NOT EXISTS mlflow;
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO mlflow;

-- Create public schema tables (business data)
-- Drop tables if they exist
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Create customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL,
    customer_type VARCHAR(50) NOT NULL CHECK (customer_type IN ('regular', 'premium', 'vip')),
    location VARCHAR(50) NOT NULL CHECK (location IN ('US', 'EU', 'ASIA', 'AU')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    amount DECIMAL(10, 2) NOT NULL,
    quantity INTEGER NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_customers_type ON customers(customer_type);

-- Insert sample customers
INSERT INTO customers (customer_name, customer_type, location) VALUES
    ('John Doe', 'regular', 'US'),
    ('Jane Smith', 'premium', 'EU'),
    ('Bob Johnson', 'vip', 'ASIA'),
    ('Alice Williams', 'regular', 'US'),
    ('Charlie Brown', 'premium', 'EU'),
    ('Diana Prince', 'vip', 'US'),
    ('Eve Davis', 'regular', 'ASIA'),
    ('Frank Miller', 'premium', 'AU'),
    ('Grace Lee', 'vip', 'EU'),
    ('Henry Wilson', 'regular', 'US');

-- Insert sample transactions (last 90 days)
INSERT INTO transactions (customer_id, amount, quantity, product_category, timestamp)
SELECT 
    (random() * 9 + 1)::int AS customer_id,
    (random() * 400 + 10)::decimal(10,2) AS amount,
    (random() * 9 + 1)::int AS quantity,
    (ARRAY['electronics', 'clothing', 'food', 'books', 'sports'])[floor(random() * 5 + 1)] AS product_category,
    NOW() - (random() * interval '90 days') AS timestamp
FROM generate_series(1, 1000);

-- Verify data
SELECT 'Customers created:' AS info, COUNT(*) AS count FROM customers
UNION ALL
SELECT 'Transactions created:', COUNT(*) FROM transactions;

-- Show sample data
SELECT 'Sample transactions:' AS info;
SELECT t.transaction_id, c.customer_name, t.amount, t.quantity, t.product_category, t.timestamp
FROM transactions t
JOIN customers c ON t.customer_id = c.customer_id
ORDER BY t.timestamp DESC
LIMIT 10;










