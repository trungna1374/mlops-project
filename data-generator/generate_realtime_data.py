#!/usr/bin/env python3
import psycopg2
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import time
import random
import os

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        dbname=os.getenv('POSTGRES_DB', 'ecommerce_db'),
        user=os.getenv('POSTGRES_USER', 'mlflow'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlflow')
    )

def generate_transaction():
    fake = Faker()
    date_range_days = int(os.getenv('DATE_RANGE_DAYS', '90'))

    # Generate random date within last N days
    days_ago = random.randint(0, date_range_days)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=random.randint(0,23), minutes=random.randint(0,59))
    
    # Product categories from the model
    categories = ['electronics', 'clothing', 'food', 'books', 'sports', 'home', 'toys']
    
    # Get or create a customer (use existing customer IDs from database)
    # For simplicity, use customer IDs 12000-20000 (from Kaggle data range)
    customer_id = random.randint(12000, 20000)

    return {
        'invoice_no': f'INV{random.randint(500000,999999)}',
        'customer_id': customer_id,
        'amount': round(random.uniform(10, 500), 2),
        'quantity': random.randint(1, 10),
        'product_category': random.choice(categories),
        'product_description': fake.catch_phrase(),
        'timestamp': timestamp
    }

def main():
    print(" Starting data generator...")

    while True:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Generate and insert transaction
            transaction = generate_transaction()

            cursor.execute("""
                INSERT INTO transactions (invoice_no, customer_id, amount, quantity, product_category, product_description, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                transaction['invoice_no'],
                transaction['customer_id'],
                transaction['amount'],
                transaction['quantity'],
                transaction['product_category'],
                transaction['product_description'],
                transaction['timestamp']
            ))

            conn.commit()
            print(f"Inserted transaction: Customer {transaction['customer_id']} - ${transaction['amount']:.2f} - {transaction['product_category']}")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f" Error: {e}")

        # Wait before next transaction
        time.sleep(int(os.getenv('GENERATION_INTERVAL', '5')))

if __name__ == "__main__":
    main()
