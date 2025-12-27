#!/usr/bin/env python3
"""
User Behavior Simulator
Simulates user interactions with the ML prediction API
"""
import os
import time
import random
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
API_URL = os.getenv('API_URL', 'http://api:8000')
INTERVAL = int(os.getenv('SIMULATION_INTERVAL', '10'))  # seconds

# Sample data for simulation - must match training data categories
CATEGORIES = ['electronics', 'clothing', 'books', 'home', 'food', 'sports', 'toys',
             'stationery', 'accessories', 'seasonal', 'general', 'gifts', 'home_decor']
CUSTOMER_TYPES = ['regular', 'premium', 'vip']
LOCATIONS = ['United Kingdom', 'Germany', 'France', 'Australia', 'Spain', 'Netherlands',
             'Belgium', 'Switzerland', 'Portugal', 'Italy', 'Finland', 'Norway', 'Denmark',
             'Austria', 'Poland', 'Sweden', 'Japan', 'USA', 'Canada']

def generate_transaction():
    """Generate a random transaction"""
    return {
        'amount': round(random.uniform(10, 500), 2),
        'quantity': random.randint(1, 10),
        'product_category': random.choice(CATEGORIES),
        'customer_type': random.choice(CUSTOMER_TYPES),
        'location': random.choice(LOCATIONS)
    }

def simulate_user_behavior():
    """Simulate continuous user behavior"""
    logger.info(f"🎭 Starting user behavior simulation...")
    logger.info(f"Target API: {API_URL}")
    logger.info(f"Interval: {INTERVAL}s")
    
    request_count = 0
    success_count = 0
    
    while True:
        try:
            # Generate transaction
            transaction = generate_transaction()
            request_count += 1
            
            # Make prediction request
            response = requests.post(
                f"{API_URL}/predict",
                json=transaction,
                timeout=5
            )
            
            if response.status_code == 200:
                success_count += 1
                result = response.json()
                logger.info(
                    f"✅ Request #{request_count} | "
                    f"Amount: ${transaction['amount']:.2f} | "
                    f"Prediction: {result.get('prediction')} | "
                    f"Probability: {result.get('probability', 0):.2f}"
                )
            else:
                logger.warning(
                    f"⚠️ Request #{request_count} failed | "
                    f"Status: {response.status_code}"
                )
            
            # Log stats every 10 requests
            if request_count % 10 == 0:
                success_rate = (success_count / request_count) * 100
                logger.info(
                    f"📊 Stats: {success_count}/{request_count} successful "
                    f"({success_rate:.1f}%)"
                )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
        
        # Wait before next request
        time.sleep(INTERVAL)

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎭 User Behavior Simulator")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    try:
        simulate_user_behavior()
    except KeyboardInterrupt:
        logger.info("\n👋 Simulator stopped by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        raise







