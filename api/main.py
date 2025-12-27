from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import os
import mlflow
import mlflow.sklearn
import numpy as np
import logging
import tempfile
import requests
import zipfile
import shutil
import json
import pandas as pd
import time
from typing import Dict, List
import psycopg2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLOps API", version="2.0.0")

# Global model cache
model = None
model_version = None
model_run_id = None
feature_columns = None  # Store feature column information
preprocessing_info = None

# Metrics collection for monitoring
api_metrics = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_error": 0,
    "response_times": [],
    "predictions_high_value": 0,
    "predictions_low_value": 0,
    "last_request_time": None
}

def log_metrics_to_db(request_data: dict, response_data: dict, response_time: float):
    """Log API metrics to PostgreSQL for Grafana monitoring"""
    # Try multiple PostgreSQL hosts (same logic as feature engineering)
    postgres_hosts = [
        os.getenv('POSTGRES_HOST', 'localhost'),
        'host.minikube.internal',  # Minikube special hostname
        '192.168.49.1',  # Minikube host IP
        'localhost',
        'host.docker.internal'
    ]
    
    conn = None
    for pg_host in postgres_hosts:
        try:
            conn = psycopg2.connect(
                host=pg_host,
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                database=os.getenv('POSTGRES_DB', 'ecommerce_db'),
                user=os.getenv('POSTGRES_USER', 'mlflow'),
                password=os.getenv('POSTGRES_PASSWORD', 'mlflow'),
                connect_timeout=3  # 3 second timeout per host
            )
            logger.debug(f"✅ Connected to PostgreSQL at {pg_host} for metrics logging")
            break
        except Exception as conn_error:
            logger.debug(f"Failed to connect to PostgreSQL at {pg_host}: {conn_error}")
            continue
    
    if conn is None:
        logger.warning("Failed to log metrics to database: Could not connect to PostgreSQL from any host")
        return
    
    try:
        cursor = conn.cursor()

        # Create metrics table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_metrics (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                endpoint VARCHAR(50),
                request_amount DECIMAL(10,2),
                request_quantity INTEGER,
                request_category VARCHAR(100),
                request_customer_type VARCHAR(50),
                request_location VARCHAR(100),
                response_prediction INTEGER,
                response_probability DECIMAL(5,4),
                response_time_ms DECIMAL(8,2),
                model_version VARCHAR(50),
                model_run_id VARCHAR(100)
            )
        """)

        # Insert metrics
        cursor.execute("""
            INSERT INTO api_metrics
            (endpoint, request_amount, request_quantity, request_category,
             request_customer_type, request_location, response_prediction,
             response_probability, response_time_ms, model_version, model_run_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            'predict',
            request_data.get('amount'),
            request_data.get('quantity'),
            request_data.get('product_category'),
            request_data.get('customer_type'),
            request_data.get('location'),
            response_data.get('prediction'),
            response_data.get('probability'),
            response_time * 1000,  # Convert to milliseconds
            response_data.get('model_version'),
            response_data.get('model_run_id')
        ))

        conn.commit()
        cursor.close()
        conn.close()
        logger.debug("✅ Metrics logged to database successfully")

    except Exception as e:
        logger.warning(f"Failed to log metrics to database: {e}")
        # Continue execution even if metrics logging fails
        try:
            if conn:
                conn.close()
        except:
            pass

class PredictionRequest(BaseModel):
    amount: float
    quantity: int
    product_category: str = "electronics"
    customer_type: str = "regular"
    location: str = "US"

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {'example': {
            'amount': 150.0,
            'quantity': 2,
            'product_category': 'electronics',
            'customer_type': 'premium',
            'location': 'US'
        }}
    }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    model_run_id: str
    timestamp: str

    model_config = {
        'protected_namespaces': (),
        'json_schema_extra': {'example': {
            'prediction': 1,
            'probability': 0.85,
            'model_version': '1.0',
            'model_run_id': 'abc123',
            'timestamp': '2025-01-01T12:00:00'
        }}
    }

def load_model():
    """Load model from MLflow"""
    global model, model_version, model_run_id
    
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    model_name = os.getenv('MODEL_NAME', 'ecommerce_classifier')
    model_tag = os.getenv('MODEL_TAG', 'latest')  # Can be 'latest', version number, or run_id
    
    # Configure S3 (MinIO) for artifact access
    # These are set via environment variables in Kubernetes
    s3_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    if s3_endpoint:
        logger.info(f"S3 (MinIO) configured: {s3_endpoint}")
    
    logger.info(f"Loading model: {model_name} with tag: {model_tag}")
    
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        client = mlflow.tracking.MlflowClient()
        
        # Determine the run_id to use
        if model_tag == 'latest':
            # Get latest run_id from experiment
            try:
                experiment = client.get_experiment_by_name("ecommerce_model_training")
                if experiment:
                    runs = client.search_runs(experiment.experiment_id, max_results=1, order_by=['start_time DESC'])
                    if runs:
                        run_id = runs[0].info.run_id
                        logger.info(f"Loading latest model from run_id: {run_id}")
                    else:
                        # Fallback to model registry
                        model_uri = f"models:/{model_name}/latest"
                        logger.info(f"Loading latest model version from registry")
                        model = mlflow.sklearn.load_model(model_uri)
                        model_version = "latest"
                        model_run_id = "registry"
                        logger.info(f"✅ Model loaded successfully: {model_version}")
                        return True
                else:
                    raise Exception("Experiment not found")
            except Exception as e:
                logger.warning(f"Could not get latest run_id, trying registry: {str(e)}")
                model_uri = f"models:/{model_name}/latest"
                logger.info(f"Loading latest model version from registry")
                model = mlflow.sklearn.load_model(model_uri)
                model_version = "latest"
                model_run_id = "registry"
                logger.info(f"✅ Model loaded successfully: {model_version}")
                return True
        elif model_tag.startswith('runs:/'):
            # Extract run_id from URI
            run_id = model_tag.replace('runs:/', '').split('/')[0]
            logger.info(f"Loading model from specific run_id: {run_id}")
        elif model_tag.isdigit():
            # Try as version number first
            try:
                model_uri = f"models:/{model_name}/{model_tag}"
                logger.info(f"Loading model version {model_tag} from registry")
                model = mlflow.sklearn.load_model(model_uri)
                model_version = model_tag
                model_run_id = "registry"
                logger.info(f"✅ Model loaded successfully: {model_version}")
                return True
            except Exception:
                # If version fails, try as run_id
                run_id = model_tag
                logger.info(f"Trying as run_id: {run_id}")
        else:
            # Treat as run_id (most common case)
            run_id = model_tag
            logger.info(f"Loading model from run_id: {run_id}")
        
        # Check if model artifact URI is file-based (old models) or S3-based
        run_info = client.get_run(run_id)
        artifact_uri = run_info.info.artifact_uri
        
        if artifact_uri.startswith('file://'):
            # Old model stored in file storage - try model registry first
            logger.warning(f"Model {run_id} uses file storage (old format). Trying model registry...")
            try:
                # Try to load from model registry (if registered)
                model_uri = f"models:/{model_name}/latest"
                logger.info(f"Attempting to load from model registry: {model_uri}")
                model = mlflow.sklearn.load_model(model_uri)
                model_version = "latest"
                model_run_id = "registry"
                logger.info("✅ Model loaded from registry (may be different version)")
                return True
            except Exception as reg_error:
                logger.warning(f"Registry load failed: {reg_error}")
                logger.error(f"❌ Cannot load model {run_id}: It's stored in file storage and not accessible.")
                logger.error(f"   Please train a NEW model - it will use S3 automatically.")
                logger.error(f"   Old artifact URI: {artifact_uri}")
                raise Exception(f"Model stored in file storage (not accessible). Train a new model to use S3.")
        else:
            # New model stored in S3 - try S3 first, then fallback to registry
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Loading model from S3 via MLflow: {model_uri}")
            try:
                model = mlflow.sklearn.load_model(model_uri)
                model_version = model_tag
                model_run_id = run_id
                logger.info("✅ Model loaded successfully from S3 (MinIO)")

                # Load preprocessing information
                try:
                    preprocessing_uri = f"runs:/{run_id}/preprocessing/preprocessing.json"
                    logger.info(f"Loading preprocessing info: {preprocessing_uri}")
                    client.download_artifacts(run_id, "preprocessing/preprocessing.json", "/tmp")
                    with open("/tmp/preprocessing.json", "r") as f:
                        global preprocessing_info, feature_columns
                        preprocessing_info = json.load(f)
                        feature_columns = preprocessing_info['feature_columns']
                    logger.info(f"✅ Preprocessing info loaded: {len(feature_columns)} features")
                except Exception as prep_error:
                    logger.warning(f"Could not load preprocessing info: {prep_error}")
                    logger.warning("Will use fallback feature engineering")

                return True
            except Exception as s3_error:
                logger.warning(f"S3 loading failed: {s3_error}")
                logger.info("Attempting fallback to model registry...")

                # Try loading from model registry as fallback
                try:
                    registry_uri = f"models:/{model_name}/latest"
                    logger.info(f"Loading from registry: {registry_uri}")
                    model = mlflow.sklearn.load_model(registry_uri)
                    model_version = "latest"
                    model_run_id = "registry_fallback"
                    logger.info("✅ Model loaded from registry (fallback)")

                    # Try to load preprocessing info from the run that was registered
                    try:
                        # Get the latest version info to find the run_id
                        model_versions = client.get_latest_versions(model_name, stages=["None"])
                        if model_versions:
                            latest_run_id = model_versions[0].run_id
                            preprocessing_uri = f"runs:/{latest_run_id}/preprocessing/preprocessing.json"
                            logger.info(f"Loading preprocessing info from run {latest_run_id}")
                            client.download_artifacts(latest_run_id, "preprocessing/preprocessing.json", "/tmp")
                            with open("/tmp/preprocessing.json", "r") as f:
                                preprocessing_info = json.load(f)
                                feature_columns = preprocessing_info['feature_columns']
                            logger.info(f"✅ Preprocessing info loaded: {len(feature_columns)} features")
                    except Exception as prep_error:
                        logger.warning(f"Could not load preprocessing info: {prep_error}")
                        logger.warning("Will use fallback feature engineering")

                    return True
                except Exception as reg_error:
                    logger.warning(f"Registry fallback also failed: {reg_error}")
                    raise Exception(f"Both S3 and registry loading failed. S3 error: {s3_error}, Registry error: {reg_error}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error(f"   Model tag: {model_tag}")
        logger.warning("Using mock model for predictions")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    return {
        "message": "MLOps API is running",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "model_version": model_version,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "api",
        "model_loaded": model is not None,
        "model_version": model_version
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using loaded model"""
    
    # Declare global variables at the start of the function
    global feature_columns, preprocessing_info

    # Metrics collection - start timing
    start_time = time.time()
    api_metrics["requests_total"] += 1
    api_metrics["last_request_time"] = datetime.now().isoformat()

    # Store request data for logging
    request_data = {
        "amount": request.amount,
        "quantity": request.quantity,
        "product_category": request.product_category,
        "customer_type": request.customer_type,
        "location": request.location
    }

    try:
        if model is None:
            # Fallback to mock prediction
            logger.warning("Using mock prediction (model not loaded)")
            return PredictionResponse(
                prediction=1 if request.amount > 100 else 0,
                probability=0.85,
                model_version="mock_v1.0.0",
                model_run_id="mock",
                timestamp=datetime.now().isoformat()
            )

        # Prepare features using the same preprocessing as training
        try:
            if feature_columns and preprocessing_info:
                # Use dynamic feature engineering that matches training
                logger.info(f"Using trained feature columns: {len(feature_columns)} features")
                logger.debug(f"Feature columns: {feature_columns[:10]}..." if len(feature_columns) > 10 else f"Feature columns: {feature_columns}")

                # Create a dataframe with the input data
                input_data = {
                    'amount': [request.amount],
                    'quantity': [request.quantity],
                    'product_category': [request.product_category],
                    'customer_type': [request.customer_type],
                    'location': [request.location]
                }
                df_input = pd.DataFrame(input_data)

                # Apply the same one-hot encoding as training
                df_encoded = pd.get_dummies(df_input, columns=preprocessing_info['categorical_columns'])

                # Ensure all expected columns exist (fill missing with 0)
                missing_cols = []
                for col in feature_columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                        missing_cols.append(col)
                
                if missing_cols:
                    logger.debug(f"Added {len(missing_cols)} missing columns: {missing_cols[:5]}...")

                # Keep only the expected feature columns in the correct order
                df_encoded = df_encoded[feature_columns]

                features = df_encoded.values
                logger.info(f"Features prepared: shape {features.shape}, expected {len(feature_columns)} features")
                
                # Verify feature count matches model expectations
                if hasattr(model, 'n_features_in_'):
                    if features.shape[1] != model.n_features_in_:
                        logger.error(f"Feature count mismatch! Model expects {model.n_features_in_}, got {features.shape[1]}")
                        raise ValueError(f"Feature count mismatch: model expects {model.n_features_in_}, got {features.shape[1]}")
            else:
                # Fallback: Try to extract feature names from model itself
                logger.warning("Preprocessing info not available. Attempting to extract features from model...")
                
                # Try to get feature names from the model
                model_feature_count = None
                if hasattr(model, 'n_features_in_'):
                    model_feature_count = model.n_features_in_
                    logger.info(f"Model expects {model_feature_count} features")
                
                # Try to get feature names if available (some scikit-learn models store this)
                model_feature_names = None
                if hasattr(model, 'feature_names_in_'):
                    model_feature_names = model.feature_names_in_.tolist()
                    logger.info(f"Model has feature names: {len(model_feature_names)} features")
                    # Use model's feature names to reconstruct preprocessing
                    feature_columns = model_feature_names
                    preprocessing_info = {
                        'categorical_columns': ['product_category', 'customer_type', 'location'],
                        'feature_columns': model_feature_names
                    }
                    logger.info("Using feature names from model")
                    
                    # Create a dataframe with the input data
                    input_data = {
                        'amount': [request.amount],
                        'quantity': [request.quantity],
                        'product_category': [request.product_category],
                        'customer_type': [request.customer_type],
                        'location': [request.location]
                    }
                    df_input = pd.DataFrame(input_data)
                    
                    # Apply one-hot encoding
                    df_encoded = pd.get_dummies(df_input, columns=preprocessing_info['categorical_columns'])
                    
                    # Ensure feature_columns is set
                    if feature_columns is None:
                        raise ValueError("feature_columns is None after extracting from model")
                    
                    # Ensure all expected columns exist
                    for col in feature_columns:
                        if col not in df_encoded.columns:
                            df_encoded[col] = 0
                    
                    # Keep only the expected feature columns in the correct order
                    df_encoded = df_encoded[feature_columns]
                    features = df_encoded.values
                    logger.info(f"Features prepared from model feature names: shape {features.shape}")
                else:
                    # Last resort: Query database for actual unique values
                    logger.warning("Model doesn't have feature names. Querying database for actual categories...")
                    try:
                        # Try multiple PostgreSQL hosts
                        postgres_hosts = [
                            os.getenv('POSTGRES_HOST', 'localhost'),
                            '192.168.49.1',  # Minikube host IP
                            'localhost',
                            'host.docker.internal'
                        ]
                        
                        conn = None
                        for pg_host in postgres_hosts:
                            try:
                                logger.info(f"Trying PostgreSQL at {pg_host}:5432...")
                                conn = psycopg2.connect(
                                    host=pg_host,
                                    port=int(os.getenv('POSTGRES_PORT', '5432')),
                                    database=os.getenv('POSTGRES_DB', 'ecommerce_db'),
                                    user=os.getenv('POSTGRES_USER', 'mlflow'),
                                    password=os.getenv('POSTGRES_PASSWORD', 'mlflow'),
                                    connect_timeout=3
                                )
                                logger.info(f"✅ Connected to PostgreSQL at {pg_host}")
                                break
                            except Exception as conn_error:
                                logger.debug(f"Failed to connect to {pg_host}: {conn_error}")
                                continue
                        
                        if conn is None:
                            raise Exception("Could not connect to PostgreSQL from any host")
                        
                        # Get actual unique values from database
                        actual_categories = pd.read_sql("SELECT DISTINCT product_category FROM public.transactions ORDER BY product_category", conn)['product_category'].tolist()
                        actual_customer_types = pd.read_sql("SELECT DISTINCT customer_type FROM public.customers ORDER BY customer_type", conn)['customer_type'].tolist()
                        actual_locations = pd.read_sql("SELECT DISTINCT location FROM public.customers ORDER BY location", conn)['location'].tolist()
                        
                        conn.close()
                        
                        logger.info(f"Found {len(actual_categories)} categories, {len(actual_customer_types)} customer types, {len(actual_locations)} locations")
                        
                        # Create feature vector using actual database values
                        features_list = [request.amount, request.quantity]
                        
                        # Add product category one-hot encoding
                        for category in actual_categories:
                            features_list.append(1 if request.product_category == category else 0)
                        
                        # Add customer type one-hot encoding
                        for cust_type in actual_customer_types:
                            features_list.append(1 if request.customer_type == cust_type else 0)
                        
                        # Add location one-hot encoding
                        for location in actual_locations:
                            features_list.append(1 if request.location == location else 0)
                        
                        features = np.array([features_list])
                        expected_features = 2 + len(actual_categories) + len(actual_customer_types) + len(actual_locations)
                        logger.info(f"Fallback preprocessing created {features.shape[1]} features (expected: {expected_features}) from database")
                        
                        # Set feature_columns for future use (build from database values)
                        feature_columns = ['amount', 'quantity']
                        feature_columns.extend([f'product_category_{cat}' for cat in actual_categories])
                        feature_columns.extend([f'customer_type_{ct}' for ct in actual_customer_types])
                        feature_columns.extend([f'location_{loc}' for loc in actual_locations])
                        
                        # Also set preprocessing_info for consistency
                        preprocessing_info = {
                            'categorical_columns': ['product_category', 'customer_type', 'location'],
                            'feature_columns': feature_columns
                        }
                        
                        logger.info(f"Set feature_columns from database: {len(feature_columns)} features")
                        
                        # Verify feature count matches model
                        if model_feature_count and features.shape[1] != model_feature_count:
                            logger.warning(f"Feature count mismatch: model expects {model_feature_count}, got {features.shape[1]}")
                            logger.warning("This may cause prediction errors. Consider retraining the model.")
                        
                    except Exception as db_error:
                        logger.error(f"Database query failed: {db_error}")
                        logger.error("Cannot create features without preprocessing info, model feature names, or database access")
                        raise Exception("Feature engineering failed: preprocessing info not available, model has no feature names, and database query failed")
                    
        except Exception as prep_error:
            logger.error(f"Feature preparation failed: {prep_error}")
            logger.error("Cannot proceed without proper feature engineering")
            raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(prep_error)}")

        # Make prediction
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][prediction])

        # Update prediction metrics
        if prediction == 1:
            api_metrics["predictions_high_value"] += 1
        else:
            api_metrics["predictions_low_value"] += 1

        logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")

        # Calculate response time
        response_time = time.time() - start_time
        api_metrics["response_times"].append(response_time)

        # Keep only last 100 response times for memory efficiency
        if len(api_metrics["response_times"]) > 100:
            api_metrics["response_times"] = api_metrics["response_times"][-100:]

        # Prepare response
        response_data = {
            "prediction": prediction,
            "probability": probability,
            "model_version": str(model_version),
            "model_run_id": str(model_run_id),
            "timestamp": datetime.now().isoformat()
        }

        # Log metrics to database asynchronously (don't block response)
        try:
            log_metrics_to_db(request_data, response_data, response_time)
        except Exception as log_error:
            logger.warning(f"Metrics logging failed: {log_error}")

        # Update success metrics
        api_metrics["requests_success"] += 1

        return PredictionResponse(**response_data)
        
    except Exception as e:
        # Update error metrics
        api_metrics["requests_error"] += 1

        # Calculate response time for failed requests too
        response_time = time.time() - start_time

        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/reload-model")
async def reload_model():
    """Reload model from MLflow"""
    success = load_model()
    
    if success:
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_version": model_version,
            "model_run_id": model_run_id
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/model-info")
async def model_info():
    """Get current model information"""
    return {
        "model_loaded": model is not None,
        "model_version": model_version,
        "model_run_id": model_run_id,
        "model_name": os.getenv('MODEL_NAME', 'ecommerce_classifier'),
        "mlflow_uri": os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'),
        "model_tag": os.getenv('MODEL_TAG', 'latest')
    }

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics for monitoring"""
    avg_response_time = sum(api_metrics["response_times"]) / len(api_metrics["response_times"]) if api_metrics["response_times"] else 0

    return {
        "api_metrics": {
            "requests_total": api_metrics["requests_total"],
            "requests_success": api_metrics["requests_success"],
            "requests_error": api_metrics["requests_error"],
            "success_rate": api_metrics["requests_success"] / api_metrics["requests_total"] if api_metrics["requests_total"] > 0 else 0,
            "error_rate": api_metrics["requests_error"] / api_metrics["requests_total"] if api_metrics["requests_total"] > 0 else 0,
            "avg_response_time_ms": avg_response_time * 1000,
            "predictions_high_value": api_metrics["predictions_high_value"],
            "predictions_low_value": api_metrics["predictions_low_value"],
            "last_request_time": api_metrics["last_request_time"]
        },
        "model_info": {
            "model_loaded": model is not None,
            "model_version": model_version,
            "model_run_id": model_run_id
        },
        "timestamp": datetime.now().isoformat()
    }
