#!/usr/bin/env python3
"""
Model Training Script with MLflow Integration
Trains a simple classification model on e-commerce data
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
from datetime import datetime
import psycopg2
import boto3

def fetch_training_data(postgres_host, postgres_port, postgres_db, postgres_user, postgres_password):
    """Fetch training data from PostgreSQL"""
    print(f"📊 Fetching training data from PostgreSQL...")
    
    conn = psycopg2.connect(
        host=postgres_host,
        port=postgres_port,
        database=postgres_db,
        user=postgres_user,
        password=postgres_password
    )
    
    # Fetch transaction data with customer info
    # Note: Data is from 2011, so we use a date range or just get all data
    query = """
    SELECT 
        t.transaction_id,
        t.customer_id,
        t.amount,
        t.quantity,
        t.product_category,
        c.customer_type,
        c.location,
        CASE WHEN t.amount > 100 THEN 1 ELSE 0 END as high_value
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    ORDER BY t.timestamp DESC
    LIMIT 10000
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Fetched {len(df)} records")
    return df

def prepare_features(df):
    """Prepare features for training"""
    print("🔧 Preparing features...")
    
    # Encode categorical features
    df_encoded = pd.get_dummies(df, columns=['product_category', 'customer_type', 'location'])
    
    # Select features and target
    feature_cols = [col for col in df_encoded.columns if col not in ['transaction_id', 'customer_id', 'high_value']]
    X = df_encoded[feature_cols]
    y = df_encoded['high_value']
    
    print(f"✅ Features prepared: {len(feature_cols)} features")
    return X, y, feature_cols

def train_model(X_train, y_train, X_test, y_test, model_name, model_params, feature_cols, run_name=None):
    """Train model with MLflow tracking"""
    print("🚀 Starting model training...")
    
    # Use custom run_name or auto-generate
    if run_name:
        mlflow_run_name = run_name
        print(f"📝 Using custom run name: {mlflow_run_name}")
    else:
        mlflow_run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"📝 Auto-generated run name: {mlflow_run_name}")
    
    # Note: Using experiment's default artifact location (should be S3)
    
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        # Log parameters
        mlflow.log_params(model_params)
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model with explicit artifact path if needed
        print("📦 Logging model to MLflow...")
        try:
            # Save preprocessing information
            preprocessing_info = {
                'feature_columns': feature_cols,
                'categorical_columns': ['product_category', 'customer_type', 'location'],
                'numerical_columns': ['amount', 'quantity'],
                'training_timestamp': datetime.now().isoformat()
            }

            # Save preprocessing info as artifact
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(preprocessing_info, f, indent=2)
                preprocessing_path = f.name

            mlflow.log_artifact(preprocessing_path, "preprocessing")
            print("✅ Preprocessing info saved!")

            # Clean up preprocessing file
            import os
            os.unlink(preprocessing_path)

            # First, try to save the model locally to verify it works
            import joblib
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                joblib.dump(model, f.name)
                local_model_path = f.name
            print(f"✅ Model saved locally to: {local_model_path}")

            # Now log to MLflow
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=model_name
            )
            print("✅ Model logged to MLflow!")

            # Clean up
            os.unlink(local_model_path)

        except Exception as e:
            print(f"❌ Failed to log model: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            raise

        print(f"✅ Model trained successfully!")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Artifact URI: {run.info.artifact_uri}")
        
        return run.info.run_id, metrics

def detect_minio_endpoint():
    """Detect MinIO endpoint with proper priority"""
    # Priority 1: Use environment variable if set (from Jenkinsfile)
    minio_endpoint = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
    if minio_endpoint:
        print(f"📦 Using MinIO endpoint from environment: {minio_endpoint}")
        return minio_endpoint
    
    # Priority 2: Check localhost:30900 (Kubernetes NodePort) - most common
    import socket
    import urllib.request
    
    endpoints_to_check = [
        ('http://localhost:30900', 'Kubernetes NodePort'),
        ('http://localhost:9000', 'Standalone Docker'),
    ]
    
    for endpoint, description in endpoints_to_check:
        try:
            # Try to connect to health endpoint
            url = f"{endpoint}/minio/health/live"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print(f"✅ Found MinIO on {endpoint} ({description})")
                    return endpoint
        except Exception:
            continue
    
    # Priority 3: Try Minikube IP with NodePort
    try:
        import subprocess
        minikube_ip = subprocess.check_output(['minikube', 'ip'], stderr=subprocess.DEVNULL).decode().strip()
        if minikube_ip:
            endpoint = f'http://{minikube_ip}:30900'
            try:
                url = f"{endpoint}/minio/health/live"
                req = urllib.request.Request(url, method='GET')
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        print(f"✅ Found MinIO via Minikube IP: {endpoint}")
                        return endpoint
            except Exception:
                pass
    except Exception:
        pass
    
    # Fallback: Use localhost:30900 (most likely to work with host networking)
    print(f"⚠️  Could not detect MinIO, using fallback: http://localhost:30900")
    return 'http://localhost:30900'

def main():
    parser = argparse.ArgumentParser(description='Train ML model with MLflow')
    parser.add_argument('--model-name', default='ecommerce_classifier', help='Model name in MLflow registry')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=10, help='Max depth of trees')
    parser.add_argument('--postgres-host', default='localhost', help='PostgreSQL host')
    parser.add_argument('--postgres-port', default='5432', help='PostgreSQL port')
    parser.add_argument('--postgres-db', default='ecommerce_db', help='PostgreSQL database')
    parser.add_argument('--postgres-user', default='mlflow', help='PostgreSQL user')
    parser.add_argument('--postgres-password', default='mlflow', help='PostgreSQL password')
    parser.add_argument('--mlflow-tracking-uri', default='http://localhost:5000', help='MLflow tracking URI')
    parser.add_argument('--run-name', default=None, help='Custom run name for MLflow (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Configure S3 (MinIO) for artifact storage
    # Use improved detection that prioritizes environment variable and checks NodePort first
    minio_endpoint = detect_minio_endpoint()
    
    # Set environment variables (these will be used by MLflow and boto3)
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = minio_endpoint
    os.environ.setdefault('AWS_ACCESS_KEY_ID', 'minio')
    os.environ.setdefault('AWS_SECRET_ACCESS_KEY', 'minio123')
    os.environ.setdefault('MLFLOW_S3_IGNORE_TLS', 'true')
    
    print(f"📦 S3 (MinIO) configured: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")

    # Configure MLflow to use mlflow schema (separated from business data)
    if 'postgresql' in args.mlflow_tracking_uri:
        # Add schema parameter to PostgreSQL connection
        if '?' not in args.mlflow_tracking_uri:
            args.mlflow_tracking_uri += '?options=-csearch_path=mlflow'
    print(f"📊 MLflow using schema: mlflow (separated from business data)")

    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Test S3 connection
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            use_ssl=False,
            verify=False
        )
        s3.head_bucket(Bucket='mlflow-artifacts')
        print(f"✅ S3 (MinIO) connection verified: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
    except Exception as e:
        print(f"⚠️  S3 connection test failed: {e}")
        print(f"⚠️  Endpoint: {os.environ['MLFLOW_S3_ENDPOINT_URL']}")
        print("⚠️  Will attempt to continue, but artifacts may fail to upload")

    # Get or create experiment with S3 artifact location
    experiment_name = "ecommerce_model_training"
    s3_experiment_name = "ecommerce_model_training_s3"
    client = mlflow.tracking.MlflowClient()
    
    # Check if S3 experiment exists
    try:
        experiment = client.get_experiment_by_name(s3_experiment_name)
        if experiment:
            print(f"✅ Using existing S3 experiment: {s3_experiment_name}")
            experiment_id = experiment.experiment_id
        else:
            raise Exception("Experiment not found")
    except:
        # Check if old file-based experiment exists
        try:
            old_experiment = client.get_experiment_by_name(experiment_name)
            if old_experiment and 'file://' in old_experiment.artifact_location:
                print(f"⚠️ Experiment '{experiment_name}' uses file storage. Creating new experiment with S3...")
        except:
            pass
        
        # Create new experiment with S3 artifact location
        try:
            artifact_location = f"s3://mlflow-artifacts/artifacts"
            experiment_id = client.create_experiment(
                s3_experiment_name,
                artifact_location=artifact_location
            )
            print(f"✅ Created new S3 experiment: {s3_experiment_name}")
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" in str(e):
                print(f"⚠️ Error checking experiment: {e}")
                experiment = client.get_experiment_by_name(s3_experiment_name)
                experiment_id = experiment.experiment_id
            else:
                raise
    
    # Set active experiment
    mlflow.set_experiment(s3_experiment_name)
    
    # Fetch and prepare data
    df = fetch_training_data(
        args.postgres_host,
        args.postgres_port,
        args.postgres_db,
        args.postgres_user,
        args.postgres_password
    )
    
    X, y, feature_cols = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model parameters
    model_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'random_state': 42
    }
    
    # Train model
    run_id, metrics = train_model(
        X_train, y_train, X_test, y_test,
        args.model_name,
        model_params,
        feature_cols,
        run_name=args.run_name
    )
    
    # Print summary
    print("")
    print("=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Model Name: {args.model_name}")
    print(f"Run ID: {run_id}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"MLflow UI: {args.mlflow_tracking_uri}")
    print("=" * 80)
    
    return run_id, metrics

if __name__ == "__main__":
    main()
