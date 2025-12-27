# 🤖 MLOps E-commerce Classification Platform

A complete end-to-end MLOps platform that trains, deploys, and monitors machine learning models for predicting high-value e-commerce transactions.

## 📋 Table of Contents

- [What This System Does](#what-this-system-does)
- [System Architecture](#system-architecture)
- [Components Overview](#components-overview)
- [Complete Workflow](#complete-workflow)
- [Quick Start Guide](#quick-start-guide)
- [Service Access](#service-access)
- [Troubleshooting](#troubleshooting)

---

## 🎯 What This System Does

**Problem**: Predict whether an e-commerce transaction will be "high-value" (> $100)

**Solution**: Train a machine learning model that analyzes:
- Transaction amount and quantity
- Product category
- Customer type (regular/premium/vip)
- Customer location

**Output**: Prediction (0 = low-value, 1 = high-value) with confidence score

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Platform                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Data       │───▶│   Training   │───▶│   Model      │     │
│  │   Storage    │    │   Pipeline   │    │   Registry   │     │
│  │              │    │              │    │              │     │
│  │ PostgreSQL  │    │  MLflow +    │    │  MLflow      │     │
│  │              │    │  Scikit-learn│    │  Registry    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                   │                    │              │
│         │                   │                    │              │
│         ▼                   ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Serving    │    │  Monitoring  │    │  Deployment  │     │
│  │              │    │              │    │              │     │
│  │  FastAPI     │    │  Grafana +   │    │  Kubernetes  │     │
│  │  API         │    │  Streamlit   │    │  + ArgoCD    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Layers

```
Layer 1: Data Layer
├── PostgreSQL (Database)
│   ├── Business Data: customers, transactions
│   └── MLflow Data: experiments, runs, models
└── MinIO (S3 Storage)
    └── Model Artifacts: trained models, preprocessing info

Layer 2: ML Layer
├── MLflow (Model Tracking)
│   ├── Experiment Tracking
│   ├── Model Registry
│   └── Artifact Storage (MinIO)
└── Training Pipeline
    ├── Data Fetching (PostgreSQL)
    ├── Feature Engineering
    └── Model Training (Scikit-learn)

Layer 3: Serving Layer
├── FastAPI (Prediction API)
│   ├── Load Model from MLflow
│   ├── Feature Engineering
│   └── Prediction Endpoint
└── Streamlit (Monitoring UI)
    └── Model Insights Dashboard

Layer 4: CI/CD Layer
├── Jenkins (Automation)
│   ├── Build Docker Images
│   ├── Train Models
│   └── Update Git Tags
└── ArgoCD (GitOps)
    ├── Watch Git Repository
    ├── Sync Kubernetes Manifests
    └── Deploy Services

Layer 5: Monitoring Layer
├── Grafana (Dashboards)
│   ├── Transaction Metrics
│   ├── Model Performance
│   └── System Health
└── PostgreSQL (Metrics Storage)
    └── API Metrics Table
```

---

## 🧩 Components Overview

### 1. Data Storage Components

| Component | Technology | Purpose | Port | Access |
|-----------|-----------|---------|------|--------|
| **PostgreSQL** | Database | Stores transactions, customers, MLflow metadata | 5432 | `localhost:5432` |
| **MinIO** | S3 Storage | Stores MLflow model artifacts | 9000/30900 | `http://localhost:9000` |

**What it does:**
- PostgreSQL: Central database for all business data and MLflow tracking
- MinIO: S3-compatible storage for model files, preprocessing info, and artifacts

### 2. ML Components

| Component | Technology | Purpose | Port | Access |
|-----------|-----------|---------|------|--------|
| **MLflow** | ML Platform | Experiment tracking, model registry, UI | 5000 | `http://localhost:5000` |
| **Training Script** | Python + Scikit-learn | Trains Random Forest model | - | Runs in Docker |

**What it does:**
- MLflow: Tracks experiments, stores models, manages model versions
- Training Script: Fetches data, engineers features, trains model, logs to MLflow

### 3. Serving Components

| Component | Technology | Purpose | Port | Access |
|-----------|-----------|---------|------|--------|
| **FastAPI** | REST API | Serves predictions via HTTP | 8000 | `http://localhost:8000` |
| **Streamlit** | Web UI | Monitoring dashboard | 8501 | `http://localhost:8501` |

**What it does:**
- FastAPI: Loads model from MLflow, processes requests, returns predictions
- Streamlit: Visualizes model performance, system status, predictions

### 4. CI/CD Components

| Component | Technology | Purpose | Port | Access |
|-----------|-----------|---------|------|--------|
| **Jenkins** | CI/CD | Automated pipelines | 8080 | `http://localhost:8080` |
| **ArgoCD** | GitOps | Auto-deploys from Git | 8443 | `https://localhost:8443` |

**What it does:**
- Jenkins: Builds images, trains models, updates Git tags
- ArgoCD: Watches Git, syncs Kubernetes deployments automatically

### 5. Monitoring Components

| Component | Technology | Purpose | Port | Access |
|-----------|-----------|---------|------|--------|
| **Grafana** | Dashboards | Visualizes metrics | 3000 | `http://localhost:3000` |
| **Data Generator** | Python | Generates synthetic transactions | - | Runs in Kubernetes |
| **User Simulator** | Python | Simulates API requests | - | Runs in Kubernetes |

**What it does:**
- Grafana: Shows transaction trends, model performance, system health
- Data Generator: Creates realistic transaction data for training
- User Simulator: Tests API with realistic load

---

## 🔄 Complete Workflow

### Phase 1: Data Ingestion

```
┌─────────────┐
│ Kaggle Data │
│   (CSV)     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Import      │────▶│ PostgreSQL   │
│ Script      │     │ (customers,  │
│             │     │ transactions)│
└─────────────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│ Data        │
│ Generator   │────▶ (Continuous data generation)
│ (Kubernetes)│
└─────────────┘
```

**Steps:**
1. Download Kaggle dataset (e-commerce transactions)
2. Import to PostgreSQL using `import-kaggle-data.sh`
3. Data Generator creates ongoing synthetic transactions
4. Data is ready for training

### Phase 2: Model Training

```
┌──────────────┐
│ PostgreSQL   │
│ (Data)       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Training     │
│ Script       │
│ (Jenkins)    │
└──────┬───────┘
       │
       ├──▶ Feature Engineering
       │    (One-hot encoding)
       │
       ├──▶ Model Training
       │    (Random Forest)
       │
       └──▶ Log to MLflow
            │
            ├──▶ Metrics (accuracy, F1)
            ├──▶ Model Artifact
            ├──▶ Preprocessing Info
            └──▶ Register Model
                 │
                 ▼
            ┌──────────────┐
            │ MinIO (S3)   │
            │ (Artifacts)  │
            └──────────────┘
```

**Steps:**
1. Jenkins pipeline triggers training
2. Training script fetches data from PostgreSQL
3. Features are engineered (one-hot encoding)
4. Random Forest model is trained
5. Model + metrics + preprocessing info saved to MLflow
6. Artifacts stored in MinIO (S3)
7. Model registered in MLflow registry

### Phase 3: Model Deployment

```
┌──────────────┐
│ MLflow       │
│ Registry     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Jenkins      │
│ Updates      │
│ Git Tags     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Git          │
│ Repository   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ ArgoCD       │
│ (Watches Git)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Kubernetes   │
│ API Pod      │
│ (FastAPI)    │
└──────────────┘
```

**Steps:**
1. Training completes, model registered in MLflow
2. Jenkins updates `image-tags.env` and `k8s/base/api.yaml` with new model run_id
3. Jenkins commits and pushes to Git
4. ArgoCD detects Git changes
5. ArgoCD syncs Kubernetes manifests
6. API pod restarts with new model
7. API loads model from MLflow using new run_id

### Phase 4: Prediction Serving

```
┌──────────────┐
│ User/Client  │
│ Request      │
└──────┬───────┘
       │
       │ POST /predict
       │ {
       │   "amount": 150.00,
       │   "quantity": 2,
       │   "product_category": "electronics",
       │   "customer_type": "premium",
       │   "location": "USA"
       │ }
       │
       ▼
┌──────────────┐
│ FastAPI      │
│ (API Pod)    │
└──────┬───────┘
       │
       ├──▶ Load Model from MLflow
       │    (using MODEL_TAG)
       │
       ├──▶ Load Preprocessing Info
       │    (feature columns)
       │
       ├──▶ Feature Engineering
       │    (match training)
       │
       ├──▶ Model Prediction
       │
       └──▶ Return Result
            │
            ▼
┌──────────────┐
│ Response     │
│ {
│   "prediction": 1,
│   "probability": 0.95,
│   "model_version": "..."
│ }
└──────────────┘
```

**Steps:**
1. Client sends prediction request to API
2. API loads model from MLflow (using MODEL_TAG from environment)
3. API loads preprocessing info (feature columns)
4. API engineers features to match training
5. Model makes prediction
6. API logs metrics to PostgreSQL
7. Response returned to client

### Phase 5: Monitoring & Feedback

```
┌──────────────┐
│ API          │
│ (Predictions)│
└──────┬───────┘
       │
       ├──▶ Log Metrics
       │    │
       │    ▼
       │ ┌──────────────┐
       │ │ PostgreSQL   │
       │ │ (api_metrics)│
       │ └──────┬───────┘
       │        │
       │        ▼
       │ ┌──────────────┐
       │ │ Grafana      │
       │ │ (Dashboards) │
       │ └──────────────┘
       │
       └──▶ Model Performance
            │
            ▼
       ┌──────────────┐
       │ Streamlit    │
       │ (UI)         │
       └──────────────┘
```

**Steps:**
1. API logs each prediction to PostgreSQL
2. Grafana queries PostgreSQL and displays:
   - Transaction volume
   - Prediction distribution
   - Model accuracy over time
   - System health metrics
3. Streamlit shows:
   - Current model version
   - Recent predictions
   - Model insights

---

## 🚀 Quick Start Guide

### Step 1: Start All Services

```bash
# Start all standalone services (PostgreSQL, MLflow, Grafana, Jenkins)
./scripts/start-all-services.sh

# Start Kubernetes (Minikube)
./scripts/start-minikube.sh
```

**What happens:**
- PostgreSQL starts on port 5432
- MLflow starts on port 5000
- Grafana starts on port 3000
- Jenkins starts on port 8080
- Minikube cluster starts

### Step 2: Initialize Database

```bash
# Setup database schema
./scripts/setup-standalone-db.sh

# Import Kaggle data
./scripts/import-kaggle-data.sh
```

**What happens:**
- Creates `customers` and `transactions` tables
- Creates `mlflow` schema for MLflow data
- Imports e-commerce transaction data
- Data ready for training

### Step 3: Setup Monitoring

```bash
# Configure Grafana dashboards
./scripts/setup-grafana-dashboard.sh
```

**What happens:**
- Adds PostgreSQL as Grafana data source
- Imports monitoring dashboards
- Ready to visualize metrics

### Step 4: Train First Model

**Option A: Via Jenkins (Recommended)**
1. Access Jenkins: `http://<server-ip>:8080`
2. Get admin password: `docker exec jenkins-ci cat /var/jenkins_home/secrets/initialAdminPassword`
3. Create "model-training" job (or use existing)
4. Run the pipeline
5. Model is trained and registered in MLflow

**Option B: Manual Training**
```bash
cd scripts
python train_model.py \
  --model-name ecommerce_classifier \
  --mlflow-tracking-uri http://localhost:5000
```

**What happens:**
- Data fetched from PostgreSQL
- Features engineered (one-hot encoding)
- Random Forest model trained
- Model saved to MLflow
- Artifacts stored in MinIO

### Step 5: Deploy Model

**Automatic (via ArgoCD):**
1. Jenkins updates Git with new model tag
2. ArgoCD detects change
3. API pod restarts with new model
4. Model is live!

**Manual:**
```bash
# Update model tag in k8s/base/api.yaml
# Apply changes
kubectl apply -f k8s/base/api.yaml -n mlops
```

### Step 6: Test Prediction

```bash
# Port-forward API (if needed)
kubectl port-forward -n mlops svc/api 8000:8000 --address 0.0.0.0 &

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 150.00,
    "quantity": 2,
    "product_category": "electronics",
    "customer_type": "premium",
    "location": "USA"
  }'
```

**Expected Response:**
```json
{
  "prediction": 1,
  "probability": 0.95,
  "model_version": "latest",
  "model_run_id": "abc123...",
  "timestamp": "2025-12-27T10:00:00"
}
```

---

## 🌐 Service Access

### Local Access (on server)

| Service | URL | Credentials |
|---------|-----|-------------|
| **Jenkins** | http://localhost:8080 | admin / (get password) |
| **MLflow** | http://localhost:5000 | No auth |
| **Grafana** | http://localhost:3000 | admin / admin123 |
| **PostgreSQL** | localhost:5432 | mlflow / mlflow |
| **MinIO** | http://localhost:9000 | minio / minio123 |

### External Access (from outside server)

**Standalone Services (Direct Access):**
- **Jenkins**: `http://<server-ip>:8080` (after firewall config)
- **MLflow**: `http://<server-ip>:5000`
- **Grafana**: `http://<server-ip>:3000`

**Kubernetes Services (Port-Forward Required):**
```bash
# API
kubectl port-forward -n mlops svc/api 8000:8000 --address 0.0.0.0 &

# Streamlit
kubectl port-forward -n mlops svc/streamlit 8501:8501 --address 0.0.0.0 &

# MinIO (if NodePort not working)
kubectl port-forward -n mlops svc/minio 9000:9000 --address 0.0.0.0 &

# ArgoCD
kubectl port-forward -n argocd svc/argocd-server 8443:443 --address 0.0.0.0 &
```

**Then access:**
- **API**: `http://<server-ip>:8000`
- **Streamlit**: `http://<server-ip>:8501`
- **ArgoCD**: `https://<server-ip>:8443`

### Configure Firewall

```bash
# Allow external access
sudo ufw allow 3000/tcp  # Grafana
sudo ufw allow 5000/tcp  # MLflow
sudo ufw allow 8080/tcp  # Jenkins
sudo ufw allow 8501/tcp  # Streamlit
sudo ufw allow 8000/tcp  # API
sudo ufw status
```

### Jenkins External Access

**Check accessibility:**
```bash
./scripts/check-jenkins-access.sh
```

**If not accessible:**
1. Allow firewall: `sudo ufw allow 8080/tcp`
2. Configure AWS Security Group (if on EC2): Add inbound rule for port 8080
3. Restart Jenkins: `./scripts/start-jenkins-with-recovery.sh`

**Get Jenkins password:**
```bash
docker exec jenkins-ci cat /var/jenkins_home/secrets/initialAdminPassword
```

---

## 🔄 Typical Workflow Example

### Scenario: Retrain Model with New Data

```
1. New Data Arrives
   └── Data Generator creates new transactions
       └── Stored in PostgreSQL

2. Trigger Training
   └── Go to Jenkins UI
       └── Run "model-training" pipeline
           ├── Fetches latest data from PostgreSQL
           ├── Trains new model
           ├── Logs to MLflow
           └── Updates Git with new model tag

3. Automatic Deployment
   └── ArgoCD detects Git change
       └── Syncs Kubernetes
           └── API pod restarts
               └── Loads new model from MLflow

4. Model is Live
   └── API serves predictions with new model
       └── Metrics logged to PostgreSQL
           └── Grafana shows updated performance
```

### Scenario: Making a Prediction

```
1. Client Request
   POST http://<server-ip>:8000/predict
   {
     "amount": 150.00,
     "quantity": 2,
     "product_category": "electronics",
     "customer_type": "premium",
     "location": "USA"
   }

2. API Processing
   ├── Loads model from MLflow (using MODEL_TAG)
   ├── Loads preprocessing info (feature columns)
   ├── Engineers features:
   │   ├── amount: 150.00
   │   ├── quantity: 2
   │   ├── product_category_electronics: 1
   │   ├── customer_type_premium: 1
   │   └── location_USA: 1
   ├── Model predicts: 1 (high-value)
   └── Returns: {"prediction": 1, "probability": 0.95}

3. Logging
   └── Metrics saved to PostgreSQL
       └── Grafana dashboard updates
```

---

## 🐛 Troubleshooting

### Quick Diagnostics

```bash
# Check all services
./scripts/check-services-status.sh

# Check Jenkins access
./scripts/check-jenkins-access.sh

# Check MinIO access
./scripts/check-minio-access.sh

# Check service connectivity
./scripts/check-service-access.sh

# Check PostgreSQL data
./scripts/check-postgres-data.sh
```

### Common Issues

**1. Jenkins not accessible externally**
- Run: `./scripts/check-jenkins-access.sh`
- Fix: `sudo ufw allow 8080/tcp` and configure AWS Security Group

**2. Model feature mismatch**
- Error: "X has 54 features, but model expects 33"
- Fix: Ensure preprocessing_info is loaded correctly
- Check: API logs for "Using trained feature columns"

**3. MinIO connection failed**
- Error: "Could not connect to endpoint URL"
- Fix: Start port-forward: `kubectl port-forward -n mlops svc/minio 9000:9000 --address 0.0.0.0 &`

**4. API can't load model**
- Check: MinIO accessibility
- Check: MODEL_TAG in k8s/base/api.yaml
- Check: MLflow tracking URI

**5. ArgoCD sync fails**
- Check: Git repository credentials
- Check: Kubernetes manifests exist
- Check: ArgoCD logs: `kubectl logs -n argocd deployment/argocd-repo-server`

---

## 📚 Key Files

### Configuration Files
- `image-tags.env` - Current model and image versions
- `k8s/base/api.yaml` - API deployment configuration
- `k8s/base/minio.yaml` - MinIO S3 storage configuration
- `k8s/argocd/application.yaml` - ArgoCD GitOps configuration

### Scripts
- `scripts/start-all-services.sh` - Start all services
- `scripts/train_model.py` - Model training script
- `scripts/check-*.sh` - Diagnostic scripts
- `Jenkinsfile-model-training` - Jenkins CI/CD pipeline

### Documentation
- `JENKINS_EXTERNAL_ACCESS.md` - Jenkins access guide
- `SERVICE_ACCESS_TROUBLESHOOTING.md` - Service troubleshooting
- `QUICK_REFERENCE.md` - Quick command reference

---

## 🎓 Learning Path

1. **Understand the Data Flow**: How data moves from PostgreSQL → Training → MLflow → API
2. **Explore MLflow**: Check experiments, models, and artifacts
3. **Test the API**: Make predictions and see results
4. **Monitor with Grafana**: View dashboards and metrics
5. **Trigger Retraining**: Use Jenkins to train new models
6. **Watch GitOps**: See ArgoCD automatically deploy changes

---

**Ready to deploy ML models to production?** This platform provides everything you need! 🚀
