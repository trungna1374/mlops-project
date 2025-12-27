from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Evidently Monitoring", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Evidently Monitoring is running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "evidently"}

@app.get("/metrics")
async def metrics():
    # Mock monitoring metrics
    return {
        "data_drift_detected": False,
        "accuracy_score": 0.89,
        "last_updated": datetime.now().isoformat(),
        "total_predictions": 15420
    }
