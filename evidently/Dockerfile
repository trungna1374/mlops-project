FROM python:3.9-slim

WORKDIR /app

# Install dependencies
# Use pydantic 1.x for evidently compatibility
RUN pip install --no-cache-dir \
    pydantic==1.10.13 \
    evidently==0.4.11 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    scikit-learn==1.3.0

# Copy application
COPY . .

EXPOSE 8085

CMD ["python", "app.py"]

