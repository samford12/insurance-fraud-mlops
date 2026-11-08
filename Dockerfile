FROM python:3.13-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY fraud_api.py .

# Include our exported MLflow model
COPY model_bundle ./model_bundle

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]