import os
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# PART 1 - LOAD THE SAVED MLFLOW MODEL
# ==========================================

# ==========================================
# PART 1 - LOAD THE SAVED MLFLOW MODEL
# ==========================================

project_folder = Path(__file__).resolve().parent

# Look for our exported model
local_model = project_folder / "model_bundle"

if (local_model / "MLmodel").is_file():
    # Use the local model inside Docker
    model_uri = str(local_model)
else:
    # Otherwise load from our MLflow model URI
    model_uri_path = project_folder / "model_uri.txt"

    if not model_uri_path.is_file():
        raise FileNotFoundError("model_uri.txt not found")

    model_uri = model_uri_path.read_text(
        encoding="utf-8"
    ).strip()

    if not model_uri.startswith("models:/"):
        raise ValueError("Invalid MLflow model URI")

mlflow.set_tracking_uri(
    os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000"
    )
)

# Load the trained model
model = mlflow.sklearn.load_model(model_uri)
# ==========================================
# PART 2 - CREATE THE FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Insurance Fraud Review API",
    version="1.0.0",
    description="Local learning API for claim review prioritization."
)


# ==========================================
# PART 3 - DEFINE THE INPUT DATA STRUCTURE
# ==========================================

class InsuranceClaim(BaseModel):

    model_config = ConfigDict(extra="forbid")

    AccidentArea: str = Field(min_length=1)
    Make: str = Field(min_length=1)
    VehicleCategory: str = Field(min_length=1)
    VehiclePrice: str = Field(min_length=1)
    PolicyType: str = Field(min_length=1)
    AgeOfVehicle: str = Field(min_length=1)


# ==========================================
# PART 4 - HEALTH CHECK ENDPOINT
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model_loaded": True
    }


# ==========================================
# PART 5 - PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict_claim(claim: InsuranceClaim):

    # Convert validated JSON data into a Python dictionary
    claim_data = claim.model_dump()

    # Convert the dictionary into a one-row DataFrame
    X_new = pd.DataFrame([claim_data])

    # Generate the fraud-class review score
    score = float(
        model.predict_proba(X_new)[0, 1]
    )

    # Demonstration threshold only
    threshold = 0.50

    return {
        "review_score": round(score, 4),
        "manual_review_flag": bool(score >= threshold),
        "threshold": threshold,
        "message": (
            "Model score for human review prioritization, "
            "not a confirmed fraud finding."
        )
    }