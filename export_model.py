from pathlib import Path
import shutil
import mlflow
from mlflow.artifacts import download_artifacts

# Find our project folder
project = Path(__file__).resolve().parent

# Read the saved model URI
model_uri = (project / "model_uri.txt").read_text(
    encoding="utf-8"
).strip()

if not model_uri.startswith("models:/"):
    raise ValueError("Invalid model URI in model_uri.txt")

# Connect to the local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Download the trained model
downloaded = download_artifacts(artifact_uri=model_uri)

# Copy it into our project
destination = project / "model_bundle"

shutil.copytree(
    downloaded,
    destination,
    dirs_exist_ok=True
)

print("Model exported to:", destination)
print("Model files found:", (destination / "MLmodel").is_file())