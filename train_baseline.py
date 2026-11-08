from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
)
# LESSON 5 - LOAD INSURANCE DATA

project_folder = Path(__file__).resolve().parent
data_path = project_folder / "data" / "carclaims.csv"

df = pd.read_csv(data_path)

# Keep only records with known target labels
df = df[df["FraudFound"].isin(["Yes", "No"])].copy()

print("Claims available:", len(df))
# Select a limited set of candidate features
feature_columns = [
    "AccidentArea",
    "Make",
    "VehicleCategory",
    "VehiclePrice",
    "PolicyType",
    "AgeOfVehicle",
]

# X = model inputs
X = df[feature_columns].copy()

# y = target (1 means fraud-labelled, 0 means not)
y = (df["FraudFound"] == "Yes").astype(int)

print("\nInput features:")
print(X.head())

print("\nTarget distribution:")
print(y.value_counts())
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("Training fraud rate:", y_train.mean())
print("Testing fraud rate:", y_test.mean())
# Handle missing values and encode categories
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, feature_columns)
])
# Create a complete ML pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

# Train on training data only
model.fit(X_train, y_train)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score
)
print("\nModel training completed!")
# LESSON 5 - MODEL EVALUATION

# 1. Get the model's fraud-class scores
fraud_scores = model.predict_proba(X_test)[:, 1]

# 2. Convert scores into binary predictions
threshold = 0.50
predictions = (fraud_scores >= threshold).astype(int)

# 3. Evaluate our model
print("\n--- MODEL EVALUATION ---")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nAverage Precision:")
print(average_precision_score(y_test, fraud_scores))

print("\nROC-AUC:")
print(roc_auc_score(y_test, fraud_scores))

print("\nFraud Prevalence Baseline:")
print(y_test.mean())