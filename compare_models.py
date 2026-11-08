from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from fraud_checks import prepare_claims

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# PART 1 - LOAD AND VALIDATE INSURANCE DATA
# ==========================================

# Find the project folder
project_folder = Path(__file__).resolve().parent

# Build the path to our CSV file
data_path = project_folder / "data" / "carclaims.csv"

# Check that the dataset exists
if not data_path.is_file():
    raise FileNotFoundError(
        f"Cannot find dataset: {data_path}"
    )

# Read historical insurance claims
df = pd.read_csv(data_path)

# Validate the data using fraud_checks.py
# The function returns:
# X = six selected insurance features
# y = fraud target encoded as 0 or 1
X, y = prepare_claims(df)

# Use the validated feature names
features = list(X.columns)

# Display dataset information
print("\n--- DATASET INFORMATION ---")
print("Total claims:", len(df))
print("Fraud-labelled claims:", int(y.sum()))
print("Number of features:", len(features))
print("Selected features:", features)


# ==========================================
# PART 2 - SPLIT THE DATA
# ==========================================

# First split:
# 80% for model development
# 20% reserved for testing
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Second split:
# 60% of original data for training
# 20% of original data for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

print("\n--- DATA SPLIT ---")
print("Training claims:", len(X_train))
print("Validation claims:", len(X_val))
print("Reserved test claims:", len(X_test))

print(
    "Training fraud rate:",
    f"{y_train.mean():.2%}"
)

print(
    "Validation fraud rate:",
    f"{y_val.mean():.2%}"
)


# ==========================================
# PART 3 - DEFINE MACHINE LEARNING MODELS
# ==========================================

# A dictionary containing our two classifiers
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
}


# ==========================================
# PART 4 - CONNECT TO MLFLOW
# ==========================================

# Connect to the locally running MLflow server
mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

# Create or select the experiment
mlflow.set_experiment(
    "insurance-fraud-model-comparison"
)


# ==========================================
# PART 5 - TRAIN AND COMPARE MODELS
# ==========================================

for model_name, classifier in models.items():

    print(f"\nTraining model: {model_name}")

    # --------------------------------------
    # 5A - PREPROCESS CATEGORICAL FEATURES
    # --------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    # Apply preprocessing to our six features
    preprocessor = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            features,
        ),
    ])

    # --------------------------------------
    # 5B - BUILD COMPLETE MODEL PIPELINE
    # --------------------------------------

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    # --------------------------------------
    # 5C - START MLFLOW RUN
    # --------------------------------------

    with mlflow.start_run(run_name=model_name):

        # Train the model using training data only
        model.fit(X_train, y_train)

        # ----------------------------------
        # 5D - GENERATE VALIDATION SCORES
        # ----------------------------------

        # Score for the fraud-labelled class
        scores = model.predict_proba(X_val)[:, 1]

        # Demonstration classification threshold
        threshold = 0.50

        predictions = (
            scores >= threshold
        ).astype(int)

        # ----------------------------------
        # 5E - CALCULATE EVALUATION METRICS
        # ----------------------------------

        average_precision = average_precision_score(
            y_val,
            scores,
        )

        roc_auc = roc_auc_score(
            y_val,
            scores,
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        # Proportion of claims flagged for review
        review_rate = predictions.mean()

        # ----------------------------------
        # PART 6 - LOG MLFLOW PARAMETERS
        # ----------------------------------

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "training_records",
            len(X_train),
        )

        mlflow.log_param(
            "validation_records",
            len(X_val),
        )

        mlflow.log_param(
            "threshold",
            threshold,
        )

        mlflow.log_param(
            "serialization_format",
            "skops",
        )

        # ----------------------------------
        # PART 7 - LOG MLFLOW METRICS
        # ----------------------------------

        mlflow.log_metric(
            "validation_average_precision",
            average_precision,
        )

        mlflow.log_metric(
            "validation_roc_auc",
            roc_auc,
        )

        mlflow.log_metric(
            "validation_precision",
            precision,
        )

        mlflow.log_metric(
            "validation_recall",
            recall,
        )

        mlflow.log_metric(
            "validation_review_rate",
            review_rate,
        )

        # ----------------------------------
        # PART 8 - SAVE TRAINED MODEL
        # ----------------------------------

        # These are specific types required by
        # models we trained locally and trust.
        trusted_types = ["numpy.dtype"]

        # Random Forest uses scikit-learn trees
        if model_name == "Random Forest":
            trusted_types.append(
                "sklearn.tree._tree.Tree"
            )

        # Save the full preprocessing + model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="fraud_detection_pipeline",
            input_example=X_val.head(3),
            serialization_format="skops",
            skops_trusted_types=trusted_types,
        )

        # ----------------------------------
        # PART 9 - DISPLAY RESULTS
        # ----------------------------------

        print(f"\n--- {model_name} RESULTS ---")

        print(
            f"Average Precision: "
            f"{average_precision:.4f}"
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall: {recall:.4f}"
        )

        print(
            f"Claims flagged for review: "
            f"{review_rate:.2%}"
        )

        print(
            "Saved MLflow model:",
            model_info.model_uri,
        )


# ==========================================
# PART 10 - FINAL CONFIRMATION
# ==========================================

print("\n================================")
print("Both model comparison runs completed.")
print("Check MLflow for metrics and models.")
print("================================")