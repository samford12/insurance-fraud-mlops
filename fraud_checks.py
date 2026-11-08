# fraud_checks.py

import pandas as pd

# These are the features our model is allowed to use.
FEATURE_COLUMNS = [
    "AccidentArea",
    "Make",
    "VehicleCategory",
    "VehiclePrice",
    "PolicyType",
    "AgeOfVehicle",
]


def prepare_claims(df):
    """Check insurance data and return features and labels."""

    # Check 1: Are all required columns present?
    required = set(FEATURE_COLUMNS) | {"FraudFound"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Check 2: Are fraud labels missing?
    if df["FraudFound"].isna().any():
        raise ValueError("Missing fraud labels")

    # Check 3: Are fraud labels valid?
    expected = {"Yes", "No"}
    actual = set(df["FraudFound"].unique())

    unexpected = actual - expected

    if unexpected:
        raise ValueError(
            f"Unexpected fraud labels: {unexpected}"
        )

    # Select the six model input columns
    X = df[FEATURE_COLUMNS].copy()

    # Convert Yes/No target labels to 1/0
    y = (df["FraudFound"] == "Yes").astype(int)

    return X, y