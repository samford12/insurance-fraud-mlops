# tests/test_fraud_checks.py

import pandas as pd
import pytest

# Import the function from our other Python file
from fraud_checks import prepare_claims


# Create two imaginary insurance claims for testing
def sample_claims():
    return pd.DataFrame({
        "AccidentArea": ["Urban", "Rural"],
        "Make": ["Honda", "Toyota"],
        "VehicleCategory": ["Sport", "Sedan"],
        "VehiclePrice": [
            "20000 to 29000",
            "30000 to 39000"
        ],
        "PolicyType": [
            "Sport - Collision",
            "Sedan - Liability"
        ],
        "AgeOfVehicle": ["3 years", "5 years"],
        "FraudFound": ["No", "Yes"]
    })


# TEST 1 - FraudFound must not be an input feature
def test_target_not_in_features():
    X, y = prepare_claims(sample_claims())

    assert "FraudFound" not in X.columns
    assert X.shape == (2, 6)


# TEST 2 - Check correct fraud label conversion
def test_target_conversion():
    X, y = prepare_claims(sample_claims())

    assert y.tolist() == [0, 1]


# TEST 3 - Reject data when a required column is missing
def test_missing_feature():
    df = sample_claims()

    df = df.drop(columns=["Make"])

    with pytest.raises(ValueError, match="Missing columns"):
        prepare_claims(df)


# TEST 4 - Reject a missing fraud label
def test_missing_target():
    df = sample_claims()

    df.loc[0, "FraudFound"] = None

    with pytest.raises(ValueError, match="Missing fraud labels"):
        prepare_claims(df)


# TEST 5 - Reject an unexpected fraud label
def test_invalid_target():
    df = sample_claims()

    df.loc[0, "FraudFound"] = "Unknown"

    with pytest.raises(ValueError, match="Unexpected fraud labels"):
        prepare_claims(df)