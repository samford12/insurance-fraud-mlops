import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Find our insurance dataset
project_folder = Path(__file__).resolve().parent
data_path = project_folder / "data" / "carclaims.csv"

# Load the CSV into Python
df = pd.read_csv(data_path)

# Explore the dataset
print("Rows and columns:", df.shape)

print("\nFirst five insurance claims:")
print(df.head())
# LESSON 4 - EXPLORATORY DATA ANALYSIS

# 1. Display all column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 2. Inspect data types and non-null counts
print("\n--- DATASET INFORMATION ---")
df.info()

# 3. Count missing values
print("\n--- MISSING VALUES ---")
print(df.isna().sum())
# 4. Inspect our target variable
print("\n--- FRAUD LABEL COUNTS ---")
print(df["FraudFound"].value_counts(dropna=False))

# 5. Check unique fraud labels
print("\n--- UNIQUE FRAUD LABELS ---")
print(df["FraudFound"].unique())
df["FraudFound"].unique()
# LESSON 4B - FRAUD RATE ANALYSIS

# 1. Keep records with known fraud labels
fraud_labels = df["FraudFound"].dropna()

# 2. Count labeled claims
total_claims = len(fraud_labels)

# 3. Identify fraud-labeled claims
is_fraud = fraud_labels == "Yes"

# 4. Count fraud and non-fraud labels
fraud_claims = is_fraud.sum()
non_fraud_claims = (fraud_labels == "No").sum()

# 5. Calculate the fraud rate
fraud_rate = fraud_claims / total_claims

# 6. Display results
print("\n--- FRAUD RATE ANALYSIS ---")
print("Total labeled claims:", total_claims)
print("Fraud-labeled claims:", fraud_claims)
print("Non-fraud-labeled claims:", non_fraud_claims)
print(f"Fraud rate: {fraud_rate:.2%}")
# LESSON 4C - GROUP ANALYSIS

# Select claims with known fraud labels
claims = df[df["FraudFound"].isin(["Yes", "No"])].copy()

# Create a binary fraud indicator
claims["is_fraud"] = claims["FraudFound"] == "Yes"

# Check the first five records
print("\n--- FRAUD INDICATOR ---")
print(claims[["AccidentArea", "FraudFound", "is_fraud"]].head())
# Group claims by accident area
fraud_by_area = claims.groupby("AccidentArea").agg(
    total_claims=("is_fraud", "size"),
    fraud_claims=("is_fraud", "sum"),
    fraud_rate=("is_fraud", "mean")
)

# Display the results
print("\n--- FRAUD BY ACCIDENT AREA ---")
print(fraud_by_area)
# Sort from highest to lowest fraud rate
fraud_by_area = fraud_by_area.sort_values(
    by="fraud_rate",
    ascending=False
)

# Display the sorted results
print("\n--- SORTED FRAUD RATES ---")
print(fraud_by_area)
# LESSON 4C - VEHICLE CATEGORY ANALYSIS

# See which vehicle categories exist
print("\n--- VEHICLE CATEGORIES ---")
print(claims["VehicleCategory"].value_counts())
fraud_by_vehicle = claims.groupby("VehicleCategory").agg(
    total_claims=("is_fraud", "size"),
    fraud_claims=("is_fraud", "sum"),
    fraud_rate=("is_fraud", "mean")
)

fraud_by_vehicle = fraud_by_vehicle.sort_values(
    by="fraud_rate",
    ascending=False
)

print("\n--- FRAUD BY VEHICLE CATEGORY ---")
print(fraud_by_vehicle)
# LESSON 4D - FRAUD DATA VISUALIZATION

# 1. Create the reports folder if needed
reports_folder = project_folder / "reports"
reports_folder.mkdir(parents=True, exist_ok=True)

# 2. Prepare accident-area fraud rates in percentages
area_rates = fraud_by_area["fraud_rate"] * 100

# 3. Create a figure and its plotting area
fig, ax = plt.subplots(figsize=(8, 5))

# 4. Draw the bar chart
bars = ax.bar(area_rates.index, area_rates.values)

# 5. Add descriptive labels
ax.set_title("Insurance Fraud Rate by Accident Area")
ax.set_xlabel("Accident Area")
ax.set_ylabel("Fraud-labelled Claims (%)")

# 6. Display values above each bar
ax.bar_label(bars, fmt="%.2f%%", padding=3)
ax.set_ylim(0, area_rates.max() * 1.2)

# 7. Save the chart
plt.tight_layout()
fig.savefig(reports_folder / "fraud_by_area.png", dpi=150)
plt.close(fig)

print("\nAccident-area chart saved successfully!")
# LESSON 4D - VEHICLE CATEGORY CHART

vehicle_rates = (
    fraud_by_vehicle["fraud_rate"] * 100
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(vehicle_rates.index, vehicle_rates.values)

ax.set_title("Fraud Rate by Vehicle Category")
ax.set_xlabel("Vehicle Category")
ax.set_ylabel("Fraud-labelled Claims (%)")

ax.bar_label(bars, fmt="%.2f%%", padding=3)
ax.set_ylim(0, vehicle_rates.max() * 1.2)

plt.xticks(rotation=30, ha="right")
plt.tight_layout()

fig.savefig(reports_folder / "fraud_by_vehicle.png", dpi=150)
plt.close(fig)

print("Vehicle-category chart saved successfully!")
# LESSON 4E - DATA QUALITY

# 1. Count missing values for each column
missing_count = df.isna().sum()

# 2. Calculate the missing percentage
missing_pct = df.isna().mean() * 100

# 3. Create a summary table
quality_report = pd.DataFrame({
    "missing_count": missing_count,
    "missing_pct": missing_pct
})

# 4. Sort columns by missing percentage
quality_report = quality_report.sort_values(
    by="missing_pct",
    ascending=False
)

print("\n--- DATA QUALITY REPORT ---")
print(quality_report.to_string())
quality_report.to_csv(
    reports_folder / "data_quality_summary.csv"
)
# LESSON 4E - DUPLICATE CHECKS

duplicate_rows = df.duplicated().sum()

print("\n--- DUPLICATE RECORDS ---")
print("Exact duplicate rows:", duplicate_rows)

# Check policy identifiers, if available
if "PolicyNumber" in df.columns:
    print(
        "Repeated policy numbers:",
        df["PolicyNumber"].duplicated().sum()
    )
    # LESSON 4E - TARGET VALIDATION

expected_labels = {"Yes", "No"}

actual_labels = set(
    df["FraudFound"].dropna().unique()
)

unexpected_labels = actual_labels - expected_labels

print("\n--- TARGET VALIDATION ---")
print("Actual labels:", actual_labels)
print("Unexpected labels:", unexpected_labels)

print(
    "Missing fraud labels:",
    df["FraudFound"].isna().sum()
)
# LESSON 4E - FEATURE LEAKAGE AUDIT

potential_risks = {
    "FraudFound": "Target variable",
    "PolicyNumber": "Record identifier",
    "RepNumber": "Check assignment timing",
    "NumberOfSuppliments": "Check when supplements are recorded",
    "PoliceReportFiled": "Check availability at scoring time"
}

print("\n--- POTENTIAL LEAKAGE AUDIT ---")

for column, reason in potential_risks.items():
    if column in df.columns:
        print(f"{column}: {reason}")