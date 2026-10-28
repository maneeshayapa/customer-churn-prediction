
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

# Churn overview
churn_counts = df["Churn"].value_counts()
total_customers = len(df)
churned = int(churn_counts.get("Yes", 0))
retained = int(churn_counts.get("No", 0))

churn_rate = churned / total_customers * 100

print("\nCUSTOMER CHURN OVERVIEW")
print(f"Total customers: {total_customers}")
print(f"Churned customers: {churned}")
print(f"Retained customers: {retained}")
print(f"Churn rate: {churn_rate:.2f}%")

# Churn by contract type
contract_summary = (
    df.groupby("Contract", observed=False)
    .agg(
        Customers=("Churn", "size"),
        Churned=("Churn", lambda s: (s == "Yes").sum()),
    )
    .reset_index()
)

contract_summary["ChurnRatePercent"] = (
    contract_summary["Churned"]
    / contract_summary["Customers"]
    * 100
)

contract_summary = contract_summary.sort_values(
    "ChurnRatePercent", ascending=False
)

print("\nCHURN BY CONTRACT TYPE")
print(contract_summary.round(2).to_string(index=False))

contract_summary.to_csv(
    REPORT_DIR / "churn_by_contract.csv",
    index=False,
)

# Churn by tenure group
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months",
            "25-48 months", "49-72 months"],
)

tenure_summary = (
    df.groupby("TenureGroup", observed=False)
    .agg(
        Customers=("Churn", "size"),
        Churned=("Churn", lambda s: (s == "Yes").sum()),
    )
    .reset_index()
)

tenure_summary["ChurnRatePercent"] = (
    tenure_summary["Churned"]
    / tenure_summary["Customers"]
    * 100
)

print("\nCHURN BY TENURE GROUP")
print(tenure_summary.round(2).to_string(index=False))

tenure_summary.to_csv(
    REPORT_DIR / "churn_by_tenure.csv",
    index=False,
)

# Create contract churn chart
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    contract_summary["Contract"],
    contract_summary["ChurnRatePercent"],
)

ax.set_title("Customer Churn Rate by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Churn Rate (%)")

fig.tight_layout()
fig.savefig(
    REPORT_DIR / "churn_by_contract.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Create tenure churn chart
fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    tenure_summary["TenureGroup"].astype(str),
    tenure_summary["ChurnRatePercent"],
)

ax.set_title("Customer Churn Rate by Tenure")
ax.set_xlabel("Customer Tenure")
ax.set_ylabel("Churn Rate (%)")

fig.tight_layout()
fig.savefig(
    REPORT_DIR / "churn_by_tenure.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Generate a concise business report
report_path = REPORT_DIR / "business_insights_report.md"

highest_contract = contract_summary.iloc[0]
highest_tenure = tenure_summary.loc[
    tenure_summary["ChurnRatePercent"].idxmax()
]

report = f"""# Customer Churn — Business Insights Report

## 1. Dataset overview

- Total customers analyzed: {total_customers}
- Customers who churned: {churned}
- Customers retained: {retained}
- Overall churn rate: {churn_rate:.2f}%

## 2. Contract analysis

The highest observed churn rate among contract groups is
**{highest_contract['Contract']}** at
**{highest_contract['ChurnRatePercent']:.2f}%**.

See `churn_by_contract.csv` and `churn_by_contract.png`.

## 3. Tenure analysis

The highest observed churn rate among tenure groups is
**{highest_tenure['TenureGroup']}** at
**{highest_tenure['ChurnRatePercent']:.2f}%**.

See `churn_by_tenure.csv` and `churn_by_tenure.png`.

## 4. Suggested business actions

- Review the customer experience for month-to-month customers.
- Provide onboarding and early satisfaction check-ins for new customers.
- Investigate pricing and service concerns before offering discounts.
- Measure retention campaign outcomes before scaling them.

## 5. Limitations

These findings describe associations in a historical dataset.
They do not prove that contract type, tenure, or charges cause churn.
Retention actions should be tested and evaluated using real outcomes.
Model predictions are estimates and should not be treated as certainties.
"""

report_path.write_text(report, encoding="utf-8")

print("\nBusiness insights files saved:")
print(REPORT_DIR / "churn_by_contract.csv")
print(REPORT_DIR / "churn_by_tenure.csv")
print(REPORT_DIR / "churn_by_contract.png")
print(REPORT_DIR / "churn_by_tenure.png")
print(report_path)
