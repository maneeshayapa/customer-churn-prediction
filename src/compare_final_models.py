
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

# Load and clean the dataset
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Recreate the same test split used during training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Load both saved models
model_files = {
    "Baseline Logistic Regression": MODEL_DIR / "churn_model.joblib",
    "Tuned Logistic Regression": MODEL_DIR / "churn_model_tuned.joblib",
}

results = []
predictions_by_model = {}

for model_name, model_path in model_files.items():
    saved = joblib.load(model_path)

    model = saved["model"]
    threshold = float(saved["threshold"])

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    predictions_by_model[model_name] = predictions

    results.append({
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })

# Save comparison metrics
results_df = pd.DataFrame(results)
results_path = REPORT_DIR / "final_model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\nFINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

# Select the model with the highest test F1-score
# This is a descriptive comparison, not further model tuning.
best_index = results_df["F1-score"].idxmax()
best_name = results_df.loc[best_index, "Model"]

print(f"\nHighest test F1-score: {best_name}")
print(
    "Note: Choose the deployed model based on business needs "
    "as well as these test results."
)

# Create a metrics comparison chart
metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]

ax = results_df.set_index("Model")[metrics].T.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_title("Customer Churn Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("Evaluation Metric")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()

chart_path = REPORT_DIR / "model_comparison.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

# Create separate confusion matrices
for model_name, predictions in predictions_by_model.items():
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    )
    display.plot(values_format="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    filename = (
        "confusion_matrix_baseline.png"
        if model_name == "Baseline Logistic Regression"
        else "confusion_matrix_tuned.png"
    )

    plt.savefig(REPORT_DIR / filename, dpi=300)
    plt.close()

print("\nReports saved:")
print(results_path)
print(chart_path)
print(REPORT_DIR / "confusion_matrix_baseline.png")
print(REPORT_DIR / "confusion_matrix_tuned.png")
