
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
OUTPUT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Load saved model and threshold
saved = joblib.load(MODEL_PATH)
model = saved["model"]
threshold = saved["threshold"]

# Predict probabilities and classes
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

# Calculate evaluation metrics
metrics = {
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
}

print("\nMODEL EVALUATION")
print("-" * 35)

for name, value in metrics.items():
    print(f"{name:12}: {value:.4f}")

print(f"\nDecision threshold: {threshold:.2f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probabilities)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {metrics['ROC-AUC']:.3f})",
)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()

# Save metrics
pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Score"],
).to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)

print("\nEvaluation completed!")
print(f"Reports saved in: {OUTPUT_DIR}")
