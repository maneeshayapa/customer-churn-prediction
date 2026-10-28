
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load and clean data
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# Split data before model tuning
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore"),
    ),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# Complete ML pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(max_iter=2000, random_state=42),
    ),
])

# Hyperparameter combinations to test
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    refit=True,
    return_train_score=True,
)

print("Training and comparing model configurations...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBEST PARAMETERS")
print(search.best_params_)

print("\nBEST 5-FOLD CROSS-VALIDATION F1")
print(round(search.best_score_, 4))

# Choose a threshold using out-of-fold training predictions
oof_probabilities = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

thresholds = [round(t, 2) for t in
              [0.20 + i * 0.01 for i in range(51)]]

threshold_results = []

for threshold in thresholds:
    oof_predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(
            y_train, oof_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_train, oof_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_train, oof_predictions, zero_division=0
        ),
    })

threshold_df = pd.DataFrame(threshold_results)

# Select the threshold with the highest cross-validated F1
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
threshold = float(best_row["threshold"])

print("\nSELECTED THRESHOLD")
print(f"Threshold: {threshold:.2f}")
print(f"Training OOF Precision: {best_row['precision']:.4f}")
print(f"Training OOF Recall:    {best_row['recall']:.4f}")
print(f"Training OOF F1-score:  {best_row['f1']:.4f}")

threshold_df.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False,
)

# Evaluate once on the untouched test set
probabilities = best_model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

print("\nFINAL TEST SET RESULTS")
print(f"Decision threshold: {threshold:.2f}")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(
    f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}"
)
print(
    f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}"
)
print(f"F1-score:  {f1_score(y_test, predictions):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probabilities):.4f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

# Save the tuned model and its selected threshold
joblib.dump(
    {
        "model": best_model,
        "threshold": threshold,
        "features": list(X.columns),
    },
    MODEL_DIR / "churn_model_tuned.joblib",
)

print("\nUpdated tuned model saved successfully.")
print(MODEL_DIR / "churn_model_tuned.joblib")
print(REPORT_DIR / "threshold_comparison.csv")


# Save every tested configuration
cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results.sort_values(
    by="rank_test_score"
)
cv_results.to_csv(
    REPORT_DIR / "hyperparameter_results.csv",
    index=False,
)

# Save separately so the current app model remains unchanged
joblib.dump(
    {
        "model": best_model,
        "threshold": threshold,
        "features": list(X.columns),
    },
    MODEL_DIR / "churn_model_tuned.joblib",
)

print("\nSaved:")
print(MODEL_DIR / "churn_model_tuned.joblib")
print(REPORT_DIR / "hyperparameter_results.csv")
