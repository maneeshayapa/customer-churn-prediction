
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load saved pipeline
saved = joblib.load(MODEL_PATH)
model = saved["model"]

# Get preprocessing and logistic regression steps
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()
coefficients = classifier.coef_[0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
})

# Positive coefficients increase predicted churn odds;
# negative coefficients decrease predicted churn odds.
importance["AbsoluteCoefficient"] = (
    importance["Coefficient"].abs()
)

importance = importance.sort_values(
    "AbsoluteCoefficient", ascending=False
)

print("\nTOP 15 FEATURES BY ABSOLUTE COEFFICIENT")
print(
    importance[
        ["Feature", "Coefficient"]
    ].head(15).to_string(index=False)
)

# Plot strongest positive and negative coefficients
top = importance.head(15).sort_values("Coefficient")

plt.figure(figsize=(10, 7))
plt.barh(top["Feature"], top["Coefficient"])
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Logistic Regression Coefficient")
plt.title("Top 15 Churn-Related Model Coefficients")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_coefficients.png",
    dpi=150,
)
plt.close()

importance.to_csv(
    OUTPUT_DIR / "feature_coefficients.csv",
    index=False,
)

print("\nSaved feature_coefficients.png")
print("Saved feature_coefficients.csv")
