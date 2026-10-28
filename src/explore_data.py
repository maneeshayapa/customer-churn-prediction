
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Original dataset shape:", df.shape)

# 2. Remove customerID because it is an identifier
df = df.drop(columns=["customerID"])

# 3. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 4. Handle missing TotalCharges values
df = df.dropna(subset=["TotalCharges"])

print("Cleaned dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())

# 5. Check churn distribution
print("\nChurn counts:")
print(df["Churn"].value_counts())

print("\nChurn percentages:")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

# 6. Plot churn distribution
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 7. Compare churn with contract type
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.show()

# 8. Compare tenure with churn
sns.boxplot(data=df, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.tight_layout()
plt.show()

# 9. Compare monthly charges with churn
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.show()
