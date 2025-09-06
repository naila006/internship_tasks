
# ============================
# Telco Customer Churn - EDA
# ============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Telco_Customer_Churn.csv")    

# Drop customerID (not useful for ML)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric (some may be blank)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing TotalCharges with 0 (or median)
df["TotalCharges"].fillna(0, inplace=True)

# ---------------------------
# 1. Basic Info
# ---------------------------
print("\nDataset Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# ---------------------------
# 2. Distribution Plots
# ---------------------------
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# ---------------------------
# 3. Churn Count
# ---------------------------
plt.figure(figsize=(5,4))
sns.countplot(data=df, x="Churn", palette="pastel")
plt.title("Churn Distribution")
plt.show()

# ---------------------------
# 4. Churn by Contract Type
# ---------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
plt.title("Churn by Contract Type")
plt.show()

# ---------------------------
# 5. Churn by Internet Service
# ---------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="InternetService", hue="Churn", palette="coolwarm")
plt.title("Churn by Internet Service")
plt.show()

# ---------------------------
# 6. Correlation Heatmap (Numerical Features)
# ---------------------------
plt.figure(figsize=(6,4))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap="Blues")
plt.title("Correlation Heatmap")
plt.show()

input("Wait for me.......")