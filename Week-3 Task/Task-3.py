#Task-3

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("Telco_Customer_Churn.csv")

# Turning TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

# -----------------------
# 2. Feature Engineering
# -----------------------
# Turning service into binary
service_cols = ["PhoneService","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]

for col in service_cols:
    if col in df.columns:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
        df[col] = df[col].map({"Yes":1, "No":0})

# new feature
df["service_count"] = df[service_cols].sum(axis=1)
df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].replace(0,1)

# -----------------------
# 3. StandardScaler
# -----------------------
scaler = StandardScaler()

num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "service_count", "avg_monthly_charge"]

df[num_cols] = scaler.fit_transform(df[num_cols])

# -----------------------
# 4. Simple Plot
# -----------------------
df["service_count"].hist(bins=10)
plt.title("Scaled Service Count Distribution")
plt.show()

# -----------------------
# 5. Save cleaned data
# -----------------------
df.to_csv("telco_scaled.csv", index=False)
print("Done! Data scaled and saved.")