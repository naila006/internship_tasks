
# ==========================
# Telco Churn - Data Prep Only
# ==========================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load dataset
df = pd.read_csv("Telco_Customer_Churn.csv")   # replace with your file path

# 2. Drop ID column
df.drop("customerID", axis=1, inplace=True)

# 3. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 4. Handle missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# 5. Separate features (X) and target (y)
y = df["Churn"].map({"Yes": 1, "No": 0})
X = df.drop("Churn", axis=1)

# 6. Identify columns
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

# 7. Define preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# 8. Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# 9. Transform data (prepared form)
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared  = preprocessor.transform(X_test)

# 10. Check shapes
print("Original shape :", df.shape)
print("Train prepared :", X_train_prepared.shape)
print("Test prepared  :", X_test_prepared.shape)