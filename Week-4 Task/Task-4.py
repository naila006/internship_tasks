#Task-4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ======================
# 1. Load & Preprocess
# ======================
df = pd.read_csv("Telco_Customer_Churn.csv")
print(df)

# Drop ID
df = df.drop("customerID", axis=1)
print(df)

# Encode categorical
for col in df.select_dtypes(include="object"):
    df[col] = LabelEncoder().fit_transform(df[col])

# Features & Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=78)

# ======================
# 2. Models
# ======================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=78),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=78),
    "Gradient Boosting": GradientBoostingClassifier(random_state=78),
}

# Try XGBoost
from xgboost import XGBClassifier
models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=78)

# Try LightGBM
from lightgbm import LGBMClassifier
models["LightGBM"] = LGBMClassifier(random_state=78)


# ======================
# 3. Train & Evaluate
# ======================
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))