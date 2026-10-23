"""Train a regression model and a classification model on the insurance dataset."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r"E:\UNI\5th sem\ML lab\Assignment 1\insurance-ml-app\data\insurance.csv")
CATEGORICAL = ["sex", "region"]

# ---------------------------------------------------------------
# 1) Regression: predict insurance charges (a number)
# ---------------------------------------------------------------
reg_features = ["age", "sex", "bmi", "children", "smoker", "region"]
X = df[reg_features]
y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])],
        remainder="passthrough")),
    ("model", LinearRegression()),
])
reg_model.fit(X_train, y_train)
r2 = r2_score(y_test, reg_model.predict(X_test))
print(f"Linear Regression  R2 score: {r2:.3f}")
joblib.dump(reg_model, "regression_model.joblib")

# ---------------------------------------------------------------
# 2) Classification: predict if a person is a smoker (yes / no)
# ---------------------------------------------------------------
clf_features = ["age", "sex", "bmi", "children", "region", "charges"]
X = df[clf_features]
y = df["smoker"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

clf_model = Pipeline([
    ("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough")),
    ("model", RandomForestClassifier(n_estimators=200, random_state=42)),
])
clf_model.fit(X_train, y_train)
acc = accuracy_score(y_test, clf_model.predict(X_test))
print(f"Random Forest      Accuracy : {acc:.3f}")
joblib.dump(clf_model, "classification_model.joblib")

print("Models saved.")