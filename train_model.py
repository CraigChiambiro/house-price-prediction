import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/housing.csv")

# =========================
# TARGET
# =========================
y = df["SalePrice"]

# =========================
# FEATURES
# =========================
X = df.drop(["SalePrice", "Id"], axis=1, errors="ignore")

# =========================
# HANDLE MISSING VALUES
# =========================
X = X.fillna(X.median(numeric_only=True))

# =========================
# ENCODE CATEGORICAL DATA
# =========================
X = pd.get_dummies(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

# =========================
# TRAIN
# =========================
model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("\n====================")
print("MODEL MAE:", mae)
print("====================\n")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "models/model.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("Model saved successfully!")