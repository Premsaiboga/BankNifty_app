import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "training_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# ENCODE STRATEGY (NUMERIC)
# =========================
STRATEGY_MAP = {
    "VWAP_PULLBACK": 0,
    "PIVOT": 1,
    "ABCD": 2
}

df["strategy"] = df["strategy"].map(STRATEGY_MAP)

# =========================
# FEATURES & TARGET
# =========================
FEATURES = [
    "strategy",
    "vwap_distance",
    "candle_size",
    "atr",
    "pattern_strength",
    "rr"
]

X = df[FEATURES]
y = df["result"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test_scaled)

print("\n📊 MODEL PERFORMANCE\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL BUNDLE
# =========================
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "strategy_map": STRATEGY_MAP
    },
    MODEL_PATH
)

print(f"\n✅ Model saved to {MODEL_PATH}")
