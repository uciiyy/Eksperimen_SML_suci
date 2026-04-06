"""
modelling.py
============
Melatih model Random Forest Classifier pada dataset Titanic preprocessing.
Menggunakan MLflow autolog untuk mencatat eksperimen.

Cara pakai:
    python modelling.py
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATA_DIR   = "titanic_preprocessing"
TRAIN_FILE = os.path.join(DATA_DIR, "titanic_train.csv")
TEST_FILE  = os.path.join(DATA_DIR, "titanic_test.csv")

EXPERIMENT_NAME = "Titanic_Classification"
RUN_NAME        = "RandomForest_Autolog"


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df  = pd.read_csv(TEST_FILE)

    X_train = train_df.drop("survived", axis=1)
    y_train = train_df["survived"]
    X_test  = test_df.drop("survived", axis=1)
    y_test  = test_df["survived"]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# TRAIN & LOG
# ─────────────────────────────────────────────
def train_model():
    X_train, X_test, y_train, y_test = load_data()

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name=RUN_NAME):

        # Inisialisasi dan latih model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        print(f"\nAkurasi pada Test Set: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Tidak Selamat", "Selamat"]))

    print("\n[MLflow] Run selesai. Buka MLflow UI dengan:")
    print("    mlflow ui")
    print("    kemudian akses http://localhost:5000")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_model()
