"""
modelling.py
============
Melatih model Random Forest Classifier pada dataset Titanic preprocessing.
Menggunakan MLflow autolog + manual logging untuk artefak lengkap.

Cara pakai:
    python modelling.py
"""

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


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

    print(f"Train : {X_train.shape}")
    print(f"Test  : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, path="confusion_matrix.png"):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Selamat", "Selamat"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix – Test Set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[artifact] saved → {path}")
    return path


def save_feature_importance(model, feature_names, path="feature_importance.png"):
    importances  = model.feature_importances_
    indices      = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names_sorted[::-1], importances[indices[::-1]], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance – Random Forest")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[artifact] saved → {path}")
    return path


def save_training_history(model, X_train, y_train, X_test, y_test,
                          path="training_history.png"):
    """Learning curve: akurasi train & test vs jumlah estimator."""
    step       = max(1, model.n_estimators // 10)
    n_values   = list(range(step, model.n_estimators + 1, step))
    train_acc  = []
    test_acc   = []

    for n in n_values:
        sub = RandomForestClassifier(
            n_estimators=n,
            max_depth=model.max_depth,
            random_state=model.random_state,
        )
        sub.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, sub.predict(X_train)))
        test_acc.append(accuracy_score(y_test,  sub.predict(X_test)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_values, train_acc, marker="o", label="Train Accuracy")
    ax.plot(n_values, test_acc,  marker="s", label="Test Accuracy")
    ax.set_xlabel("Number of Estimators")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training History – Accuracy vs n_estimators")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[artifact] saved → {path}")
    return path


# ─────────────────────────────────────────────
# TRAIN & LOG
# ─────────────────────────────────────────────
def train_model():
    X_train, X_test, y_train, y_test = load_data()

    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Autolog dengan log_models=True agar folder model/ terbentuk
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
    )

    with mlflow.start_run(run_name=RUN_NAME) as run:

        # ── Train ────────────────────────────────────────────────
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # ── Evaluate ─────────────────────────────────────────────
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)

        print(f"\nAkurasi Test Set : {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=["Tidak Selamat", "Selamat"]
        ))

        # ── Manual logging ───────────────────────────────────────
        # Metrik tambahan
        mlflow.log_metric("test_accuracy", acc)

        # Artefak visual
        mlflow.log_artifact(
            save_confusion_matrix(y_test, y_pred)
        )
        mlflow.log_artifact(
            save_feature_importance(model, list(X_train.columns))
        )
        mlflow.log_artifact(
            save_training_history(model, X_train, y_train, X_test, y_test)
        )

        print(f"\n[MLflow] Experiment : {EXPERIMENT_NAME}")
        print(f"[MLflow] Run ID     : {run.info.run_id}")

    print("\n[MLflow] Selesai. Jalankan MLflow UI dengan:")
    print("    mlflow ui")
    print("    Buka → http://127.0.0.1:5000")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_model()