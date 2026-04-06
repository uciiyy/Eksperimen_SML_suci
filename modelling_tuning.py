"""
modelling_tuning.py
===================
Melatih model Random Forest dengan Hyperparameter Tuning (GridSearchCV).
Menggunakan MLflow MANUAL LOGGING (bukan autolog) — Kriteria 2 Skilled.

Cara pakai:
    python modelling_tuning.py
"""

import os
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATA_DIR   = "titanic_preprocessing"
TRAIN_FILE = os.path.join(DATA_DIR, "titanic_train.csv")
TEST_FILE  = os.path.join(DATA_DIR, "titanic_test.csv")

EXPERIMENT_NAME = "Titanic_Classification"
RUN_NAME        = "RandomForest_HyperparamTuning"


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
# BUAT CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Tidak Selamat", "Selamat"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# BUAT FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
def plot_feature_importance(model, feature_names, save_path="feature_importance.png"):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names_sorted, importances[indices], color="#3498db")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# TRAIN DENGAN TUNING + MANUAL LOGGING
# ─────────────────────────────────────────────
def train_with_tuning():
    X_train, X_test, y_train, y_test = load_data()

    # Grid pencarian hyperparameter
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth"   : [3, 5, 7],
        "min_samples_split": [2, 5],
    }

    base_model = RandomForestClassifier(random_state=42)

    print("\n[GridSearch] Mencari hyperparameter terbaik ...")
    start = time.time()
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model  = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_acc = grid_search.best_score_

    print(f"\n[GridSearch] Selesai dalam {elapsed:.2f} detik")
    print(f"[GridSearch] Best params : {best_params}")
    print(f"[GridSearch] Best CV Acc : {best_cv_acc:.4f}")

    # ── Evaluasi pada test set ──
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)

    # Cross-val score (5-fold) pada best model
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")

    print(f"\nTest Accuracy : {acc:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc_auc:.4f}")
    print(f"CV Mean Acc   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Buat artefak lokal ──
    cm_path = plot_confusion_matrix(y_test, y_pred)
    fi_path = plot_feature_importance(best_model, list(X_train.columns))

    # ── MLflow MANUAL LOGGING ──
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=RUN_NAME):

        # Log hyperparameter terbaik
        mlflow.log_param("n_estimators"    , best_params["n_estimators"])
        mlflow.log_param("max_depth"       , best_params["max_depth"])
        mlflow.log_param("min_samples_split", best_params["min_samples_split"])
        mlflow.log_param("cv_folds"        , 5)
        mlflow.log_param("test_size"       , 0.2)
        mlflow.log_param("random_state"    , 42)

        # Log metrik (sama seperti autolog)
        mlflow.log_metric("accuracy"              , acc)
        mlflow.log_metric("precision"             , precision)
        mlflow.log_metric("recall"                , recall)
        mlflow.log_metric("f1_score"              , f1)
        mlflow.log_metric("roc_auc"               , roc_auc)
        mlflow.log_metric("best_cv_accuracy"      , best_cv_acc)
        mlflow.log_metric("cv_mean_accuracy"      , cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy"       , cv_scores.std())
        mlflow.log_metric("training_time_seconds" , elapsed)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path="model")

        # Log artefak tambahan
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(fi_path, artifact_path="plots")

        # Log classification report sebagai teks
        report = classification_report(
            y_test, y_pred,
            target_names=["Tidak Selamat", "Selamat"]
        )
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        run_id = mlflow.active_run().info.run_id
        print(f"\n[MLflow] Run ID: {run_id}")

    print("\n[MLflow] Run selesai. Buka MLflow UI dengan:")
    print("    mlflow ui")
    print("    kemudian akses http://localhost:5000")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_with_tuning()
