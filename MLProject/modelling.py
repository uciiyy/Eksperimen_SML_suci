"""
modelling.py  (versi MLProject — Kriteria 3)
=============================================
Sama seperti modelling_tuning.py, tetapi menerima argumen CLI
sehingga bisa dipanggil oleh MLflow Projects runner.

Cara pakai manual:
    python modelling.py --n_estimators 100 --max_depth 5

Cara pakai via MLflow Projects:
    mlflow run . -P n_estimators=100 -P max_depth=5
"""

import os
import argparse
import time
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# ARGPARSE
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Titanic Classifier")
    parser.add_argument("--n_estimators"     , type=int  , default=100)
    parser.add_argument("--max_depth"        , type=int  , default=5)
    parser.add_argument("--min_samples_split", type=int  , default=2)
    parser.add_argument("--test_size"        , type=float, default=0.2)
    return parser.parse_args()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def load_data(data_dir="titanic_preprocessing"):
    train_df = pd.read_csv(os.path.join(data_dir, "titanic_train.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "titanic_test.csv"))
    X_train  = train_df.drop("survived", axis=1)
    y_train  = train_df["survived"]
    X_test   = test_df.drop("survived", axis=1)
    y_test   = test_df["survived"]
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Tidak Selamat", "Selamat"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_feature_importance(model, feature_names, save_path="feature_importance.png"):
    importances  = model.feature_importances_
    indices      = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in indices]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names_sorted, importances[indices], color="#2ecc71")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────
def main():
    args = parse_args()

    X_train, X_test, y_train, y_test = load_data()

    # ✅ set_experiment() tetap ada, tapi harus matching dengan --experiment-name di CI
    mlflow.set_experiment("Titanic_CI_Pipeline")

    with mlflow.start_run():  # ✅ WAJIB ada ini agar tidak konflik dengan mlflow run
        start = time.time()

        model = RandomForestClassifier(
            n_estimators      = args.n_estimators,
            max_depth         = args.max_depth,
            min_samples_split = args.min_samples_split,
            random_state      = 42,
        )
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        roc_auc   = roc_auc_score(y_test, y_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        # Log params
        mlflow.log_param("n_estimators"     , args.n_estimators)
        mlflow.log_param("max_depth"        , args.max_depth)
        mlflow.log_param("min_samples_split", args.min_samples_split)
        mlflow.log_param("test_size"        , args.test_size)

        # Log metrics
        mlflow.log_metric("accuracy"            , acc)
        mlflow.log_metric("precision"           , precision)
        mlflow.log_metric("recall"              , recall)
        mlflow.log_metric("f1_score"            , f1)
        mlflow.log_metric("roc_auc"             , roc_auc)
        mlflow.log_metric("cv_mean_accuracy"    , cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy"     , cv_scores.std())
        mlflow.log_metric("training_time_seconds", elapsed)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Plot & log artifacts
        cm_path = plot_confusion_matrix(y_test, y_pred)
        fi_path = plot_feature_importance(model, list(X_train.columns))
        mlflow.log_artifact(cm_path, "plots")
        mlflow.log_artifact(fi_path, "plots")

        report = classification_report(
            y_test, y_pred,
            target_names=["Tidak Selamat", "Selamat"]
        )
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc_auc:.4f}")
        print(f"CV Mean  : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Run ID   : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()