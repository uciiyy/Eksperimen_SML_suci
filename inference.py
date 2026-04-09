"""
inference.py
============
Serving model MLflow sebagai REST API menggunakan Flask + Prometheus Exporter.
Kriteria 4: Sistem Monitoring dan Logging.

Cara pakai:
    1. Pastikan MLflow model sudah ada di folder mlruns/
    2. Jalankan:  python inference.py
    3. Akses API: POST http://localhost:5001/predict

Contoh request:
    curl -X POST http://localhost:5001/predict \
      -H "Content-Type: application/json" \
      -d '{"pclass":1,"sex":1,"age":0.5,"sibsp":0,"parch":0,"fare":1.2,"embarked":2,"family_size":0,"is_alone":1}'
"""

import os
import time
import logging
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────

# 1. Total prediksi yang masuk
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total jumlah request prediksi yang diterima",
    ["method", "endpoint", "status"]
)

# 2. Latensi prediksi
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latensi request prediksi dalam detik",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# 3. Distribusi hasil prediksi
PREDICTION_DISTRIBUTION = Counter(
    "prediction_class_total",
    "Distribusi hasil prediksi per kelas",
    ["predicted_class"]
)

# 4. Prediksi yang sedang diproses
PREDICTIONS_IN_FLIGHT = Gauge(
    "predictions_in_flight",
    "Jumlah prediksi yang sedang diproses saat ini"
)

# 5. Jumlah error
PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total error saat melakukan prediksi",
    ["error_type"]
)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    """
    Load model dari mlruns lokal secara otomatis.
    """
    models_path = os.path.join("mlruns", "1", "models")

    if os.path.exists(models_path):
        # Ambil semua folder model
        model_folders = sorted(
            [d for d in os.listdir(models_path)
             if os.path.isdir(os.path.join(models_path, d))],
            reverse=True
        )
        for folder in model_folders:
            artifacts_path = os.path.join(models_path, folder, "artifacts")
            if os.path.exists(os.path.join(artifacts_path, "MLmodel")):
                logger.info(f"Model ditemukan di: {artifacts_path}")
                return mlflow.sklearn.load_model(artifacts_path)

    raise Exception(
        "Model tidak ditemukan! Pastikan mlruns/1/models/ ada."
    )


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)

# Load model saat startup
try:
    model = load_model()
    logger.info("Model berhasil dimuat!")
except Exception as e:
    logger.error(f"Gagal memuat model: {e}")
    model = None

FEATURE_COLS = [
    "pclass", "sex", "age", "sibsp", "parch",
    "fare", "embarked", "family_size", "is_alone"
]


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint prediksi survival Titanic.
    Body JSON: nilai fitur sesuai FEATURE_COLS
    """
    PREDICTIONS_IN_FLIGHT.inc()
    start_time = time.time()

    try:
        data = request.get_json(force=True)

        if model is None:
            PREDICTION_ERRORS.labels(error_type="model_not_loaded").inc()
            return jsonify({"error": "Model belum dimuat"}), 503

        # Validasi fitur
        missing_features = [f for f in FEATURE_COLS if f not in data]
        if missing_features:
            PREDICTION_ERRORS.labels(error_type="missing_features").inc()
            return jsonify({"error": f"Fitur kurang: {missing_features}"}), 400

        # Buat DataFrame
        input_df = pd.DataFrame([{f: data[f] for f in FEATURE_COLS}])

        # Prediksi
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][prediction])
        label = "Selamat" if prediction == 1 else "Tidak Selamat"

        # Update metrics
        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        PREDICTION_DISTRIBUTION.labels(predicted_class=label).inc()

        logger.info(f"Prediksi: {label} (prob={probability:.4f}, latency={latency:.4f}s)")

        return jsonify({
            "prediction" : prediction,
            "label"      : label,
            "probability": round(probability, 4),
            "latency_ms" : round(latency * 1000, 2),
        })

    except Exception as e:
        PREDICTION_ERRORS.labels(error_type="runtime_error").inc()
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        logger.error(f"Error saat prediksi: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        PREDICTIONS_IN_FLIGHT.dec()


@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint metrics untuk Prometheus scrape."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service"  : "Titanic Survival Predictor",
        "endpoints": {
            "/health" : "Health check",
            "/predict": "POST — melakukan prediksi",
            "/metrics": "GET  — Prometheus metrics",
        }
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Inference Server pada port 5001 ...")
    app.run(host="0.0.0.0", port=5001, debug=False)