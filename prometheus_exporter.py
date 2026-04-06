"""
prometheus_exporter.py
======================
Custom Prometheus Exporter tambahan untuk monitoring sistem ML.
Menjalankan server metrics pada port 8000 yang discrap oleh Prometheus.

Cara pakai:
    python prometheus_exporter.py

Kemudian tambahkan ke prometheus.yml:
    - job_name: 'ml_custom_exporter'
      static_configs:
        - targets: ['localhost:8000']
"""

import time
import random
import threading
from prometheus_client import (
    start_http_server,
    Counter, Gauge, Histogram, Summary,
)


# ─────────────────────────────────────────────
# DEFINISI METRICS (Minimal 5 untuk Skilled)
# ─────────────────────────────────────────────

# 1. Akurasi model saat ini (disimulasi, di produksi ambil dari evaluasi)
MODEL_ACCURACY = Gauge(
    "ml_model_accuracy",
    "Akurasi model pada dataset evaluasi terkini"
)

# 2. Total prediksi kumulatif
TOTAL_PREDICTIONS = Counter(
    "ml_total_predictions",
    "Total prediksi yang pernah dilakukan sejak server nyala"
)

# 3. Distribusi confidence score
CONFIDENCE_SCORE = Histogram(
    "ml_prediction_confidence",
    "Distribusi confidence score prediksi",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# 4. Penggunaan memori (MB) — simulasi
MEMORY_USAGE_MB = Gauge(
    "ml_memory_usage_mb",
    "Perkiraan penggunaan memori model dalam MB"
)

# 5. Data drift score (simulasi)
DATA_DRIFT_SCORE = Gauge(
    "ml_data_drift_score",
    "Skor data drift antara distribusi training dan produksi (0-1)"
)

# 6. Jumlah batch yang diproses
BATCH_PROCESSED = Counter(
    "ml_batch_processed_total",
    "Total batch data yang sudah diproses"
)

# 7. Latensi preprocessing per request
PREPROCESSING_LATENCY = Summary(
    "ml_preprocessing_latency_seconds",
    "Waktu preprocessing per request"
)


# ─────────────────────────────────────────────
# SIMULASI UPDATE METRICS
# ─────────────────────────────────────────────
def update_metrics():
    """
    Update metrics secara periodik (simulasi produksi).
    Di production, ganti dengan data nyata dari sistem Anda.
    """
    base_accuracy = 0.82

    while True:
        # Simulasi fluktuasi akurasi
        accuracy = base_accuracy + random.uniform(-0.03, 0.03)
        MODEL_ACCURACY.set(round(accuracy, 4))

        # Simulasi prediksi masuk
        n_preds = random.randint(1, 10)
        TOTAL_PREDICTIONS.inc(n_preds)

        # Simulasi confidence scores
        for _ in range(n_preds):
            conf = random.betavariate(8, 2)  # distribusi condong ke kanan
            CONFIDENCE_SCORE.observe(conf)

        # Simulasi memori (antara 80 - 120 MB)
        MEMORY_USAGE_MB.set(random.uniform(80, 120))

        # Simulasi data drift (semakin lama semakin drift)
        drift = random.uniform(0.01, 0.15)
        DATA_DRIFT_SCORE.set(round(drift, 4))

        # Simulasi batch processing
        BATCH_PROCESSED.inc(random.randint(0, 3))

        # Simulasi preprocessing latency
        PREPROCESSING_LATENCY.observe(random.uniform(0.001, 0.05))

        time.sleep(5)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Jalankan HTTP server untuk scraping Prometheus
    start_http_server(8000)
    print("[Prometheus Exporter] Server berjalan di http://localhost:8000/metrics")
    print("[Prometheus Exporter] Tekan Ctrl+C untuk berhenti")

    # Mulai thread update metrics
    t = threading.Thread(target=update_metrics, daemon=True)
    t.start()

    # Jaga proses tetap hidup
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Prometheus Exporter] Berhenti.")
