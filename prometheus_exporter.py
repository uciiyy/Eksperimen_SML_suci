"""
prometheus_exporter.py
======================
Real Prometheus Exporter — mengambil metrik langsung dari inference server.
Bukan simulasi, data diambil real-time dari /metrics endpoint.

Cara pakai:
    python prometheus_exporter.py

Kemudian tambahkan ke prometheus.yml:
    - job_name: 'ml_custom_exporter'
      static_configs:
        - targets: ['localhost:8000']
"""

import time
import requests
from prometheus_client import start_http_server, Gauge

# ─────────────────────────────────────────────
# DEFINISI METRICS REAL
# ─────────────────────────────────────────────

# 1. Status model (1=up, 0=down)
MODEL_UP = Gauge(
    "ml_model_up",
    "Status inference server (1=up, 0=down)"
)

# 2. Total request sukses
TOTAL_SUCCESS = Gauge(
    "ml_total_requests_success",
    "Total request prediksi yang sukses (status 200)"
)

# 3. Total request error
TOTAL_ERROR = Gauge(
    "ml_total_requests_error",
    "Total request prediksi yang error (status 500)"
)

# 4. Total prediksi kelas Selamat
PRED_SELAMAT = Gauge(
    "ml_prediction_selamat_total",
    "Total prediksi kelas Selamat"
)

# 5. Total prediksi kelas Tidak Selamat
PRED_TIDAK_SELAMAT = Gauge(
    "ml_prediction_tidak_selamat_total",
    "Total prediksi kelas Tidak Selamat"
)

# 6. Jumlah prediksi sedang diproses
IN_FLIGHT = Gauge(
    "ml_predictions_in_flight_current",
    "Jumlah prediksi yang sedang diproses saat ini"
)

# 7. Total error missing features
ERROR_MISSING_FEATURES = Gauge(
    "ml_error_missing_features_total",
    "Total error karena fitur kurang"
)

INFERENCE_URL = "http://localhost:5001"


# ─────────────────────────────────────────────
# PARSE METRIK DARI INFERENCE SERVER
# ─────────────────────────────────────────────
def parse_metric_value(metrics_text, metric_name, labels=None):
    """Parse nilai metrik dari teks Prometheus format."""
    for line in metrics_text.split('\n'):
        if line.startswith('#'):
            continue
        if metric_name not in line:
            continue
        if labels:
            if all(f'{k}="{v}"' in line for k, v in labels.items()):
                try:
                    return float(line.split()[-1])
                except:
                    pass
        else:
            if '{' not in line:
                try:
                    return float(line.split()[-1])
                except:
                    pass
    return 0.0


# ─────────────────────────────────────────────
# COLLECT METRICS DARI INFERENCE SERVER
# ─────────────────────────────────────────────
def collect_metrics():
    while True:
        try:
            # Cek health inference server
            health_resp = requests.get(f"{INFERENCE_URL}/health", timeout=3)
            if health_resp.status_code == 200 and health_resp.json().get("status") == "ok":
                MODEL_UP.set(1)
            else:
                MODEL_UP.set(0)

            # Ambil semua metrics dari /metrics endpoint
            metrics_resp = requests.get(f"{INFERENCE_URL}/metrics", timeout=3)
            metrics_text = metrics_resp.text

            # Parse setiap metrik
            TOTAL_SUCCESS.set(parse_metric_value(
                metrics_text, "prediction_requests_total",
                {"method": "POST", "endpoint": "/predict", "status": "200"}
            ))

            TOTAL_ERROR.set(parse_metric_value(
                metrics_text, "prediction_requests_total",
                {"method": "POST", "endpoint": "/predict", "status": "500"}
            ))

            PRED_SELAMAT.set(parse_metric_value(
                metrics_text, "prediction_class_total",
                {"predicted_class": "Selamat"}
            ))

            PRED_TIDAK_SELAMAT.set(parse_metric_value(
                metrics_text, "prediction_class_total",
                {"predicted_class": "Tidak Selamat"}
            ))

            IN_FLIGHT.set(parse_metric_value(
                metrics_text, "predictions_in_flight"
            ))

            ERROR_MISSING_FEATURES.set(parse_metric_value(
                metrics_text, "prediction_errors_total",
                {"error_type": "missing_features"}
            ))

            print(f"[Exporter] Metrics updated successfully")

        except requests.exceptions.ConnectionError:
            MODEL_UP.set(0)
            print(f"[Exporter] Inference server tidak bisa dijangkau!")
        except Exception as e:
            MODEL_UP.set(0)
            print(f"[Exporter] Error: {e}")

        time.sleep(10)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    start_http_server(8000)
    print("[Exporter] Server berjalan di http://localhost:8000/metrics")
    print("[Exporter] Mengambil data real dari http://localhost:5001/metrics")
    print("[Exporter] Tekan Ctrl+C untuk berhenti")
    collect_metrics()