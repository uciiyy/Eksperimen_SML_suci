# 📦 MSML Submission — Titanic Classification
**Nama Siswa:** NamaSiswa  
**Dataset:** Titanic (Seaborn built-in)  
**Target:** Skilled (3 pts) per kriteria

---

## 🗂️ Struktur Repository

```
.
├── Eksperimen_SML_NamaSiswa/          ← Kriteria 1
│   ├── .github/workflows/
│   │   └── preprocessing.yml
│   ├── titanic_raw/
│   │   └── titanic_raw.csv
│   ├── preprocessing/
│   │   ├── Eksperimen_NamaSiswa.ipynb
│   │   └── automate_NamaSiswa.py
│   └── titanic_preprocessing/
│       ├── titanic_train.csv
│       └── titanic_test.csv
│
├── Membangun_model/                   ← Kriteria 2
│   ├── modelling.py                  (Basic — autolog)
│   ├── modelling_tuning.py           (Skilled — manual log + tuning)
│   ├── titanic_preprocessing/
│   ├── requirements.txt
│   ├── screenshot_dashboard.jpg      (isi sendiri)
│   └── screenshot_artifak.jpg        (isi sendiri)
│
├── Workflow-CI/                       ← Kriteria 3
│   ├── .github/workflows/
│   │   └── ci_training.yml
│   └── MLProject/
│       ├── MLProject
│       ├── conda.yaml
│       ├── modelling.py
│       └── titanic_preprocessing/
│
└── Monitoring_dan_Logging/            ← Kriteria 4
    ├── inference.py
    ├── prometheus_exporter.py
    └── prometheus.yml
```

---

## ✅ Kriteria 1 — Eksperimen & Preprocessing

### Langkah Setup
```bash
cd Eksperimen_SML_NamaSiswa/preprocessing
pip install pandas numpy scikit-learn seaborn matplotlib

# Jalankan notebook
jupyter notebook Eksperimen_NamaSiswa.ipynb

# Atau jalankan script otomatis
python automate_NamaSiswa.py
```

Output: `titanic_raw/titanic_raw.csv` dan `titanic_preprocessing/titanic_train.csv`, `titanic_test.csv`

---

## ✅ Kriteria 2 — Membangun Model dengan MLflow

### Setup
```bash
cd Membangun_model
pip install -r requirements.txt

# Salin dataset hasil preprocessing
cp -r ../Eksperimen_SML_NamaSiswa/titanic_preprocessing .
```

### Basic — autolog
```bash
python modelling.py
```

### Skilled — manual log + hyperparameter tuning
```bash
python modelling_tuning.py
```

### Buka MLflow UI
```bash
mlflow ui
# Akses http://localhost:5000
```

---

## ✅ Kriteria 3 — Workflow CI

### Jalankan MLflow Project secara lokal
```bash
cd Workflow-CI/MLProject
pip install mlflow pandas numpy scikit-learn seaborn matplotlib

# Salin dataset
cp -r ../../Eksperimen_SML_NamaSiswa/titanic_preprocessing .

# Jalankan project
mlflow run . --env-manager=local
mlflow run . -P n_estimators=150 -P max_depth=7 --env-manager=local
```

### GitHub Actions CI
Push ke branch `main` untuk memicu workflow otomatis, atau trigger manual dari tab **Actions** → **MLflow CI Training** → **Run workflow**.

Artefak tersimpan di:
- GitHub Actions Artifacts (30 hari)
- Folder `MLProject/mlruns/` di repository

---

## ✅ Kriteria 4 — Monitoring & Logging

### 1. Serving Model
```bash
cd Monitoring_dan_Logging
pip install flask prometheus_client mlflow scikit-learn pandas

# Jalankan inference server
python inference.py
# → Berjalan di http://localhost:5001
```

### 2. Jalankan Custom Prometheus Exporter
```bash
python prometheus_exporter.py
# → Metrics tersedia di http://localhost:8000/metrics
```

### 3. Jalankan Prometheus
```bash
# Download Prometheus dari https://prometheus.io/download/
./prometheus --config.file=prometheus.yml
# → Akses http://localhost:9090
```

### 4. Jalankan Grafana
```bash
# Setelah install Grafana:
# Akses http://localhost:3000 (default: admin/admin)
# Tambahkan Data Source → Prometheus → http://localhost:9090
# Import dashboard dan buat panel untuk setiap metrik
```

### Metrik yang Tersedia (≥5 untuk Skilled)
| Metrik | Sumber | Deskripsi |
|--------|--------|-----------|
| `prediction_requests_total` | inference.py | Total request masuk |
| `prediction_latency_seconds` | inference.py | Latensi prediksi |
| `prediction_class_total` | inference.py | Distribusi kelas prediksi |
| `prediction_errors_total` | inference.py | Total error prediksi |
| `ml_model_accuracy` | exporter.py | Akurasi model saat ini |
| `ml_total_predictions` | exporter.py | Total prediksi kumulatif |
| `ml_prediction_confidence` | exporter.py | Distribusi confidence |
| `ml_memory_usage_mb` | exporter.py | Penggunaan memori |
| `ml_data_drift_score` | exporter.py | Skor data drift |

### Test Prediksi
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1, "sex": 0, "age": -0.5,
    "sibsp": 0, "parch": 0, "fare": 1.2,
    "embarked": 2, "family_size": 0, "is_alone": 1
  }'
```

---

## 📌 Catatan Penting

1. **Ganti `NamaSiswa`** di semua nama file dan folder dengan nama Anda yang terdaftar di Dicoding.
2. **Screenshot Dashboard Grafana** harus mencantumkan username Dicoding Anda.
3. Untuk Kriteria 4 Alerting (Skilled), buat minimal **1 alert rule** di Grafana, contoh:
   - Alert ketika `prediction_latency_seconds` > 0.5 detik
   - Alert ketika `ml_model_accuracy` < 0.75
