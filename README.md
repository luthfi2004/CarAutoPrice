# AutoPrice AI — Prediksi Harga Mobil
## Final Project Mata Kuliah Sains Data

---

## 📁 Struktur File

```
project/
├── app.py                  ← Backend Flask API (semua kode ML ada di sini)
├── index.html              ← Frontend Web (tampilan untuk end-user)
├── Car_sales.xls           ← Dataset (letakkan di folder yang sama)
├── requirements.txt        ← Daftar library Python
└── README.md
```

---

## 🚀 Cara Menjalankan

### Langkah 1: Install Library
```bash
pip install -r requirements.txt
```

### Langkah 2: Pastikan File Dataset Ada
Letakkan file `Car_sales.xls` di folder yang sama dengan `app.py`.

### Langkah 3: Jalankan Server Flask
```bash
python app.py
```
Output di terminal:
```
✅ Data berhasil di-load! (157 baris, 16 kolom)
✅ Model berhasil dilatih!
   RMSE : x.xxxx
   R²   : x.xxxx
🚀 Server Flask berjalan di http://localhost:5000
```

### Langkah 4: Buka Frontend
Buka file `index.html` di browser, **atau** akses:
```
http://localhost:5000
```

---

## 🌐 API Endpoints

| Method | Endpoint | Keterangan |
|--------|----------|------------|
| GET | `/api/model-info` | Info & koefisien model |
| GET | `/api/statistik` | Statistik dataset |
| POST | `/api/prediksi` | Prediksi harga (JSON body) |
| GET | `/api/top10-penjualan` | Data top 10 penjualan |
| GET | `/api/chart/top10-penjualan` | Chart PNG base64 |
| GET | `/api/chart/harga-top10` | Chart PNG base64 |
| GET | `/api/chart/evaluasi` | Scatter plot evaluasi |
| GET | `/api/chart/korelasi` | Heatmap korelasi |

### Contoh POST /api/prediksi
```json
{
  "vehicle_type": "Passenger",
  "engine_size": 3.0,
  "horsepower": 200,
  "wheelbase": 108.0,
  "curb_weight": 3.3,
  "fuel_capacity": 16.0,
  "fuel_efficiency": 27.0,
  "power_perf_factor": 80.0
}
```

### Response:
```json
{
  "success": true,
  "harga_ribu": 28.543,
  "harga_usd": 28543.00,
  "harga_format": "$28,543"
}
```

---

## 🛠️ Teknologi

- **Backend**: Python · Flask · scikit-learn · pandas · matplotlib
- **Frontend**: HTML · CSS · Vanilla JavaScript
- **Komunikasi**: REST API (JSON over HTTP) + CORS
- **Metode ML**: Linear Regression (CRISP-DM)
