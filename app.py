# ============================================================
# FINAL PROJECT - MATA KULIAH SAINS DATA
# Prediksi Harga Mobil menggunakan CRISP-DM
# Backend API: Flask
# ============================================================
# CARA MENJALANKAN:
#   pip install flask flask-cors pandas numpy scikit-learn
#   python app.py
# Lalu buka index.html di browser (atau akses http://localhost:5000)
# ============================================================

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import base64
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Flask otomatis mencari:
#   templates/index.html  → untuk render_template()
#   static/               → untuk file CSS, JS, gambar statis
app = Flask(__name__)

# CORS eksplisit — izinkan semua origin, method POST+GET, dan header Content-Type
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# ============================================================
# ===========================================================
# LANGKAH 1: MEMANGGIL LIBRARY YANG DIPERLUKAN
# ===========================================================
# (sudah diimport di atas)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ===========================================================
# LANGKAH 2: LOAD DATA
# ===========================================================

DATA_PATH = 'Car_sales.xls'
df = pd.read_excel(DATA_PATH, engine='xlrd')

print(f"✅ Data berhasil di-load! ({df.shape[0]} baris, {df.shape[1]} kolom)")

# ===========================================================
# LANGKAH 3: MELIHAT DATA
# ===========================================================

print("\nLANGKAH 3: EKSPLORASI AWAL DATA")
print(df.head())
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ===========================================================
# LANGKAH 4: MENGHAPUS MISSING VALUE (Drop)
# ===========================================================

print(f"\nJumlah data sebelum drop: {df.shape[0]} baris")
df_clean = df.dropna(subset=['Price_in_thousands'])
print(f"Jumlah data setelah drop baris tanpa harga: {df_clean.shape[0]} baris")

# ===========================================================
# LANGKAH 5: MENGISI MISSING VALUE (Imputation)
# ===========================================================

numeric_cols = ['Engine_size', 'Horsepower', 'Wheelbase', 'Width',
                'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency',
                'Power_perf_factor', '__year_resale_value']

for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"   ✅ Kolom '{col}' diisi dengan median: {median_val:.3f}")

print(f"✅ Total data bersih: {df_clean.shape[0]} baris")

# ===========================================================
# LANGKAH 6: EKSPLORASI DATA (EDA)
# ===========================================================

df_clean['Car_Name'] = df_clean['Manufacturer'] + ' ' + df_clean['Model']
top10_full = df_clean.nlargest(10, 'Sales_in_thousands').reset_index(drop=True)

# ===========================================================
# LANGKAH 7: PENENTUAN VARIABEL
# ===========================================================

features = ['Engine_size', 'Horsepower', 'Wheelbase', 'Width',
            'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency']

le = LabelEncoder()
df_clean['Vehicle_type_encoded'] = le.fit_transform(df_clean['Vehicle_type'])
features_all = features + ['Vehicle_type_encoded']

# ===========================================================
# LANGKAH 8: MEMBUAT MODEL PREDIKSI
# ===========================================================

# 8a. Pisahkan variabel
X = df_clean[features_all].copy()
y = df_clean['Price_in_thousands'].copy()

# 8b. Split 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8c. Buat model
model = LinearRegression()
model.fit(X_train, y_train)

# 8d. Evaluasi
y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
r2   = float(r2_score(y_test, y_pred))
mae  = float(np.mean(np.abs(y_test - y_pred)))

print(f"\n✅ Model berhasil dilatih!")
print(f"   RMSE : {rmse:.4f}")
print(f"   R²   : {r2:.4f}")
print(f"   MAE  : {mae:.4f}")

# Simpan data test untuk scatter plot
y_test_list  = y_test.tolist()
y_pred_list  = y_pred.tolist()

# ============================================================
# HELPER: convert matplotlib figure → base64 PNG string
# ============================================================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

# ============================================================
# ROUTES — Serve HTML
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

# ============================================================
# API 1: Info model & evaluasi
# ============================================================

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'rmse'         : round(rmse, 4),
        'r2'           : round(r2, 4),
        'mae'          : round(mae, 4),
        'total_data'   : int(df.shape[0]),
        'data_bersih'  : int(df_clean.shape[0]),
        'fitur'        : features_all,
        'intercept'    : round(float(model.intercept_), 4),
        'koefisien'    : {f: round(float(c), 6) for f, c in zip(features_all, model.coef_)},
        'train_size'   : X_train.shape[0],
        'test_size'    : X_test.shape[0],
    })

# ============================================================
# API 2: Prediksi harga (POST)
# ===========================================================
# LANGKAH 9 & 10: Prediksi & tampilkan hasil

@app.route('/api/prediksi', methods=['POST', 'OPTIONS'])
def prediksi():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    # force=True agar tetap parse JSON meski Content-Type tidak sempurna
    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({'success': False, 'error': 'Request body tidak valid. Kirim JSON.'}), 400

    try:
        vehicle_type_val = 1 if data.get('vehicle_type', 'Passenger') == 'Passenger' else 0

        # Urutan HARUS sama persis dengan features_all:
        # ['Engine_size','Horsepower','Wheelbase','Width','Length',
        #  'Curb_weight','Fuel_capacity','Fuel_efficiency','Vehicle_type_encoded']
        fitur_input = np.array([[
            float(data['engine_size']),
            float(data['horsepower']),
            float(data['wheelbase']),
            float(data['width']),
            float(data['length']),
            float(data['curb_weight']),
            float(data['fuel_capacity']),
            float(data['fuel_efficiency']),
            vehicle_type_val
        ]])

        harga_k   = float(model.predict(fitur_input)[0])
        harga_usd = max(harga_k * 1000, 0)
        return jsonify({
            'success'      : True,
            'harga_ribu'   : round(harga_k, 3),
            'harga_usd'    : round(harga_usd, 2),
            'harga_format' : f"${harga_usd:,.0f}",
            'input'        : data
        })
    except KeyError as e:
        return jsonify({'success': False, 'error': f'Field tidak ditemukan: {e}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# API 3: Top 10 penjualan (data tabel)
# ============================================================

@app.route('/api/top10-penjualan', methods=['GET'])
def top10_penjualan():
    top10 = df_clean.nlargest(10, 'Sales_in_thousands')[['Car_Name', 'Sales_in_thousands', 'Price_in_thousands']].reset_index(drop=True)
    return jsonify(top10.to_dict(orient='records'))

# ============================================================
# API 4: Chart top 10 penjualan (PNG base64)
# ============================================================

@app.route('/api/chart/top10-penjualan', methods=['GET'])
def chart_top10():
    top10 = df_clean.nlargest(10, 'Sales_in_thousands')[['Car_Name', 'Sales_in_thousands']].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", 10)
    bars = ax.barh(top10['Car_Name'][::-1], top10['Sales_in_thousands'][::-1], color=colors)
    ax.set_xlabel('Penjualan (ribuan unit)', fontsize=11)
    ax.set_title('TOP 10 Mobil — Penjualan Terbanyak', fontsize=13, fontweight='bold')
    ax.bar_label(bars, fmt='%.1f K', padding=3, fontsize=9)
    ax.set_xlim(0, top10['Sales_in_thousands'].max() * 1.15)
    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})

# ============================================================
# API 5: Chart harga top 10
# ============================================================

@app.route('/api/chart/harga-top10', methods=['GET'])
def chart_harga():
    top10 = df_clean.nlargest(10, 'Sales_in_thousands').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("rocket", 10)
    bars = ax.bar(range(len(top10)), top10['Price_in_thousands'], color=colors, edgecolor='white')
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels(top10['Car_Name'], rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Harga (ribuan USD)', fontsize=11)
    ax.set_title('Harga 10 Mobil Terlaris', fontsize=13, fontweight='bold')
    ax.bar_label(bars, fmt='$%.1fK', padding=3, fontsize=8)
    ax.set_ylim(0, top10['Price_in_thousands'].max() * 1.2)
    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})

# ============================================================
# API 6: Scatter plot evaluasi model
# ============================================================

@app.route('/api/chart/evaluasi', methods=['GET'])
def chart_evaluasi():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Evaluasi Model Linear Regression', fontsize=13, fontweight='bold')
    min_val = min(min(y_test_list), min(y_pred_list))
    max_val = max(max(y_test_list), max(y_pred_list))
    axes[0].scatter(y_test_list, y_pred_list, alpha=0.7, color='steelblue', edgecolors='white', s=70)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('Harga Aktual (K USD)')
    axes[0].set_ylabel('Harga Prediksi (K USD)')
    axes[0].set_title(f'Aktual vs Prediksi  |  R²={r2:.4f}')
    axes[0].legend()
    residuals = [a - p for a, p in zip(y_test_list, y_pred_list)]
    axes[1].scatter(y_pred_list, residuals, alpha=0.7, color='coral', edgecolors='white', s=70)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Harga Prediksi (K USD)')
    axes[1].set_ylabel('Residual')
    axes[1].set_title(f'Plot Residual  |  RMSE={rmse:.4f}')
    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})

# ============================================================
# API 7: Heatmap korelasi
# ============================================================

@app.route('/api/chart/korelasi', methods=['GET'])
def chart_korelasi():
    cols = ['Price_in_thousands', 'Engine_size', 'Horsepower', 'Wheelbase',
            'Width', 'Length', 'Curb_weight', 'Fuel_capacity', 'Fuel_efficiency',
            'Sales_in_thousands']
    corr = df_clean[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                mask=mask, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Heatmap Korelasi Antar Variabel', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return jsonify({'image': fig_to_base64(fig)})

# ============================================================
# API 8: Statistik dataset (untuk dashboard)
# ============================================================

@app.route('/api/statistik', methods=['GET'])
def statistik():
    return jsonify({
        'total_mobil'       : int(df.shape[0]),
        'total_merek'       : int(df['Manufacturer'].nunique()),
        'harga_rata'        : round(float(df_clean['Price_in_thousands'].mean()), 2),
        'harga_min'         : round(float(df_clean['Price_in_thousands'].min()), 2),
        'harga_max'         : round(float(df_clean['Price_in_thousands'].max()), 2),
        'penjualan_total'   : round(float(df_clean['Sales_in_thousands'].sum()), 1),
        'mobil_terlaris'    : df_clean.loc[df_clean['Sales_in_thousands'].idxmax(), 'Car_Name'],
        'mobil_termahal'    : df_clean.loc[df_clean['Price_in_thousands'].idxmax(), 'Car_Name'],
    })

# ============================================================

if __name__ == '__main__':
    print("\n🚀 Server Flask berjalan di http://localhost:5001")
    print("   Buka browser dan akses: http://localhost:5001")
    
import os
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=False, host="0.0.0.0", port=port)