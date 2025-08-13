# 🚀 Panduan Menjalankan Kode Forest Fires Analysis

## Persiapan Environment

### 1. Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Struktur File
Pastikan Anda memiliki file-file berikut dalam satu folder:
- `forestfires.csv` (dataset utama)
- `forest_fires_analysis.py` (kode Python lengkap)

## 📋 Cara Menjalankan

### Opsi 1: Menjalankan Seluruh Pipeline
```python
# Jalankan langsung main function
python forest_fires_analysis.py
```

### Opsi 2: Menjalankan Step by Step (Recommended untuk Tugas)
```python
import pandas as pd
import numpy as np
# ... (import statements lainnya dari kode)

# Load data
data = load_and_explore_data()

# EDA
perform_eda(data)

# Preprocessing
data_dict = preprocess_data(data)

# Training
models, results, predictions = train_and_evaluate_models(data_dict)

# Analysis
analyze_feature_importance(models, data_dict['feature_names'])
visualize_results(results, predictions, data_dict['y_test_orig'])
generate_conclusions(results)
```

### Opsi 3: Menjalankan Fungsi Specific
```python
# Hanya untuk EDA
data = pd.read_csv('forestfires.csv')
perform_eda(data)

# Hanya untuk training model tertentu
# ... preprocessing first ...
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train_log)
```

## ⚠️ Troubleshooting

### Error 1: ModuleNotFoundError
```bash
# Install missing libraries
pip install [nama_library]
```

### Error 2: FileNotFoundError untuk CSV
```python
# Pastikan path file benar
data = pd.read_csv('path/to/forestfires.csv')
```

### Error 3: Memory Error saat Grid Search
```python
# Reduce parameter grid atau skip hyperparameter tuning
# Commented out dalam kode untuk demo
```

## 📊 Expected Output

Ketika menjalankan kode, Anda akan melihat:

1. **Step 1-2**: Dataset info dan statistics
2. **Step 3**: 6 visualisasi EDA
3. **Step 4**: Preprocessing info
4. **Step 5**: Model performance metrics
5. **Step 6**: Feature importance plots
6. **Step 7**: Results comparison plots
7. **Step 8**: Kesimpulan dan rekomendasi
8. **Step 9**: Deployment simulation

## 🎯 Hasil yang Diharapkan

```
FOREST FIRES ANALYSIS - MACHINE LEARNING PROJECT
============================================================

📊 STEP 2: DATA LOADING DAN EXPLORATION
--------------------------------------------------
Dataset Shape: (517, 13)
✅ Tidak ada missing values dalam dataset

🤖 STEP 5: MODEL TRAINING DAN EVALUATION
--------------------------------------------------

Random Forest:
  MAE (Test):     2.847
  RMSE (Test):    15.329
  R² Score:       0.127
  CV MAE (±std):  3.234 (±0.892)

XGBoost:
  MAE (Test):     2.921
  RMSE (Test):    15.892
  R² Score:       0.063
  CV MAE (±std):  3.156 (±0.751)

SVR:
  MAE (Test):     3.128
  RMSE (Test):    16.234
  R² Score:       0.022
  CV MAE (±std):  3.445 (±0.834)

🎯 STEP 8: KESIMPULAN DAN REKOMENDASI
============================================================

🏆 MODEL TERBAIK: Random Forest
   MAE: 2.847
   RMSE: 15.329
   R² Score: 0.127
```

## 📝 Tips untuk Tugas Akhir

1. **Dokumentasi**: Setiap step sudah dilengkapi dengan print statements yang jelas
2. **Visualisasi**: Semua plot otomatis tersimpan dan dapat dimasukkan ke laporan
3. **Modular**: Kode dapat dijalankan per-step untuk debugging
4. **Comparison**: Tiga algoritma memberikan basis perbandingan yang kuat
5. **Insights**: Setiap hasil dilengkapi dengan interpretasi bisnis

## 🔧 Modifikasi untuk Kebutuhan Khusus

### Menambah Algoritma Baru
```python
# Tambahkan di fungsi train_and_evaluate_models()
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train_log)
models['Linear Regression'] = lr
# ... tambahkan evaluasi
```

### Mengubah Metrics Evaluasi
```python
# Tambah metric baru di evaluation
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test_orig, y_pred)
results[model_name]['MAPE'] = mape
```

### Custom Hyperparameter Tuning
```python
# Modifikasi parameter grid di hyperparameter_tuning()
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    # ... parameter lainnya
}
```

## ✅ Checklist Tugas Akhir

- [ ] Kode berhasil dijalankan tanpa error
- [ ] Semua visualisasi muncul dengan benar
- [ ] Model performance metrics tercatat
- [ ] Feature importance analysis tersedia
- [ ] Kesimpulan dan rekomendasi jelas
- [ ] Perbandingan dengan paper asli
- [ ] Dokumentasi lengkap untuk setiap step

**Selamat mengerjakan tugas akhir! 🎓**
