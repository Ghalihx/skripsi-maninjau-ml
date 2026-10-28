# Prediksi Suhu Permukaan Danau Maninjau & Analisis Upwelling

Aplikasi Dashboard Interaktif Streamlit dan Repositori Penelitian Skripsi:
**"Analisis Performa Machine Learning Berbasis Fitur Multi-Sumber untuk Prediksi Suhu Permukaan Danau Maninjau"**

---

## 🌟 Fitur Dashboard (Streamlit App)
Aplikasi web ini menyajikan 6 menu navigasi utama yang interaktif:
1. **Ringkasan Eksekutif & Metodologi**: Informasi latar belakang, kerangka pemikiran, rumusan masalah, dan arsitektur pemodelan.
2. **Audit Grid & Kalibrasi Sensor**: Perbandingan spasial ERA5-Land vs MODIS Terra LST (evaluasi bias dan akurasi).
3. **Hasil Tuning & Optimasi Model**: Parameter terbaik hasil tuning validasi 2022–2023 untuk XGBoost, SVR, Ridge, dan Stacking.
4. **Evaluasi Uji Independen (2024–2025)**: Metrik performa (RMSE, MAE, R²) pada horizon $t+1$ dan $t+3$ hari untuk skema Full Features (73 fitur) vs Meteorology-Only (55 fitur).
5. **Dinamika Upwelling & Validasi Lapangan**: Overlay deret waktu suhu terhadap 15 catatan riwayat kematian massal ikan (tubo belerang) 2016–2025.
6. **Simulasi Prediksi Interaktif**: Modul interaktif untuk simulasi skenario suhu dan deteksi indikasi anomali penurunan suhu danau.

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

### 1. Clone Repositori
```bash
git clone https://github.com/Ghalihx/skripsi-maninjau-ml.git
cd skripsi-maninjau-ml
```

### 2. Buat Environment & Pasang Dependensi
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Jalankan Streamlit
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka pada browser di `http://localhost:8501`.

---

## 📂 Struktur Berkas Repositori
- `app.py` : Kode utama dashboard interaktif Streamlit.
- `requirements.txt` : Daftar pustaka Python yang dibutuhkan.
- `audit_dan_eksperimen_ml.py` : Skrip pipeline training, validasi time-series, dan pengujian model ML.
- `buat_grafik_dan_naskah_lengkap.py` : Skrip visualisasi hasil eksperimen resolusi tinggi (300 DPI).
- `buat_dokumen_word_skripsi.py` : Skrip pembuat dokumen naskah Word (.docx) Sempro dan Semhas.
- `*.csv` : Data hasil audit, tuning, prediksi independen, dan inventaris upwelling yang telah terkunci.
- `Gambar_Skripsi/` : Direktori visualisasi grafik naskah dan dashboard.
- `PROPOSAL_SEMPRO_MANINJAU.docx` & `NASKAH_SEMHAS_MANINJAU.docx` : Naskah terformat sesuai panduan akademik FILKOM Universitas Djuanda.

---

## 🌐 Deployment di Streamlit Community Cloud
1. Kunjungi [share.streamlit.io](https://share.streamlit.io).
2. Masuk menggunakan akun GitHub Anda (`Ghalihx`).
3. Klik **"New app"**.
4. Pilih:
   - **Repository**: `Ghalihx/skripsi-maninjau-ml`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **"Deploy!"**.
