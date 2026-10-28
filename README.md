# Dashboard dan pelatihan ulang ML — Danau Maninjau

Aplikasi Streamlit untuk **melatih ulang dari awal** Random Forest, XGBoost, dan SVR dari CSV serta menampilkan dashboard hasil penelitian.

## Unggah ke GitHub dan deploy
1. Unggah **isi folder** ini ke akar repository GitHub (`app.py` harus di root).
2. Di Streamlit Community Cloud pilih repository ini, main file `app.py`.
3. Buka menu **Latih Model dari CSV**, unggah dataset `maninjau_daily_gee_2016_2025.csv`, lalu jalankan.
4. Alternatif lokal: `pip install -r requirements.txt` kemudian `streamlit run app.py`.

## Input
- Wajib: CSV harian GEE dengan **3653 tanggal lengkap 2016-01-01 s.d. 2025-12-31** dan kolom dataset asli. Ambil contoh pada folder `contoh_data/` atau tombol unduh di aplikasi.
- Opsional: CSV Landsat dan inventaris 15 catatan. Bila tidak diunggah, aplikasi memakai **data pendamping bawaan penelitian**. Untuk dataset periode lain, jangan gunakan hasil ini sebagai klaim validasi.

## Proses (menggunakan skrip penelitian asli)
- `audit_dan_eksperimen_ml.py`: versi eksperimen `(3)` dari penelitian. Dipanggil terpisah melalui `training_runner.py`, sehingga tiap sesi menggunakan direktori sementara sendiri.
- Klimatologi kalender 365 hari hanya dari latih tiap fold.
- 3-fold rolling-origin validation; dua horizon (t+1, t+3); skema 73 dan 55 fitur; RF, XGBoost, SVR; dua persistence baseline.
- Evaluasi independen 2024–2025; grafik/tabel baru dan unduh ZIP hasil.
- *Catatan:* hasil pelatihan satu sesi **tidak menimpa** sembilan CSV referensi yang dipakai menu dashboard lain.

## Keterbatasan operasional
- Ini **reproduksi retrospektif atas dataset GEE yang sudah diekstrak**, **bukan ekstraksi GEE langsung dari Streamlit**.
- Proses pelatihan penuh cukup berat (banyak kandidat model/fold), bisa melampaui batas waktu/memori pada Streamlit Cloud gratis. Untuk eksperimen reproduksibilitas yang andal, jalankan secara lokal atau notebook Google Colab.
- Hasil sesi hanya tersimpan selama sesi Streamlit berjalan; **unduh ZIP** setelah selesai.
- Tidak melakukan deteksi/upwelling atau prediksi suhu danau in-situ; target reanalisis ERA5-Land/FLake.
- Penjelasan metodologi ilmiah lengkap tetap berada dalam naskah BAB III dan BAB IV.

## Struktur
```text
app.py
training_ui.py
training_runner.py
audit_dan_eksperimen_ml.py
requirements.txt
README.md
data/                   # arsip hasil skripsi untuk menu dashboard
contoh_data/            # contoh dataset harian, Landsat, inventaris
```
