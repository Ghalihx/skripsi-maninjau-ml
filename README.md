# Dashboard Analisis ERA5-Land Danau Maninjau

Proyek **Streamlit** untuk presentasi hasil penelitian skripsi (Semhas), berisi tabel dan visualisasi interaktif dari **CSV eksperimen asli**. Tidak melatih ulang model, tidak mengakses GEE, dan tidak mengeluarkan prediksi operasional atau deteksi *upwelling*.

## Fitur

Ringkasan hasil, komparasi RMSE/MAE/R² dan dua baseline, deret waktu prediksi uji independen 2024–2025, 3-fold ROCV, *feature importance*, audit satelit serta grid, dan analisis deskriptif 15 catatan historis.

## Menjalankan lokal

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Dashboard membuka halaman lokal di `http://localhost:8501`.

## Publikasi di GitHub dan Streamlit Community Cloud

1. Ekstrak ZIP. Buat repository GitHub baru, lalu unggah **isi folder** `STREAMLIT_MANINJAU_GITHUB` ke **root** repository. Jangan mengunggah folder luarnya saja sehingga `app.py` berada terlalu dalam.
2. Pastikan `app.py`, `requirements.txt`, `.streamlit/config.toml`, dan semua file dalam `data/` sudah ikut di-commit.
3. Buka https://share.streamlit.io/ lalu hubungkan repository GitHub tersebut.
4. Pilih `app.py` sebagai *Main file path*, kemudian *Deploy*.

Jika tombol deploy meminta izin/akses repository, kamu harus memberi izin sendiri di GitHub. Tidak ada token API atau informasi rahasia pada proyek ini.

## Identitas data

File `data/` menggunakan snapshot akhir dari unggahan penelitian:

- `evaluasi_model.csv` ← `hasil_evaluasi_test_2024_2025(2).csv`
- `tuning_rocv.csv` ← `hasil_tuning_validasi_2022_2023(2).csv`
- `feature_importance.csv` ← `hasil_feature_importance(2).csv`
- `prediksi_t1.csv` ← `prediksi_uji_independen_t1_2024_2025(1).csv`
- `prediksi_t3.csv` ← `prediksi_uji_independen_t3_2024_2025(1).csv`
- `audit_satelit.csv` ← `evaluasi_khusus_era5_vs_modis_terra.csv`
- `evaluasi_upwelling.csv` ← `hasil_evaluasi_eksploratif_upwelling(3).csv`
- `inventaris_upwelling.csv` ← `inventaris_upwelling_maninjau_2016_2025(3).csv`
- `audit_grid.csv` ← `maninjau_audit_grid_era5.csv`

**PENTING:** tabel evaluasi `evaluasi_model.csv` memiliki 16 baris, termasuk persistence anomali dan suhu mentah. Prediksi t+1 berjumlah 730 baris dan t+3 berjumlah 728 baris. CSV dari eksperimen versi lama **tidak boleh dicampur** ke dalam folder ini.

## Catatan akademik

Target penelitian adalah anomali `lake_mix_layer_temperature` keluaran FLake pada ERA5-Land. Ini **bukan suhu aktual danau berdasarkan pengukuran in-situ**. Audit satelit adalah pembanding observasional dengan karakteristik fisik/waktu berbeda. Catatan historis *upwelling* dianalisis secara deskriptif dan tidak menjadi bukti klasifikasi kejadian.

Untuk reproduksi pelatihan, gunakan notebook Colab dan paket data penelitian yang terpisah. Dashboard ini sengaja **read-only** agar angka hasil Semhas stabil dan mudah diaudit.
