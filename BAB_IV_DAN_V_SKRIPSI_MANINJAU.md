# BAB IV — HASIL DAN PEMBAHASAN

Bab ini menyajikan hasil penelitian dan pembahasan berdasarkan tahapan metodologi Skema 1 yang telah dirancang pada Bab III, meliputi: (1) deskripsi objek dan hasil implementasi ekstraksi data *Google Earth Engine* (GEE) beserta audit spasial sel grid *ERA5-Land*, (2) hasil pengujian kesesuaian suhu *ERA5-Land* terhadap pengamatan satelit *MODIS* dan *Landsat 8/9*, seleksi *hyperparameter* *3-Fold Rolling-Origin Cross-Validation* (ROCV), komparasi akurasi model pada data uji independen (2024–2025) untuk Skema Utama (73 fitur) dan Skema Ablasi Meteorologi (55 fitur), *feature importance*, serta evaluasi deskriptif catatan *upwelling*, dan (3) analisis serta pembahasan komprehensif terhadap rumusan masalah dan hipotesis penelitian.

---

## A. Deskripsi Objek dan Hasil Implementasi Ekstraksi Data

### 1. Hasil Implementasi Pipa Komputasi GEE dan Python
Implementasi pengembangan model dan analisis dalam penelitian ini diwujudkan melalui empat skrip komputasi modular yang dijalankan secara berurutan:
1. **Skrip Ekstraksi Multi-Sumber GEE (`gee_extract_maninjau.js`):** Mengekstrak deret waktu harian 22 kolom dari katalog `ECMWF/ERA5_LAND/DAILY_AGGR`, `UCSB-CHG/CHIRPS/DAILY`, `MODIS/061/MOD11A1`, dan `MODIS/061/MYD11A1`, serta citra `LANDSAT/LC08/C02/T1_L2` dan `LANDSAT/LC09/C02/T1_L2` pada zona inti perairan Danau Maninjau.
2. **Skrip Audit Spasial Grid dan Piksel MODIS GEE (`gee_audit_grid_dan_modis.js`):** Mengekstrak geometri poligon sel grid asli *ERA5-Land* beserta fraksi perairan permanen `JRC/GSW1_4/GlobalSurfaceWater` (`occurrence > 90%`) dan statistik piksel harian *MODIS* (`valid_pixels`, `std_c`, `view_time_hr`, dan `view_angle_deg`).
3. **Skrip Evaluasi Sensitivitas Spasial-Satelit (`analisis_audit_lanjutan_gee.py`):** Menjalankan analisis sensitivitas kesesuaian suhu harian *ERA5-Land* terhadap *MODIS Terra/Aqua* pada berbagai kombinasi ambang jumlah piksel valid, homogenitas spasial, dan sudut pandang sensor.
4. **Skrip Pemodelan Regresi Deret Waktu dengan Pengendalian Kebocoran Informasi Masa Depan (`audit_dan_eksperimen_ml.py`):** Mengimplementasikan indeks kalender konsisten 365 hari (`cal_doy_365`), isolasi klimatologi per *fold*, pemisahan kronologis tanggal target, matriks fitur 73 kolom (Skema Utama) dan 55 kolom (Skema Ablasi), seleksi *hyperparameter* *3-Fold Rolling-Origin CV* (2016–2023), evaluasi pada data uji independen (2024–2025), serta evaluasi deskriptif terhadap 15 kandidat catatan *upwelling*.

Potongan kode inti pada fungsi `build_leak_free_dataset()` di dalam `audit_dan_eksperimen_ml.py` yang menerapkan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi (`date <= clim_cutoff`) dan pemisahan kronologis target di perbatasan periode pelatihan disajikan berikut ini:

```python
# Isolasi perhitungan klimatologi 365-hari hanya dari periode pelatihan aktif
ref_df = df[df['date'] <= clim_cutoff]
doy_means = ref_df.groupby('cal_doy_365')['era5_lake_mix_temp_c'].mean()
doy_means = doy_means.reindex(range(1, 366)).interpolate(method='linear')
doy_padded = pd.concat([doy_means.iloc[-7:], doy_means, doy_means.iloc[:7]])
doy_clim_smooth = doy_padded.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]

# Pemotongan tanggal target (target_date <= tr_end) agar tidak bocor ke periode uji
tr_sub = df_f[(df_f['date'] <= tr_end) & (df_f['target_date'] <= tr_end)].copy()
va_sub = df_f[(df_f['date'] >= va_start) & (df_f['date'] <= va_end) & (df_f['target_date'] <= va_end)].copy()
```

### 2. Kelengkapan Dataset Harian GEE (2016–2025)
Ekstraksi deret waktu multi-sumber pada zona inti perairan (*Core Water ROI*, radius $2\text{ km}$ di titik pusat $[100,187^\circ\text{ BT}, -0,315^\circ\text{ LS}]$ dengan masker perairan permanen `JRC > 90%`) menghasilkan 3.653 baris pengamatan harian tanpa tanggal yang hilang dari 1 Januari 2016 hingga 31 Desember 2025. Ringkasan kelengkapan dan statistik deskriptif sumber data disajikan pada Tabel 5.

**Tabel 5. Ringkasan Ketersediaan dan Statistik Suhu Multi-Sumber di GEE (2016–2025)**

| Sumber Data / Variabel Suhu | Waktu Pengamatan / Representasi | Jumlah Hari Valid (dari 3.653 Hari) | Persentase Ketersediaan | Rata-Rata (°C) | Simpangan Baku (°C) | Minimum (°C) | Maksimum (°C) |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **ERA5-Land `lake_mix_layer_temperature`** | Rata-rata harian model (24 jam) | 3.653 | 100,0% | 27,57 | 0,56 | 25,53 | 29,15 |
| **ERA5-Land `skin_temperature`** | Rata-rata harian model (24 jam) | 3.653 | 100,0% | 27,24 | 0,58 | 24,96 | 28,88 |
| **ERA5-Land `temperature_2m` (Suhu Udara)** | Rata-rata harian model (24 jam) | 3.653 | 100,0% | 22,08 | 0,73 | 19,77 | 24,51 |
| **MODIS Terra Siang (`MOD11A1`)** | Sesaat ($10,26 \pm 0,53\text{ WIB}$) | 922 | 25,2% | 25,53 | 3,02 | 10,59 | 37,60 |
| **MODIS Aqua Siang (`MYD11A1`)** | Sesaat ($13,73 \pm 0,46\text{ WIB}$) | 686 | 18,8% | 26,59 | 2,71 | 13,25 | 39,26 |
| **MODIS Aqua Malam (`MYD11A1`)** | Sesaat ($01,79 \pm 0,52\text{ WIB}$) | 270 | 7,4% | 22,11 | 2,84 | 10,98 | 28,41 |
| **MODIS Terra Malam (`MOD11A1`)** | Sesaat ($22,26 \pm 0,52\text{ WIB}$) | 155 | 4,2% | 22,84 | 2,79 | 11,58 | 27,95 |
| **Landsat 8/9 Level-2 (`ST_B10`)** | Sesaat ($\approx 10:00\text{ WIB}$) | 61 | 1,7% | 28,77 | 2,98 | 18,92 | 33,90 |

*Sumber: Hasil Pengolahan Data GEE (2026)*

Sebagaimana terlihat pada Tabel 5, produk reanalisis *ERA5-Land* dan curah hujan *CHIRPS* memiliki ketersediaan harian 100% sepanjang 10 tahun. Sebaliknya, observasi satelit termal mengalami keterbatasan ketersediaan yang berat akibat tutupan awan tropis di kawasan Bukit Barisan, di mana *MODIS Terra* siang hanya tersedia pada 25,2% hari dan *Landsat 8/9* hanya menyediakan 61 citra dalam 10 tahun ($\approx 6$ citra per tahun).

### 3. Hasil Audit Spasial Sel Grid ERA5-Land di Kawasan Kaldera Maninjau
Mengingat *ERA5-Land* memiliki resolusi spasial asli $0,1^\circ \times 0,1^\circ$ ($\approx 11.132\text{ m}$), dilakukan audit spasial terhadap 4 sel grid di sekitar kawasan kaldera Danau Maninjau (`maninjau_audit_grid_era5.csv`). Hasil audit spasial disajikan pada Tabel 6.

**Tabel 6. Karakteristik Spasial dan Parameter Termal Rata-Rata (2016–2025) pada Empat Sel Grid ERA5-Land di Sekitar Danau Maninjau**

| ID Grid (Koordinat Pusat) | Beririsan dengan ROI Inti ($r=2\text{ km}$) | Luas Sel Grid ($\text{km}^2$) | Fraksi Air Permanen JRC (%) | Luas Air JRC ($\text{km}^2$) | Elevasi Rata-Rata SRTM (m) | $T_{\text{mix}}$ Danau (°C) | $T_{\text{bottom}}$ Dasar (°C) | Kedalaman $h_{\text{mix}}$ (m) | $T_{\text{skin}}$ Grid (°C) | $T_{\text{air,2m}}$ Grid (°C) |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **`1001999970` (`100,2°BT, -0,3°LS`)** | **Ya ($12,42\text{ km}^2$)** | 123,64 | **52,20%** | **64,54** | 589,89 | **27,57** | **24,57** | **7,16** | **27,24** | 22,08 |
| `1001999960` (`100,2°BT, -0,4°LS`) | Tidak ($0,00\text{ km}^2$) | 123,64 | 23,46% | 29,00 | 723,97 | 24,94 | 24,34 | 21,08 | 23,59 | 23,05 |
| `1000999970` (`100,1°BT, -0,3°LS`) | Tidak ($0,00\text{ km}^2$) | 123,64 | 0,00% | 0,00 | 507,01 | 24,93 | 24,55 | 18,64 | 23,49 | 23,39 |
| `1000999960` (`100,1°BT, -0,4°LS`) | Tidak ($0,00\text{ km}^2$) | 123,64 | 0,00% | 0,00 | 336,48 | 25,72 | 25,29 | 19,29 | 24,30 | 24,35 |

*Sumber: Hasil Audit Spasial GEE `maninjau_audit_grid_era5.csv` (2026)*

Berdasarkan Tabel 6, seluruh area ROI Inti perairan ($12,42\text{ km}^2$) berada di dalam satu sel grid tunggal (`1001999970`, berpusat di $100,20^\circ\text{ BT}, -0,30^\circ\text{ LS}$). Di dalam sel grid seluas $123,64\text{ km}^2$ ini, sebesar **52,20% ($64,54\text{ km}^2$)** teridentifikasi sebagai perairan permanen berdasarkan produk `JRC Global Surface Water` resolusi $30\text{ m}$, sedangkan **47,80%** sisanya merupakan daratan dan lereng perbukitan kaldera (elevasi rata-rata grid $589,89\text{ m}$ mdpl). Sel grid utama ini menunjukkan stratifikasi vertikal antara suhu lapisan tercampur ($27,57^\circ\text{C}$, kedalaman rata-rata $7,16\text{ m}$) dan suhu lapisan dasar model ($24,57^\circ\text{C}$, selisih $+3,00^\circ\text{C}$), serta memiliki korelasi internal yang sangat kuat terhadap `skin_temperature` grid ($r = +0,976$).

---

## B. Hasil Pengujian Kesesuaian Satelit dan Kinerja Model Regresi

### 1. Hasil Evaluasi Sensitivitas Suhu ERA5-Land terhadap Satelit MODIS dan Landsat
Untuk menguji apakah variasi suhu harian `era5_lake_mix_temp_c` memiliki kesesuaian dengan pengamatan satelit pada tanggal yang sama, dilakukan analisis sensitivitas bertahap berdasarkan atribut kualitas cakupan piksel (maksimum **22 piksel air** *MODIS* di dalam ROI radius $2\text{ km}$), keseragaman spasial antar-piksel (`std_c`), dan sudut pandang sensor (`view_angle_deg`) pada `maninjau_audit_modis_pixels_2016_2025.csv`. Hasil evaluasi disajikan pada Tabel 7 dan divisualisasikan pada Gambar 2.

**Tabel 7. Analisis Sensitivitas Kesesuaian Suhu Harian ERA5-Land (`era5_lake_mix_temp_c`) terhadap MODIS Siang dan Landsat 8/9 (2016–2025)**

| Sensor (Waktu Pengamatan Lokal) | Kriteria Penyaringan Kualitas Cakupan & Geometri Sensor | Jumlah Hari ($N$) | Rata-Rata Satelit (°C) | Rentang Min–Max (°C) | Bias vs ERA5 (°C) | MAE vs ERA5 (°C) | RMSE vs ERA5 (°C) | Korelasi Pearson ($r$) | Korelasi Spearman ($\rho$) |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **MODIS Terra Siang** ($10,26 \pm 0,53\text{ WIB}$) | 1. Semua hari lolos QC ($\ge 1/22$ piksel) | 922 | 25,53 | 10,59 – 37,60 | −2,22 | 2,49 | 3,12 | 0,223 | 0,218 |
| **MODIS Terra Siang** ($10,26 \pm 0,53\text{ WIB}$) | 2. Ambang cakupan $\ge 10/22$ piksel | 735 | 25,86 | 17,53 – 37,60 | −1,89 | 2,15 | 2,66 | 0,233 | 0,216 |
| **MODIS Terra Siang** ($10,26 \pm 0,53\text{ WIB}$) | **3. Ambang cakupan memadai ($\ge 15/22$ piksel)** | **641** | **26,00** | **17,53 – 34,92** | **−1,76** | **2,00** | **2,49** | **0,223** | **0,205** |
| **MODIS Terra Siang** ($10,26 \pm 0,53\text{ WIB}$) | 4. Cakupan penuh ($22/22$ piksel) | 435 | 26,38 | 20,74 – 34,92 | −1,38 | 1,65 | 2,11 | 0,201 | 0,213 |
| **MODIS Terra Siang** ($10,26 \pm 0,53\text{ WIB}$) | 5. $\ge 15$ piksel + Homogen (`std <= 0,5°C`) | 275 | 26,34 | 20,74 – 34,92 | −1,44 | 1,59 | 2,02 | 0,292 | 0,266 |
| **MODIS Terra Siang** ($10,26 \pm 0,53\text{ WIB}$) | **6. $\ge 15$ piksel + Homogen (`std <= 0,5°C`) + *Near-Nadir* (`|angle| <= 30°`)** | **117** | **26,78** | **22,75 – 29,14** | **−0,96** | **1,10** | **1,52** | **0,298** | **0,330** |
| **MODIS Aqua Siang** ($13,73 \pm 0,46\text{ WIB}$) | 1. Semua hari lolos QC ($\ge 1/22$ piksel) | 686 | 26,60 | 13,59 – 39,29 | −1,12 | 2,02 | 2,69 | 0,132 | 0,119 |
| **MODIS Aqua Siang** ($13,73 \pm 0,46\text{ WIB}$) | 2. Ambang cakupan memadai ($\ge 15/22$ piksel) | 441 | 27,39 | 16,46 – 39,29 | −0,35 | 1,56 | 2,13 | 0,127 | 0,112 |
| **MODIS Aqua Siang** ($13,73 \pm 0,46\text{ WIB}$) | 3. $\ge 15$ piksel + Homogen (`std <= 0,5°C`) + *Near-Nadir* (`|angle| <= 30°`) | 85 | 28,30 | 23,70 – 30,83 | +0,53 | 1,16 | 1,43 | 0,206 | 0,214 |
| **Landsat 8/9 (`ST_B10`)** ($\approx 10:00\text{ WIB}$) | 1. Lolos QA awal ($\ge 100$ piksel, `ST_QA < 3K`) | 61 | 28,77 | 18,92 – 33,90 | +1,02 | 3,00 | 3,66 | 0,242 | 0,231 |
| **Landsat 8/9 (`ST_B10`)** ($\approx 10:00\text{ WIB}$) | 2. Filter QA ketat ($\ge 5.000$ piksel, `std < 0,5°C`, awan $< 30\%$) | 16 | 30,68 | 29,11 – 32,87 | +2,88 | 2,88 | 3,05 | 0,364 | 0,350 |

*Sumber: Hasil Pengolahan Data `evaluasi_khusus_era5_vs_modis_terra.csv` (2026)*

![Gambar 2 Evaluasi kesesuaian suhu harian ERA5-Land terhadap MODIS Terra siang](Gambar_Skripsi/Gambar_1_Audit_ERA5_vs_MODIS_Terra.png)

**Gambar 2. Evaluasi kesesuaian suhu harian ERA5-Land terhadap MODIS Terra siang berdasarkan kriteria kualitas cakupan piksel dan geometri sensor (2016–2025)**
*Sumber: Hasil Pengolahan Data (2026)*

### 2. Hasil Seleksi Hyperparameter melalui *3-Fold Rolling-Origin Cross-Validation* (2016–2023)
Pemilihan *hyperparameter* dilakukan menggunakan rata-rata RMSE dari 3 *fold* berurutan pada periode pra-uji (`Fold 1: 2020–2021`, `Fold 2: 2021–2022`, `Fold 3: 2022–2023`), di mana klimatologi setiap *fold* dihitung secara terisolasi hanya dari data latihnya. Proses *tuning* dijalankan secara terpisah untuk **Rezim 1 (Skema Utama: 73 fitur termal historis + meteorologi $\le t$)** dan **Rezim 2 (Skema Ablasi: 55 fitur murni meteorologi $\le t$)**. Ringkasan konfigurasi terbaik hasil ROCV (`hasil_tuning_validasi_2022_2023.csv`) disajikan pada Tabel 8.

**Tabel 8. Konfigurasi Hyperparameter Terbaik Berdasarkan 3-Fold Rolling-Origin CV (2016–2023)**

| Horizon | Rezim Fitur Input ($\le t$) | Algoritma | Konfigurasi Terpilih (Minimum `ROCV_Mean_RMSE`) | Fold 1 RMSE (°C) | Fold 2 RMSE (°C) | Fold 3 RMSE (°C) | Rata-Rata ROCV RMSE $\pm$ Std (°C) | Rata-Rata ROCV $R^2$ Anomali |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **$t+1$** | **1. Utama** (73 Fitur) | *Random Forest* | `n_estimators=300, max_depth=12, min_samples_leaf=3` | 0,1196 | 0,1091 | 0,1092 | $0,1126 \pm 0,0049$ | 0,9224 |
| **$t+1$** | **1. Utama** (73 Fitur) | **XGBoost** | `n_estimators=400, lr=0.02, max_depth=3, subsample=0.8` | 0,1094 | 0,1022 | 0,1092 | **$0,1069 \pm 0,0034$** | **0,9314** |
| **$t+1$** | **1. Utama** (73 Fitur) | *SVR (RBF)* | `kernel='rbf', C=1.0, gamma=0.002, epsilon=0.01` | 0,1062 | 0,1097 | 0,1081 | $0,1080 \pm 0,0014$ | 0,9289 |
| **$t+1$** | **2. Ablasi** (55 Fitur) | *Random Forest* | `n_estimators=300, max_depth=12, min_samples_leaf=3` | 0,2466 | 0,2875 | 0,3424 | $0,2922 \pm 0,0392$ | 0,5017 |
| **$t+1$** | **2. Ablasi** (55 Fitur) | **XGBoost** | `n_estimators=300, lr=0.05, max_depth=4, subsample=0.8` | 0,2163 | 0,2571 | 0,2959 | **$0,2564 \pm 0,0325$** | **0,6133** |
| **$t+1$** | **2. Ablasi** (55 Fitur) | *SVR (RBF)* | `kernel='rbf', C=1.0, gamma='scale', epsilon=0.02` | 0,2384 | 0,2631 | 0,3041 | $0,2685 \pm 0,0271$ | 0,5770 |
| **$t+3$** | **1. Utama** (73 Fitur) | *Random Forest* | `n_estimators=500, max_depth=10, min_samples_leaf=10` | 0,2031 | 0,1930 | 0,1982 | $0,1981 \pm 0,0041$ | 0,7623 |
| **$t+3$** | **1. Utama** (73 Fitur) | *XGBoost* | `n_estimators=200, lr=0.03, max_depth=3, subsample=0.8` | 0,1927 | 0,1917 | 0,2028 | $0,1957 \pm 0,0050$ | 0,7703 |
| **$t+3$** | **1. Utama** (73 Fitur) | **SVR (RBF)** | `kernel='rbf', C=0.5, gamma=0.002, epsilon=0.01` | 0,1943 | 0,1939 | 0,1877 | **$0,1920 \pm 0,0030$** | **0,7763** |
| **$t+3$** | **2. Ablasi** (55 Fitur) | *Random Forest* | `n_estimators=300, max_depth=12, min_samples_leaf=3` | 0,2757 | 0,3135 | 0,3683 | $0,3192 \pm 0,0380$ | 0,4045 |
| **$t+3$** | **2. Ablasi** (55 Fitur) | **XGBoost** | `n_estimators=300, lr=0.05, max_depth=4, subsample=0.8` | 0,2562 | 0,2878 | 0,3285 | **$0,2908 \pm 0,0296$** | **0,5045** |
| **$t+3$** | **2. Ablasi** (55 Fitur) | *SVR (RBF)* | `kernel='rbf', C=0.5, gamma=0.005, epsilon=0.01` | 0,2753 | 0,2955 | 0,3175 | $0,2961 \pm 0,0172$ | 0,4842 |

*Sumber: Hasil Pengolahan Data `hasil_tuning_validasi_2022_2023.csv` (2026)*

### 3. Hasil Pengujian pada Data Uji Independen (2024–2025): Skema Utama (73 Fitur) dan Skema Ablasi (55 Fitur)
Setelah konfigurasi terbaik dipilih dari ROCV, masing-masing model dilatih satu kali pada seluruh periode pra-uji (`2016–2023`, dengan pemotongan tanggal target $t+h \le \text{31 Desember 2023}$ dan klimatologi 365 hari dari 2016–2023) lalu dievaluasi pada **Data Uji Independen (`2024–2025`, $N=730$ hari untuk $t+1$ dan $N=728$ hari untuk $t+3$)**. Hasil pengujian disajikan pada Tabel 9, sedangkan perbandingan grafik RMSE dan deret waktu prediksinya disajikan pada Gambar 3 dan Gambar 4.

**Tabel 9. Perbandingan Kinerja Model pada Data Uji Independen (2024–2025) terhadap Target Reanalisis ERA5-Land**

| Horizon | Rezim Fitur Input ($\le t$) | Model Prediksi | MAE (°C) | RMSE (°C) *(Metrik Utama)* | $R^2$ Anomali ($A_{t+h}$) | $R^2$ Suhu ERA5 ($T_{t+h}$) | Penurunan RMSE vs *Persistence* Anomali (%) |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **$t+1$** | **1. Utama** (73 Fitur) | *Baseline Persistence* (Suhu Mentah: $\hat{T}_{t+1} = T_t$) | 0,0963 | 0,1309 | 0,8910 | 0,9353 | −0,20% |
| **$t+1$** | **1. Utama** (73 Fitur) | *Baseline Persistence* (Anomali: $\hat{A}_{t+1} = A_t$) | 0,0963 | 0,1306 | 0,8915 | 0,9355 | 0,00% (Acuan) |
| **$t+1$** | **1. Utama** (73 Fitur) | **SVR (RBF)** | 0,0834 | 0,1215 | 0,9061 | 0,9442 | +6,98% |
| **$t+1$** | **1. Utama** (73 Fitur) | **Random Forest** | 0,0860 | 0,1157 | 0,9149 | 0,9494 | +11,44% |
| **$t+1$** | **1. Utama** (73 Fitur) | **XGBoost** *(Model Terbaik $t+1$)* | **0,0794** | **0,1063** | **0,9281** | **0,9573** | **+18,61%** |
| **$t+1$** | **2. Ablasi** (55 Fitur) | **Random Forest (Meteo)** | 0,2082 | 0,2760 | 0,5156 | 0,7123 | — |
| **$t+1$** | **2. Ablasi** (55 Fitur) | **SVR (RBF Meteo)** | 0,1943 | 0,2578 | 0,5773 | 0,7489 | — |
| **$t+1$** | **2. Ablasi** (55 Fitur) | **XGBoost (Meteo)** *(Model Terbaik Ablasi $t+1$)* | **0,1816** | **0,2331** | **0,6545** | **0,7948** | — |
| **$t+3$** | **1. Utama** (73 Fitur) | *Baseline Persistence* (Suhu Mentah: $\hat{T}_{t+3} = T_t$) | 0,1680 | 0,2177 | 0,6981 | 0,8213 | −1,00% |
| **$t+3$** | **1. Utama** (73 Fitur) | *Baseline Persistence* (Anomali: $\hat{A}_{t+3} = A_t$) | 0,1654 | 0,2156 | 0,7040 | 0,8249 | 0,00% (Acuan) |
| **$t+3$** | **1. Utama** (73 Fitur) | **Random Forest** | 0,1535 | 0,1986 | 0,7489 | 0,8514 | +7,89% |
| **$t+3$** | **1. Utama** (73 Fitur) | **SVR (RBF)** *(MAE Terendah $t+3$)* | **0,1444** | 0,1980 | 0,7503 | 0,8523 | +8,16% |
| **$t+3$** | **1. Utama** (73 Fitur) | **XGBoost** *(Model Terbaik $t+3$ berdasar RMSE & $R^2$)* | 0,1501 | **0,1952** | **0,7574** | **0,8564** | **+9,46%** |
| **$t+3$** | **2. Ablasi** (55 Fitur) | **Random Forest (Meteo)** | 0,2403 | 0,3152 | 0,3674 | 0,6257 | — |
| **$t+3$** | **2. Ablasi** (55 Fitur) | **XGBoost (Meteo)** | 0,2087 | 0,2710 | 0,5324 | 0,7233 | — |
| **$t+3$** | **2. Ablasi** (55 Fitur) | **SVR (RBF Meteo)** *(Model Terbaik Ablasi $t+3$)* | **0,2045** | **0,2624** | **0,5615** | **0,7405** | — |

*Sumber: Hasil Pengolahan Data `hasil_evaluasi_test_2024_2025.csv` (2026)*

![Gambar 3 Perbandingan RMSE model pada data uji independen 2024–2025](Gambar_Skripsi/Gambar_2_Komparasi_RMSE_Uji_2024_2025.png)

**Gambar 3. Perbandingan RMSE model pada data uji independen (2024–2025) terhadap target reanalisis ERA5-Land untuk Skema Utama (73 fitur) dan Skema Ablasi (55 fitur)**
*Sumber: Hasil Pengolahan Data (2026)*

![Gambar 4 Deret waktu anomali suhu referensi ERA5-Land terhadap prediksi model pada data uji independen](Gambar_Skripsi/Gambar_3_Deret_Waktu_Uji_Independen_2024_2025.png)

**Gambar 4. Deret waktu anomali suhu referensi ERA5-Land terhadap prediksi model pada data uji independen (2024–2025) untuk horizon t+1 dan t+3**
*Sumber: Hasil Pengolahan Data (2026)*

### 4. Hasil Analisis *Feature Importance*
Hasil ekstraksi kontribusi fitur relatif (*feature importance*) pada *Random Forest* dan *XGBoost* (`hasil_feature_importance.csv`) disajikan secara eksak sesuai nama kolom program pada Tabel 10 dan divisualisasikan pada Gambar 5.

**Tabel 10. Lima Fitur dengan Kontribusi Relatif Tertinggi pada Skema Utama (73 Fitur) dan Skema Ablasi Meteorologi (55 Fitur)**

| Rezim Fitur Input | Model | Peringkat 1 | Peringkat 2 | Peringkat 3 | Peringkat 4 | Peringkat 5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Utama ($t+1$, 73 Fitur)** | *Random Forest* | `anom_doy_c` (0,9332) | `precip_era5_mm` (0,0117) | `thermal_net_rad_mjm2` (0,0046) | `u_wind_10m_ms` (0,0037) | `precip_chirps_mm` (0,0034) |
| **1. Utama ($t+1$, 73 Fitur)** | *XGBoost* | `anom_doy_c` (0,3313) | `anom_doy_lag1` (0,3212) | `anom_doy_lag2` (0,0714) | `anom_doy_lag3` (0,0310) | `thermal_net_rad_mjm2` (0,0198) |
| **1. Utama ($t+3$, 73 Fitur)** | *Random Forest* | `anom_doy_c` (0,8773) | `precip_era5_mm` (0,0121) | `anom_doy_lag1` (0,0076) | `anom_doy_lag2` (0,0075) | `thermal_net_rad_mjm2` (0,0062) |
| **1. Utama ($t+3$, 73 Fitur)** | *XGBoost* | `anom_doy_c` (0,2937) | `anom_doy_lag1` (0,1662) | `anom_doy_lag2` (0,0832) | `air_temp_mean_c` (0,0279) | `anom_doy_lag3` (0,0277) |
| **2. Ablasi Meteo ($t+1$, 55 Fitur)** | *Random Forest* | `air_temp_mean_c_mean14d` (0,2661) | `air_temp_mean_c_mean7d` (0,1238) | `sin_doy` (0,1207) | `cos_doy` (0,0923) | `solar_net_rad_mjm2_mean14d` (0,0736) |
| **2. Ablasi Meteo ($t+1$, 55 Fitur)** | *XGBoost* | `air_temp_mean_c_mean14d` (0,1489) | `air_temp_mean_c_mean7d` (0,1457) | `solar_net_rad_mjm2_mean14d` (0,0573) | `air_temp_mean_c_mean3d` (0,0506) | `sin_doy` (0,0427) |
| **2. Ablasi Meteo ($t+3$, 55 Fitur)** | *Random Forest* | `air_temp_mean_c_mean14d` (0,1998) | `air_temp_mean_c_mean7d` (0,1681) | `sin_doy` (0,1090) | `cos_doy` (0,1008) | `thermal_net_rad_mjm2_mean14d` (0,0772) |
| **2. Ablasi Meteo ($t+3$, 55 Fitur)** | *XGBoost* | `air_temp_mean_c_mean7d` (0,1668) | `air_temp_mean_c_mean14d` (0,1318) | `thermal_net_rad_mjm2_mean14d` (0,0627) | `cos_doy` (0,0480) | `sin_doy` (0,0418) |

*Sumber: Hasil Pengolahan Data `hasil_feature_importance.csv` (2026)*

![Gambar 5 Perbandingan kontribusi fitur relatif XGBoost pada Skema Utama dan Skema Ablasi](Gambar_Skripsi/Gambar_4_Feature_Importance_XGBoost.png)

**Gambar 5. Perbandingan kontribusi fitur relatif (feature importance) XGBoost pada Skema Utama (73 fitur) dan Skema Ablasi Murni Meteorologi (55 fitur) untuk horizon t+1**
*Sumber: Hasil Pengolahan Data (2026)*

### 5. Hasil Evaluasi Deskriptif-Eksploratif pada Kandidat Catatan Upwelling (2016–2025)
Sesuai batasan penelitian, analisis terhadap 15 kandidat catatan historis pada `inventaris_upwelling_maninjau_2016_2025.csv` disajikan secara deskriptif dan dipisahkan ke dalam tiga kelompok menurut kategori presisi tanggal dan durasi jendelanya (`hasil_evaluasi_eksploratif_upwelling.csv`). Kelompok 1 (`Exact_Day`, $N=10$ catatan dengan jendela seragam 5 hari $[t_0-3, t_0+1]$) disajikan pada Tabel 11 dan Gambar 6, sedangkan Kelompok 2 (`Multi_Day`, $N=3$) dan Kelompok 3 (`Month_Only`, $N=2$) disajikan pada Tabel 12.

**Tabel 11. Perilaku Deskriptif Anomali Termal Reanalisis ERA5-Land dan Prediksi XGBoost pada Catatan Harian Spesifik (`Exact_Day`, Jendela $H-3$ s/d $H+1$)**

| ID Catatan (Bulan/Tahun) | Rezim Evaluasi (Klimatologi Acuan) | Kerugian Ikan (Ton) | Anomali ERA5 Hari-$t_0$ (°C) | Mean Anomali Jendela 5-Hari (°C) | Min Anomali Jendela 5-Hari (°C) | Perubahan 3-Hari $\Delta T_{3\text{d}}$ (°C) | Prediksi Mean XGBoost $t+1$ (°C) | Prediksi Min XGBoost $t+1$ (°C) | Curah Hujan Maks (mm) |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| **REC-01** (Agu 2016) | 1. Latih Historis (Clim 2016–2021) | 1.500,0 | −0,19 | −0,24 | −0,30 | −0,05 | — *(In-Sample)* | — *(In-Sample)* | 30,4 |
| **REC-03** (Nov 2017) | 1. Latih Historis (Clim 2016–2021) | 50,0 | −0,69 | −0,59 | −0,69 | −0,37 | — *(In-Sample)* | — *(In-Sample)* | 43,5 |
| **REC-05** (Jan 2020) | 1. Latih Historis (Clim 2016–2021) | 10,0 | +0,35 | +0,62 | +0,35 | −0,39 | — *(In-Sample)* | — *(In-Sample)* | 35,9 |
| **REC-06** (Feb 2020 A) | 1. Latih Historis (Clim 2016–2021) | 15,0 | +0,66 | +0,61 | +0,59 | +0,08 | — *(In-Sample)* | — *(In-Sample)* | 18,7 |
| **REC-07** (Feb 2020 B) | 1. Latih Historis (Clim 2016–2021) | 10,0 | +0,37 | +0,34 | +0,27 | +0,01 | — *(In-Sample)* | — *(In-Sample)* | 2,8 |
| **REC-08** (Apr 2021) | 1. Latih Historis (Clim 2016–2021) | 10,0 | −0,27 | −0,22 | −0,27 | −0,08 | — *(In-Sample)* | — *(In-Sample)* | 15,5 |
| **REC-10** (Feb 2022) | 2. Validasi (Clim 2016–2021) | 130,0 | −0,59 | −0,45 | −0,70 | −0,48 | −0,30 | −0,54 | 7,0 |
| **REC-12** (Mei 2023) | 2. Validasi (Clim 2016–2021) | 15,0 | +0,19 | +0,08 | −0,20 | +0,26 | +0,01 | −0,22 | 7,4 |
| **REC-13** (Jan 2025) | **3. Uji Independen (Clim 2016–2023)** | +50,0 *(Total 75,0)* | **+0,29** | **+0,48** | **+0,29** | **−0,41** | **+0,55** | **+0,34** | 33,0 |
| **REC-14** (Nov 2025) | **3. Uji Independen (Clim 2016–2023)** | 8,0 *(Awal)* | **−0,91** | **−0,73** | **−1,11** | **−0,77** | **−0,57** | **−0,95** | 120,0 |

*Sumber: Hasil Pengolahan Data `hasil_evaluasi_eksploratif_upwelling.csv` (2026)*

![Gambar 6 Distribusi deskriptif anomali suhu reanalisis ERA5-Land pada 10 catatan harian spesifik](Gambar_Skripsi/Gambar_5_Evaluasi_Deskriptif_Upwelling_ExactDay.png)

**Gambar 6. Distribusi deskriptif anomali suhu reanalisis ERA5-Land pada 10 catatan harian spesifik (Exact_Day) periode 2016–2025**
*Sumber: Hasil Pengolahan Data (2026)*

**Tabel 12. Perilaku Deskriptif Anomali Termal pada Episode Multi-Hari ($N=3$) dan Catatan Bulanan ($N=2$)**

| ID Catatan (Periode) | Kategori Presisi (Durasi Jendela) | Rezim Evaluasi (Klimatologi Acuan) | Kerugian Ikan (Ton) | Mean Anomali Periode (°C) | Min Anomali Periode (°C) | Perubahan 3-Hari Maks $\Delta T_{3\text{d}}$ (°C) | Prediksi Mean XGBoost $t+1$ (°C) | Prediksi Min XGBoost $t+1$ (°C) |
| :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **REC-09** (1–24 Des 2021) | `Multi_Day_Episode` (24 hari) | 1. Latih Historis (Clim 2016–2021) | 1.455,00 | +0,25 | −0,45 | −0,98 | — *(In-Sample)* | — *(In-Sample)* |
| **REC-11** (26–29 Nov 2022) | `Multi_Day_Observation` BRIN (4 hari) | 2. Validasi (Clim 2016–2021) | 0,00 *(Riset)* | −0,70 | −0,86 | +0,10 | −0,69 | −0,87 |
| **REC-15** (1–20 Des 2025) | `Multi_Day_Episode` (20 hari) | **3. Uji Independen (Clim 2016–2023)** | 1.428,73 | **−0,50** | **−0,94** | **−0,71** | **−0,51** | **−0,90** |
| **REC-02** (Maret 2017) | `Month_Only` (31 hari — Bulanan) | 1. Latih Historis (Clim 2016–2021) | 77,00 | −0,60 | −1,24 | −0,24 | — | — |
| **REC-04** (Februari 2018) | `Month_Only` (28 hari — Bulanan) | 1. Latih Historis (Clim 2016–2021) | 160,00 | −0,28 | −0,42 | −0,09 | — | — |

*Sumber: Hasil Pengolahan Data `hasil_evaluasi_eksploratif_upwelling.csv` (2026)*

---

## C. Analisis dan Pembahasan

### 1. Pembahasan Keterbatasan Fisik Grid ERA5-Land dan Validasi Satelit (Jawaban Rumusan Masalah 1 & Hipotesis $H_1$)
Berdasarkan hasil audit spasial pada Tabel 6 dan analisis sensitivitas pada Tabel 7, terdapat tiga temuan utama yang menjawab Rumusan Masalah 1 sekaligus menguji Hipotesis $H_1$:
1. **Representasi Campuran Daratan–Perairan pada Grid ERA5-Land:** Meskipun sel grid utama *ERA5-Land* (`1001999970`) mencakup seluruh zona inti perairan Danau Maninjau dengan proporsi air permanen JRC sebesar **52,20% ($64,54\text{ km}^2$)** dan menunjukkan dinamika termal internal yang konsisten terhadap radiasi matahari ($r = +0,782$), suhu udara ($r = +0,518$), dan curah hujan ($r = -0,375$), sekitar **47,80% luas sel grid tersebut merupakan daratan perbukitan kaldera**. Di samping itu, parameter kedalaman lapisan tercampur rata-rata pada modul 1-dimensi *FLake* ($7,16\text{ m}$) dan suhu dasar model ($24,57^\circ\text{C}$) merupakan penyederhanaan numerik yang belum terkalibrasi terhadap batimetri aktual Danau Maninjau ($105-165\text{ m}$).
2. **Sensitivitas Kualitas Piksel MODIS dan Perbedaan Waktu Lintas Lokal:** Pada *MODIS Terra* siang ($\approx 10:15\text{ WIB}$), penerapan syarat jumlah piksel valid dari $\ge 1/22$ piksel ($N=922$) ke $\ge 15/22$ piksel ($N=641$) dan $22/22$ piksel ($N=435$) secara konsisten menurunkan MAE dari $2,49^\circ\text{C}$ menjadi $2,00^\circ\text{C}$ dan $1,65^\circ\text{C}$. Ketika dikombinasikan dengan syarat keseragaman spasial antar-piksel (`std_c <= 0,5°C`) dan sudut pandang dekat nadir (`|view_angle_deg| <= 30°`), penyaringan ini mengurangi kemungkinan kontaminasi piksel campuran maupun gangguan atmosfer/awan tipis sub-piksel sehingga nilai suhu ekstrem berkurang (rentang menyempit menjadi $22,75^\circ\text{C}-29,14^\circ\text{C}$) dan MAE turun menjadi **$1,10^\circ\text{C}$** ($r = 0,298$, $\rho = 0,330$). Selain itu, pada filter ketat terlihat perbedaan arah bias yang selaras dengan siklus pemanasan diurnal: *MODIS Terra* pagi menjelang siang ($10,26\text{ WIB}$) rata-rata lebih dingin **$-0,96^\circ\text{C}$** daripada *ERA5-Land*, sedangkan *MODIS Aqua* siang ($13,73\text{ WIB}$) rata-rata lebih hangat **$+0,53^\circ\text{C}$** daripada *ERA5-Land*.
3. **Evaluasi Hipotesis Pertama ($H_1$):** Temuan empiris ini **mendukung Hipotesis Pertama ($H_1$)**, di mana penyaringan kualitas cakupan piksel dan geometri pengamatan berasosiasi dengan penurunan kesalahan absolut terhadap *ERA5-Land* (dari $2,49^\circ\text{C}$ menjadi $1,10^\circ\text{C}$), namun hanya menyisakan **117 hari dari 3.653 hari (3,2% periode studi)** dengan korelasi yang masih lemah–moderat ($r = 0,298$). Oleh sebab itu, secara ilmiah `era5_lake_mix_temp_c` ditetapkan sebagai target pemodelan reanalisis, dan capaian akurasi model regresi pada penelitian ini tidak diklaim sebagai akurasi terhadap suhu air lapangan sebenarnya.

### 2. Pembahasan Komparasi Kinerja Random Forest, XGBoost, dan SVR serta Kriteria Metrik Penentu Model Terbaik (Jawaban Rumusan Masalah 2 & Hipotesis $H_2$)
Berdasarkan hasil pengujian independen (2024–2025) pada Tabel 9, ketiga algoritma *machine learning* (*Random Forest*, *XGBoost*, dan SVR) yang dilatih pada Skema Utama (73 fitur) berhasil menghasilkan nilai MAE dan RMSE yang lebih rendah serta $R^2$ yang lebih tinggi dibandingkan kedua model acuan *Baseline Persistence* ($\hat{A}_{t+h}=A_t$ dan $\hat{T}_{t+h}=T_t$) baik pada horizon $t+1$ maupun $t+3$. Dengan demikian, **hasil pengujian mendukung Hipotesis Kedua ($H_2$)**.

Dalam menentukan peringkat algoritma terbaik, penelitian ini menetapkan **RMSE dan $R^2$ sebagai kriteria evaluasi utama**, sedangkan **MAE digunakan sebagai metrik evaluasi pendamping**:
1. **Pada Horizon $t+1$ (Skema Utama — 73 Fitur):** Algoritma **XGBoost** unggul secara mutlak pada seluruh metrik evaluasi, dengan nilai **$\text{RMSE} = 0,1063^\circ\text{C}$**, **$\text{MAE} = 0,0794^\circ\text{C}$**, **$R^2\text{ Anomali} = 0,9281$**, dan **$R^2\text{ Suhu} = 0,9573$** (menurunkan RMSE sebesar **18,61%** dibandingkan *Baseline Persistence* Anomali), diikuti oleh *Random Forest* ($\text{RMSE} = 0,1157^\circ\text{C}$, $R^2\text{ Anomali} = 0,9149$) dan SVR ($\text{RMSE} = 0,1215^\circ\text{C}$, $R^2\text{ Anomali} = 0,9061$).
2. **Pada Horizon $t+3$ (Skema Utama — 73 Fitur):** Berdasarkan kriteria utama RMSE dan $R^2$, **XGBoost** kembali menjadi model terbaik dengan **$\text{RMSE} = 0,1952^\circ\text{C}$**, **$R^2\text{ Anomali} = 0,7574$**, dan **$R^2\text{ Suhu} = 0,8564$** (menurunkan RMSE sebesar **9,46%** dibandingkan *Baseline Persistence* Anomali). Di posisi kedua berdasarkan RMSE adalah **SVR (RBF)** dengan $\text{RMSE} = 0,1980^\circ\text{C}$ ($R^2\text{ Anomali} = 0,7503$) dan di posisi ketiga adalah *Random Forest* dengan $\text{RMSE} = 0,1986^\circ\text{C}$ ($R^2\text{ Anomali} = 0,7489$). Namun, apabila ditinjau dari metrik rata-rata kesalahan absolut (**MAE**), **SVR (RBF)** mencatatkan nilai MAE paling rendah pada horizon $t+3$ sebesar **$0,1444^\circ\text{C}$**, lebih rendah dibandingkan *XGBoost* ($0,1501^\circ\text{C}$) dan *Random Forest* ($0,1535^\circ\text{C}$). Perbedaan peringkat antara RMSE dan MAE pada horizon $t+3$ ini diduga berkaitan dengan perbedaan karakteristik fungsi kerugian kedua algoritma; fungsi kerugian *epsilon-insensitive* ($\varepsilon = 0,01$) pada SVR dapat berkontribusi menghasilkan rata-rata kesalahan absolut harian (MAE) yang sedikit lebih kecil pada fluktuasi normal, sedangkan optimisasi fungsi kerugian kuadratik pada *XGBoost* cenderung memberi penalti lebih besar terhadap simpangan ekstrem sehingga dapat berkontribusi menghasilkan nilai RMSE dan $R^2$ yang sedikit lebih baik.

### 3. Pembahasan Ketergantungan Autoregresif vs Kemampuan Murni Meteorologi (Pengujian Hipotesis $H_3$)
Perbandingan antara Skema Utama (73 fitur) dan Skema Ablasi Murni Meteorologi (55 fitur) pada Tabel 9 dan Tabel 10 memberikan bukti kuantitatif yang menjawab Hipotesis $H_3$:
1. **Kontribusi Dominan Fitur Autoregresif pada Skema Utama (73 Fitur):** Sebagaimana ditunjukkan pada Tabel 10, pada Skema Utama algoritma *Random Forest* menumpukan **93,32% ($t+1$)** dan **87,73% ($t+3$)** bobot kepentingannya pada satu fitur tunggal, yaitu `anom_doy_c`. Sebaliknya, **XGBoost** mendistribusikan bobot pemisahan pohonnya secara lebih proporsional di antara `anom_doy_c` ($0,3313$ pada $t+1$; $0,2937$ pada $t+3$), `anom_doy_lag1` ($0,3212$ pada $t+1$; $0,1662$ pada $t+3$), `anom_doy_lag2` ($0,0714$ pada $t+1$; $0,0832$ pada $t+3$), serta mengintegrasikan koreksi dari `thermal_net_rad_mjm2` ($0,0198$) dan `air_temp_mean_c` ($0,0279$). Mekanisme *gradient boosting* memungkinkan pohon-pohon sekuensial mempelajari laju perubahan multi-hari dan pemaksa cuaca sebagai koreksi atas persistensi suhu hari sebelumnya.
2. **Dampak Penghapusan Fitur Termal pada Skema Ablasi (55 Fitur) dan Evaluasi Hipotesis $H_3$:** Ketika ke-18 fitur riwayat suhu dan anomali danau dihapus sehingga model hanya menggunakan 55 fitur kalender dan meteorologi, nilai $R^2\text{ Anomali}$ turun dari kisaran $0,9061-0,9281$ menjadi **$0,5156-0,6545$** pada horizon $t+1$, dan dari kisaran $0,7489-0,7574$ menjadi **$0,3674-0,5615$** pada horizon $t+3$. Penurunan substansial ini **mendukung Hipotesis Ketiga ($H_3$)** dan menunjukkan dominannya informasi riwayat suhu dalam kemampuan prediksi harian. Menariknya, pada kondisi tanpa riwayat suhu danau ini, **XGBoost** unggul pada horizon $t+1$ ($\text{RMSE} = 0,2331^\circ\text{C}$, $R^2\text{ Anomali} = 0,6545$), sedangkan **SVR (RBF)** berbalik mengungguli *XGBoost* dan *Random Forest* pada horizon $t+3$ ($\text{RMSE} = 0,2624^\circ\text{C}$, $\text{MAE} = 0,2045^\circ\text{C}$, $R^2\text{ Anomali} = 0,5615$). Tabel 10 memperlihatkan bahwa tanpa fitur suhu danau, model bertumpu pada agregasi cuaca bergulir 7 hingga 14 hari, yaitu `air_temp_mean_c_mean14d`, `air_temp_mean_c_mean7d`, `solar_net_rad_mjm2_mean14d`, dan `thermal_net_rad_mjm2_mean14d`, yang sejalan dengan respons termal perairan terhadap pemaksa atmosfer mingguan (Li dkk., 2024).

### 4. Pembahasan Kritis Anomali Termal Reanalisis terhadap Kejadian Upwelling
Berdasarkan Tabel 11 dan Tabel 12, evaluasi deskriptif terhadap 15 kandidat catatan kejadian *upwelling* menunjukkan dua sisi temuan yang harus dilaporkan secara objektif:
1. **Kemampuan Melacak Anomali Dingin pada Episode Skala Besar:** Pada periode validasi dan uji independen, model *XGBoost* mampu melacak penurunan anomali suhu *ERA5-Land* secara akurat. Pada periode survei lapangan BRIN (`REC-11`, 26–29 November 2022, saat Putri dkk. [2024] mencatat pelemahan stratifikasi termal dan kenaikan sulfida ke lapisan permukaan), anomali rata-rata *ERA5-Land* bernilai $-0,70^\circ\text{C}$ (minimum $-0,86^\circ\text{C}$) dan diprediksi oleh *XGBoost* sebesar $-0,69^\circ\text{C}$ (minimum $-0,87^\circ\text{C}$). Demikian pula pada episode kematian ikan masif akhir November hingga Desember 2025 (`REC-14` dan `REC-15`, total kematian $1.436,73\text{ ton}$ pasca curah hujan ekstrem $120,0\text{ mm/hari}$), anomali *ERA5-Land* jatuh ke $-0,91^\circ\text{C}$ (minimum jendela $-1,11^\circ\text{C}$) pada `REC-14` dan rata-rata $-0,50^\circ\text{C}$ (minimum $-0,94^\circ\text{C}$) pada `REC-15`, yang berhasil diprediksi oleh *XGBoost* pada data uji independen sebesar $-0,57^\circ\text{C}$ (minimum $-0,95^\circ\text{C}$) dan $-0,51^\circ\text{C}$ (minimum $-0,90^\circ\text{C}$).
2. **Ketidakkonsistenan pada Kejadian Skala Lokal:** Sebaliknya, Tabel 11 juga membuktikan bahwa **tidak semua catatan kematian ikan berasosiasi dengan anomali suhu negatif pada produk *ERA5-Land***. Dari 10 catatan harian spesifik (`Exact_Day`), terdapat 5 catatan yang justru memiliki anomali positif pada hari kejadian maupun rata-rata jendela 5 hari, yaitu `REC-05` ($+0,35^\circ\text{C}$), `REC-06` ($+0,66^\circ\text{C}$), `REC-07` ($+0,37^\circ\text{C}$), `REC-12` ($+0,19^\circ\text{C}$), serta **`REC-13` pada 13 Januari 2025 ($+0,29^\circ\text{C}$, rata-rata jendela $+0,48^\circ\text{C}$)**. Fakta empiris ini menegaskan bahwa anomali suhu lapisan tercampur *ERA5-Land* pada satu sel grid beresolusi $11,1\text{ km}$ belum dapat dijadikan indikator tunggal (*standalone detector*) untuk mendeteksi kejadian *upwelling* di Danau Maninjau, karena kematian ikan skala lokal ($10-50\text{ ton}$) di teluk-teluk KJA dapat dipicu oleh dinamika angin lokal atau deplesi oksigen biokimia setempat yang tidak tercermin pada rata-rata termal skala grid.

---
---

# BAB V — KESIMPULAN DAN SARAN

## A. Kesimpulan
Berdasarkan hasil audit data geospasial di *Google Earth Engine*, pengujian model melalui *3-Fold Rolling-Origin Cross-Validation* (2016–2023), evaluasi pada data uji independen (2024–2025), serta analisis eksploratif yang telah dilakukan, maka dapat ditarik kesimpulan yang menjawab rumusan masalah dan hipotesis penelitian sebagai berikut:

1. **Karakteristik Representasi Spasial ERA5-Land dan Kesesuaiannya terhadap Satelit (Jawaban Rumusan Masalah 1 & Hipotesis $H_1$):**
   * Sel grid utama *ERA5-Land* (`100,20° BT, -0,30° LS`, luas $123,64\text{ km}^2$) yang menaungi zona inti perairan Danau Maninjau mencakup **52,20% ($64,54\text{ km}^2$) perairan permanen JRC** dan menunjukkan perilaku stratifikasi vertikal pada modul *FLake* ($T_{\text{mix}} = 27,57^\circ\text{C}$ vs $T_{\text{bottom}} = 24,57^\circ\text{C}$). Namun, sekitar 47,80% luas grid masih berupa daratan perbukitan kaldera (elevasi rata-rata $589,89\text{ m}$ mdpl) dan konfigurasi internal *FLake* belum tervalidasi terhadap pengukuran lapangan.
   * Evaluasi sensitivitas terhadap *MODIS Terra* siang menunjukkan bahwa penerapan syarat kualitas cakupan ($\ge 15$ dari 22 piksel valid, $N=641$), homogenitas spasial (`std <= 0,5°C`), dan sudut pandang dekat nadir (`|view_angle| <= 30°`, $N=117$) mampu menurunkan MAE dari $2,49^\circ\text{C}$ menjadi **$1,10^\circ\text{C}$**, namun hanya menyisakan **3,2% hari pengamatan** dan menghasilkan korelasi yang masih lemah–moderat ($r = 0,223$ pada $\ge 15$ piksel; $r = 0,298$ pada filter ketat). Hasil evaluasi ini **mendukung Hipotesis Pertama ($H_1$)** dan menegaskan bahwa model *machine learning* dalam penelitian ini secara spesifik memprediksi **dinamika suhu lapisan tercampur keluaran reanalisis *ERA5-Land***, bukan suhu air aktual yang tervalidasi pengamatan langsung.

2. **Komparasi Kinerja Algoritma Random Forest, XGBoost, dan SVR serta Eksplorasi Kejadian Upwelling (Jawaban Rumusan Masalah 2 & Hipotesis $H_2, H_3$):**
   * Pada **Skema Utama (73 fitur: 18 fitur termal historis + 55 fitur kalender dan meteorologi $\le t$)** di periode uji independen (2024–2025), ketiga algoritma berhasil mengungguli kedua *Baseline Persistence* ($\hat{A}_{t+h}=A_t$ dan $\hat{T}_{t+h}=T_t$), sehingga **hasil pengujian mendukung Hipotesis Kedua ($H_2$)**. Berdasarkan kriteria utama RMSE dan $R^2$, algoritma **XGBoost** menjadi model terbaik pada kedua horizon prediksi, dengan **$\text{RMSE} = 0,1063^\circ\text{C}$, $\text{MAE} = 0,0794^\circ\text{C}$, dan $R^2\text{ Anomali} = 0,9281$** ($R^2\text{ Suhu} = 0,9573$, menurunkan RMSE sebesar **18,61%** dibanding *Persistence*) pada horizon **$t+1$**, serta **$\text{RMSE} = 0,1952^\circ\text{C}$, $\text{MAE} = 0,1501^\circ\text{C}$, dan $R^2\text{ Anomali} = 0,7574$** ($R^2\text{ Suhu} = 0,8564$, menurunkan RMSE sebesar **9,46%**) pada horizon **$t+3$**. Adapun berdasarkan metrik pendamping MAE pada horizon $t+3$, algoritma **SVR (RBF)** mencatatkan rata-rata kesalahan absolut terendah sebesar **$0,1444^\circ\text{C}$** ($\text{RMSE} = 0,1980^\circ\text{C}$, $R^2\text{ Anomali} = 0,7503$).
   * Analisis *feature importance* dan uji ablasi menunjukkan bahwa tingginya akurasi pada Skema Utama sangat ditopang oleh fitur anomali suhu hari sebelumnya (`anom_doy_c` dan `anom_doy_lag1`). Ketika seluruh 18 fitur suhu danau historis dihapus pada **Skema Ablasi Murni Meteorologi (55 fitur $\le t$)**, nilai $R^2\text{ Anomali}$ turun secara substansial menjadi $0,5156-0,6545$ ($t+1$) dan $0,3674-0,5615$ ($t+3$), sehingga **hasil pengujian mendukung Hipotesis Ketiga ($H_3$)**. Pada skema tanpa riwayat suhu danau tersebut, **XGBoost** unggul pada horizon $t+1$ ($\text{RMSE} = 0,2331^\circ\text{C}$, $R^2\text{ Anomali} = 0,6545$), sedangkan **SVR (RBF)** menjadi algoritma terbaik pada horizon $t+3$ ($\text{RMSE} = 0,2624^\circ\text{C}$, $\text{MAE} = 0,2045^\circ\text{C}$, $R^2\text{ Anomali} = 0,5615$) dengan mengandalkan fitur rata-rata bergulir 7–14 hari (`air_temp_mean_c_mean14d`, `air_temp_mean_c_mean7d`, `solar_net_rad_mjm2_mean14d`, dan `thermal_net_rad_mjm2_mean14d`).
   * Evaluasi deskriptif-eksploratif terhadap 15 kandidat catatan historis menunjukkan bahwa meskipun episode kematian ikan skala masif (Februari 2022 dan November–Desember 2025) serta periode observasi kenaikan sulfida BRIN (November 2022) bertepatan dengan anomali dingin kuat pada *ERA5-Land* dan berhasil diikuti oleh prediksi *XGBoost* ($-0,54^\circ\text{C}$ hingga $-0,95^\circ\text{C}$), terdapat 5 dari 10 catatan harian spesifik (termasuk Januari 2025 sebesar $+0,29^\circ\text{C}$) yang memiliki anomali positif. Dengan demikian, anomali suhu lapisan tercampur *ERA5-Land* hanya berperan sebagai informasi konteks termal skala regional dan belum terbukti sebagai indikator tunggal pendeteksi *upwelling*.

---

## B. Saran
Berdasarkan temuan dan keterbatasan penelitian yang telah teridentifikasi, disarankan beberapa langkah teknis yang dapat ditindaklanjuti pada penelitian berikutnya:
1. **Integrasi Data Pengukuran *In-Situ* Berkelanjutan:** Penelitian selanjutnya disarankan memasang atau mengintegrasikan data sensor termistor vertikal (*buoy* / *data logger* suhu air dan oksigen terlarut pada kedalaman $0\text{ m}, 5\text{ m}, 10\text{ m}, \text{dan } 20\text{ m}$) di zona KJA Danau Maninjau guna mengkalibrasi bias antara suhu lapisan tercampur *ERA5-Land*, suhu kulit satelit *MODIS/Landsat*, dan suhu kolom air sebenarnya.
2. **Pemeriksaan Parameterisasi Internal *FLake* dan Pemodelan Hidrodinamika Spasial:** Mengingat satu sel grid *ERA5-Land* ($11,1\text{ km}$) merata-ratakan seluruh cekungan danau, penelitian lanjutan perlu mengkaji konfigurasi parameter kedalaman danau di dalam model *FLake* ECMWF atau menggabungkan prediksi *machine learning* dengan model hidrodinamika 3-dimensi untuk menangkap proses *upwelling* lokal di sub-teluk (seperti Tanjung Sani dan Bayua).
3. **Penyelarasan Jam Pengamatan (*Diurnal Matching*) pada Data Satelit:** Untuk meningkatkan kesesuaian antara reanalisis dan satelit, penelitian berikutnya dapat mengekstrak produk *ERA5-Land Hourly* (`ECMWF/ERA5_LAND/HOURLY`) tepat pada jam lintas lokal satelit (`view_time_hr`, yaitu pukul 10:00 WIB untuk *MODIS Terra* dan *Landsat* serta pukul 14:00 WIB untuk *MODIS Aqua*).
4. **Penanganan Jeda Rilis Data (*Operational Latency*) untuk Sistem Prakiraan:** Apabila model akan diimplementasikan sebagai sistem prakiraan operasional harian, pipa input perlu diintegrasikan dengan data prakiraan cuaca numerik (*Numerical Weather Prediction* seperti *NOAA GFS* atau *ECMWF IFS* di GEE) atau stasiun cuaca otomatis (*AWS*) lokal untuk mengatasi jeda rilis sekitar 5 hari pada produk *ERA5-Land Daily*.
