from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Dashboard Analisis ERA5-Land — Danau Maninjau",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_locked_artifacts():
    grid_df = pd.read_csv(BASE_DIR / "maninjau_audit_grid_era5.csv")
    modis_sens = pd.read_csv(BASE_DIR / "evaluasi_khusus_era5_vs_modis_terra.csv")
    rocv_df = pd.read_csv(BASE_DIR / "hasil_tuning_validasi_2022_2023.csv")
    test_df = pd.read_csv(BASE_DIR / "hasil_evaluasi_test_2024_2025.csv")
    feat_df = pd.read_csv(BASE_DIR / "hasil_feature_importance.csv")
    inv_df = pd.read_csv(BASE_DIR / "inventaris_upwelling_maninjau_2016_2025.csv")
    upw_df = pd.read_csv(BASE_DIR / "hasil_evaluasi_eksploratif_upwelling.csv")
    pred_t1 = pd.read_csv(
        BASE_DIR / "prediksi_uji_independen_t1_2024_2025.csv",
        parse_dates=["date", "target_date"]
    )
    pred_t3 = pd.read_csv(
        BASE_DIR / "prediksi_uji_independen_t3_2024_2025.csv",
        parse_dates=["date", "target_date"]
    )
    return grid_df, modis_sens, rocv_df, test_df, feat_df, inv_df, upw_df, pred_t1, pred_t3


grid_df, modis_sens, rocv_df, test_df, feat_df, inv_df, upw_df, pred_t1, pred_t3 = load_locked_artifacts()

# Header utama & pembatasan klaim ilmiah
st.title("📊 Dashboard Analisis ERA5-Land — Danau Maninjau")
st.caption(
    "**Pendukung Demonstrasi Seminar Hasil (Skema 1: Pengembangan Algoritma / Model / Analisis)** — "
    "Komparasi Algoritma *Random Forest*, *XGBoost*, dan *Support Vector Regression* (SVR) untuk Prediksi Anomali Suhu "
    "Lapisan Tercampur Reanalisis *ERA5-Land* di Kawasan Danau Maninjau Berbasis *Google Earth Engine* (2016–2025)."
)

st.info(
    "**Catatan Batasan Ilmiah (Read-Only Experimental Viewer):** Dashboard ini menampilkan hasil eksperimen retrospektif "
    "yang telah dikunci dari periode Latih/ROCV (`2016–2023`) dan Uji Independen (`2024–2025`). Target yang diprediksi adalah "
    "**anomali suhu lapisan tercampur keluaran reanalisis ERA5-Land (`era5_lake_mix_temp_c`)**, bukan pengukuran suhu air "
    "lapangan (*in-situ*) dan bukan sistem deteksi *upwelling* operasional secara langsung."
)

# Baris metrik ringkasan utama
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="Model Terbaik t+1 (73 Fitur)",
        value="XGBoost (0,1063°C)",
        delta="-18,61% RMSE vs Persistence"
    )
    st.caption("MAE = 0,0794°C · R² Anomali = 0,9281")
with col2:
    st.metric(
        label="Model Terbaik t+3 (73 Fitur - RMSE)",
        value="XGBoost (0,1952°C)",
        delta="-9,46% RMSE vs Persistence"
    )
    st.caption("MAE Terendah t+3: SVR = 0,1444°C · R² XGB = 0,7574")
with col3:
    st.metric(
        label="Skema Ablasi Meteo (55 Fitur)",
        value="XGB (t+1) / SVR (t+3)",
        delta="R² = 0,6545 (t+1) & 0,5615 (t+3)"
    )
    st.caption("Tanpa 18 fitur riwayat suhu/anomali danau")
with col4:
    st.metric(
        label="Audit Grid & MODIS Terra Ketat",
        value="52,20% Air | MAE 1,10°C",
        delta="N = 117 dari 3.653 hari (3,2%)"
    )
    st.caption("Korelasi MODIS Terra ketat: r = 0,298 (ρ = 0,330)")

st.divider()

# Navigasi 6 menu sesuai rekomendasi
menu = st.sidebar.radio(
    "Pilih Menu Eksplorasi Hasil:",
    [
        "1. Ringkasan Penelitian & Batasan Data",
        "2. Komparasi RF, XGBoost, SVR & Baseline",
        "3. Grafik Target Referensi ERA5-Land vs Prediksi",
        "4. Feature Importance & Ablasi (73 vs 55 Fitur)",
        "5. Audit ERA5-Land vs MODIS/Landsat",
        "6. Eksplorasi 15 Catatan Historis Upwelling",
    ]
)

if menu == "1. Ringkasan Penelitian & Batasan Data":
    st.subheader("1. Ringkasan Desain Penelitian, Batasan Validitas, dan Poin Penting Sidang")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(
            """
            #### A. Pembagian Periode Data & Pengendalian *Lookahead Bias*
            * **Rentang Studi:** 1 Januari 2016 – 31 Desember 2025 (**3.653 hari**, ketersediaan ERA5-Land & CHIRPS 100%).
            * **Indeks Kalender Konsisten (`cal_doy_365`):** Pemetaan pasangan `(bulan, hari)` ke 1..365 (29 Februari pada tahun kabisat 2016, 2020, 2024 dipetakan ke indeks 59 bersama 28 Februari) agar tidak bergeser pasca-Februari.
            * **Isolasi Klimatologi & Pemisahan Kronologis Target:** Rata-rata klimatologi harian (dihaluskan 15 hari melingkar) dihitung **hanya dari data latih aktif** pada masing-masing *fold* (`date <= tr_end`), dan sampel pelatihan dipotong agar tanggal target tidak menyeberang ke periode validasi/uji (`target_date <= tr_end`).
            * **Seleksi Hyperparameter (*3-Fold Rolling-Origin CV*):**
              - *Fold 1:* Latih `2016–2019` → Validasi `2020–2021`
              - *Fold 2:* Latih `2016–2020` → Validasi `2021–2022`
              - *Fold 3:* Latih `2016–2021` → Validasi `2022–2023`
            * **Pengujian Final Independen (*Out-of-Sample*):** Dilatih satu kali pada `2016–2023`, diuji satu kali pada **`2024–2025`** ($N=730$ hari untuk $t+1$; $N=728$ hari untuk $t+3$).
            """
        )
    with c_right:
        st.markdown(
            """
            #### B. Ringkasan Jawaban Kunci untuk Pertanyaan Penguji (Sempro & Semhas)
            1. **Mengapa memilih `ERA5-Land` sebagai target, bukan satelit MODIS/Landsat?**
               Karena di kaldera tropis Bukit Barisan, tutupan awan menyebabkan MODIS Terra siang hanya tersedia pada 25,2% hari (dan hanya **3,2% hari [117 hari]** setelah penyaringan kualitas ketat $\ge 15/22$ piksel, homogenitas $\le 0,5^\circ\text{C}$, dan *near-nadir* $\le 30^\circ$), sedangkan regresi deret waktu berlag harian membutuhkan deret kontinu 100%.
            2. **Apa perbedaan utama dari penelitian Syakir dkk. (2025)?**
               Syakir dkk. (2025) membentuk label kelas potensi *upwelling* dari klasterisasi tanpa pengawasan (*K-Means*) pada variabel cuaca. Penelitian ini memodelkan besaran kontinu **anomali suhu lapisan tercampur reanalisis** dengan pengendalian *lookahead bias*, mengujinya terhadap **dua *Baseline Persistence*** ($\hat{A}_{t+h}=A_t$ dan $\hat{T}_{t+h}=T_t$), melakukan **uji ablasi fitur (73 vs 55 fitur)**, serta mengaudit keterbatasannya terhadap satelit dan 15 catatan kejadian.
            3. **Mengapa performa menurun dari $t+1$ ke $t+3$ dan dari 73 fitur ke 55 fitur?**
               Pada $t+3$, ketidakpastian gangguan cuaca atmosfer selama 3 hari ke depan meningkat sehingga RMSE naik dari $0,1063^\circ\text{C}$ menjadi $0,1952^\circ\text{C}$. Penghapusan 18 fitur termal danau pada Skema Ablasi (55 fitur) menurunkan $R^2$ dari $0,9281$ ke $0,6545$ ($t+1$), yang menunjukkan dominannya informasi riwayat suhu dalam kemampuan prediksi jangka pendek.
            """
        )

    st.markdown("#### C. Bagan Kerangka Pemikiran Penelitian (Gambar 1)")
    img_kerangka = BASE_DIR / "Gambar_0_Kerangka_Pemikiran.png"
    if img_kerangka.is_file():
        st.image(str(img_kerangka), use_container_width=True)

elif menu == "2. Komparasi RF, XGBoost, SVR & Baseline":
    st.subheader("2. Komparasi Kinerja Random Forest, XGBoost, SVR, dan Dua Baseline Persistence")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sel_hor = st.selectbox("Pilih Horizon Prediksi:", ["Semua Horizon (t+1 & t+3)", "t+1", "t+3"])
    with f_col2:
        sel_reg = st.selectbox(
            "Pilih Rezim Fitur Input:",
            ["Semua Rezim (73 & 55 Fitur)", "1_Utama (73 Fitur: Termal + Meteo)", "2_Ablasi (55 Fitur: Murni Meteo)"]
        )

    sub_test = test_df.copy()
    if sel_hor != "Semua Horizon (t+1 & t+3)":
        sub_test = sub_test[sub_test["Horizon"] == sel_hor]
    if sel_reg.startswith("1_Utama"):
        sub_test = sub_test[sub_test["Rezim_Fitur"].str.startswith("1_Utama")]
    elif sel_reg.startswith("2_Ablasi"):
        sub_test = sub_test[sub_test["Rezim_Fitur"].str.startswith("2_Ablasi")]

    st.markdown("##### Tabel Hasil Evaluasi pada Data Uji Independen (2024–2025)")
    st.dataframe(
        sub_test.style.format({
            "Test_Anom_MAE": "{:.4f}",
            "Test_Anom_RMSE": "{:.4f}",
            "Test_Anom_R2": "{:.4f}",
            "Test_Temp_MAE": "{:.4f}",
            "Test_Temp_RMSE": "{:.4f}",
            "Test_Temp_R2": "{:.4f}",
            "Skill_RMSE_vs_PersistAnom_Pct": "{:+.2f}%",
        }),
        use_container_width=True
    )

    st.markdown("##### Tabel Hasil Seleksi Hyperparameter 3-Fold Rolling-Origin CV (2016–2023)")
    st.dataframe(
        rocv_df.style.format({
            "Fold1_2020_2021_RMSE": "{:.4f}",
            "Fold2_2021_2022_RMSE": "{:.4f}",
            "Fold3_2022_2023_RMSE": "{:.4f}",
            "ROCV_Mean_RMSE": "{:.4f}",
            "ROCV_Std_RMSE": "{:.4f}",
            "ROCV_Mean_MAE": "{:.4f}",
            "ROCV_Mean_R2": "{:.4f}",
        }),
        use_container_width=True
    )

    img_g2 = BASE_DIR / "Gambar_2_Komparasi_RMSE_Uji_2024_2025.png"
    if img_g2.is_file():
        st.markdown("##### Visualisasi Perbandingan RMSE pada Data Uji Independen (Gambar 3 Naskah)")
        st.image(str(img_g2), use_container_width=True)

elif menu == "3. Grafik Target Referensi ERA5-Land vs Prediksi":
    st.subheader("3. Deret Waktu Target Referensi ERA5-Land vs Prediksi Model (Periode Uji Independen 2024–2025)")

    c_h, c_d1, c_d2 = st.columns([1.2, 1, 1])
    with c_h:
        hor_choice = st.radio("Pilih Horizon:", ["Horizon t+1 (1 Hari ke Depan)", "Horizon t+3 (3 Hari ke Depan)"], horizontal=True)
    df_p = pred_t1 if "t+1" in hor_choice else pred_t3
    min_d = df_p["target_date"].min().date()
    max_d = df_p["target_date"].max().date()
    with c_d1:
        start_d = st.date_input("Tanggal Mulai:", value=min_d, min_value=min_d, max_value=max_d)
    with c_d2:
        end_d = st.date_input("Tanggal Akhir:", value=max_d, min_value=min_d, max_value=max_d)

    sub_p = df_p[(df_p["target_date"].dt.date >= start_d) & (df_p["target_date"].dt.date <= end_d)].copy()

    fig, ax = plt.subplots(figsize=(12.2, 4.2), dpi=200)
    ax.axhline(0, color="#666666", linestyle="--", linewidth=0.9)
    ax.plot(sub_p["target_date"], sub_p["target_anom"], label="Target Referensi ERA5-Land", color="black", linewidth=1.6)
    ax.plot(sub_p["target_date"], sub_p["pred_anom_XGBoost"], label="Prediksi XGBoost (Skema Utama 73 Fitur)", color="#e41a1c", linewidth=1.25)
    ax.plot(sub_p["target_date"], sub_p["pred_anom_SVR"], label="Prediksi SVR (Skema Utama 73 Fitur)", color="#377eb8", linewidth=1.05, alpha=0.85)
    ax.plot(sub_p["target_date"], sub_p["pred_anom_RandomForest"], label="Prediksi Random Forest (Skema Utama 73 Fitur)", color="#2ca25f", linewidth=0.95, alpha=0.75)
    ax.set_ylabel("Anomali Suhu Lapisan Tercampur (°C)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Tanggal Target (Periode Uji Independen)", fontsize=9.5, fontweight="bold")
    ax.set_title(f"Perbandingan Target Referensi ERA5-Land vs Prediksi Model — {hor_choice}", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower left", ncol=2, fontsize=8.5, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Lihat Tabel Nilai Prediksi Harian (CSV Uji Independen)"):
        st.dataframe(sub_p, use_container_width=True)

elif menu == "4. Feature Importance & Ablasi (73 vs 55 Fitur)":
    st.subheader("4. Analisis Feature Importance & Perbandingan Skema Utama (73 Fitur) vs Ablasi (55 Fitur)")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        fi_hor = st.selectbox("Horizon:", ["t+1", "t+3"])
    with col_f2:
        fi_reg = st.selectbox("Rezim Fitur:", ["1_Utama", "2_Ablasi_Murni_Meteo"])
    with col_f3:
        fi_mod = st.selectbox("Algoritma Pohon:", ["XGBoost", "Random Forest"])

    sub_fi = feat_df[
        (feat_df["Horizon"] == fi_hor) &
        (feat_df["Rezim_Fitur"] == fi_reg) &
        (feat_df["Model"] == fi_mod)
    ].copy()

    st.markdown(f"##### Top-15 Fitur dengan Kontribusi Relatif Tertinggi ({fi_mod} — {fi_reg}, Horizon {fi_hor})")
    st.dataframe(sub_fi.style.format({"Importance": "{:.4f}"}), use_container_width=True)

    img_g4 = BASE_DIR / "Gambar_4_Feature_Importance_XGBoost.png"
    if img_g4.is_file():
        st.markdown("##### Grafik Perbandingan Kontribusi Fitur XGBoost: 73 Fitur vs 55 Fitur (Gambar 5 Naskah)")
        st.image(str(img_g4), use_container_width=True)

elif menu == "5. Audit ERA5-Land vs MODIS/Landsat":
    st.subheader("5. Hasil Audit Spasial Sel Grid ERA5-Land & Sensitivitas Kualitas Piksel Satelit")

    st.markdown("##### A. Audit Spasial 4 Sel Grid ERA5-Land di Sekitar Kaldera Danau Maninjau (`maninjau_audit_grid_era5.csv`)")
    st.dataframe(grid_df, use_container_width=True)

    st.markdown("##### B. Analisis Sensitivitas Kesesuaian Suhu Harian ERA5-Land terhadap MODIS Terra Siang (`evaluasi_khusus_era5_vs_modis_terra.csv`)")
    st.dataframe(
        modis_sens.style.format({
            "Persen_Hari_Tersedia": "{:.2f}%",
            "Mean_MODIS_Terra_C": "{:.2f}",
            "Mean_ERA5_Mix_C": "{:.2f}",
            "Bias_Terra_minus_ERA5_Mix_C": "{:+.2f}",
            "MAE_vs_ERA5_Mix_C": "{:.2f}",
            "RMSE_vs_ERA5_Mix_C": "{:.2f}",
            "Pearson_r_vs_ERA5_Mix": "{:.3f}",
            "Spearman_rho_vs_ERA5_Mix": "{:.3f}",
        }),
        use_container_width=True
    )

    img_g1 = BASE_DIR / "Gambar_1_Audit_ERA5_vs_MODIS_Terra.png"
    if img_g1.is_file():
        st.markdown("##### Visualisasi Sensitivitas Galat (MAE), Bias Dingin, dan Korelasi MODIS Terra Siang (Gambar 2 Naskah)")
        st.image(str(img_g1), use_container_width=True)

elif menu == "6. Eksplorasi 15 Catatan Historis Upwelling":
    st.subheader("6. Evaluasi Deskriptif-Eksploratif pada 15 Kandidat Catatan Historis (2016–2025)")
    st.warning(
        "**Penting:** Evaluasi ini bersifat deskriptif-eksploratif. Dari 10 catatan harian spesifik (`Exact_Day`), "
        "terdapat 5 kejadian yang justru memiliki anomali suhu positif pada ERA5-Land (termasuk `REC-13` pada Januari 2025 "
        "sebesar $+0,29^\\circ\\text{C}$). Oleh karena itu, anomali suhu reanalisis skala grid 11,1 km tidak diklaim sebagai "
        "pendeteksi tunggal kejadian *upwelling*."
    )

    merged_upw = pd.merge(
        upw_df,
        inv_df[["record_id", "reported_event_date", "start_date", "end_date", "location_nagari", "source_agency", "source_url", "notes"]],
        on="record_id",
        how="left"
    )

    cat_filter = st.radio(
        "Pilih Kategori Presisi Tanggal Catatan:",
        ["Exact_Day (N=10 Catatan Harian Spesifik)", "Multi_Day (N=3 Episode Multi-Hari)", "Month_Only (N=2 Catatan Bulanan)"],
        horizontal=True
    )
    if cat_filter.startswith("Exact_Day"):
        sub_u = merged_upw[merged_upw["date_precision"] == "Exact_Day"]
    elif cat_filter.startswith("Multi_Day"):
        sub_u = merged_upw[merged_upw["date_precision"].str.startswith("Multi_Day")]
    else:
        sub_u = merged_upw[merged_upw["date_precision"] == "Month_Only"]

    st.dataframe(sub_u, use_container_width=True)

    img_g5 = BASE_DIR / "Gambar_5_Evaluasi_Deskriptif_Upwelling_ExactDay.png"
    if img_g5.is_file() and cat_filter.startswith("Exact_Day"):
        st.markdown("##### Distribusi Deskriptif Anomali Suhu ERA5-Land pada 10 Catatan Harian Spesifik (Gambar 6 Naskah)")
        st.image(str(img_g5), use_container_width=True)
