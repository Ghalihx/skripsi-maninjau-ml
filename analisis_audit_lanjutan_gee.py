import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
daily_df = pd.read_csv(BASE_DIR / 'maninjau_daily_gee_2016_2025.csv')
grid_df = pd.read_csv(BASE_DIR / 'maninjau_audit_grid_era5.csv')
modis_df = pd.read_csv(BASE_DIR / 'maninjau_audit_modis_pixels_2016_2025.csv')
landsat_df = pd.read_csv(BASE_DIR / 'maninjau_landsat_lst_2016_2025.csv')

daily_df['date'] = pd.to_datetime(daily_df['date'])
modis_df['date'] = pd.to_datetime(modis_df['date'])
landsat_df['date'] = pd.to_datetime(landsat_df['date'])

merged = pd.merge(daily_df, modis_df, on='date', how='left')

print("=" * 88)
print("1. AUDIT REPRESENTASI FLAKE & KONTRAS SPASIAL 4 GRID ERA5-LAND DI KALDERA MANINJAU")
print("=" * 88)
grid_df['delta_skin_minus_air_c'] = grid_df['mean_skin_temp_c'] - grid_df['mean_air_temp_2m_c']
grid_df['delta_mix_minus_bottom_c'] = grid_df['mean_lake_mix_temp_c'] - grid_df['mean_lake_bottom_temp_c']
cols_g = [
    'grid_id', 'intersects_roi_core_2km', 'jrc_water_fraction_pct', 'srtm_elev_mean_m',
    'mean_lake_mix_temp_c', 'mean_lake_bottom_temp_c', ' delta_mix_minus_bottom_c'.strip(),
    'mean_lake_mix_depth_m', 'mean_skin_temp_c', 'mean_air_temp_2m_c', 'delta_skin_minus_air_c'
]
print(grid_df[cols_g].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n" + "=" * 88)
print("2. AUDIT GEOMETRI PENGAMATAN, PIKSEL VALID, & KONTAMINASI SUB-PIKSEL MODIS")
print("=" * 88)

# Check why extreme cold values occur in Terra Day even when valid_pixels >= 15
td = merged[merged['terra_day_valid_pixels'] > 0].copy()
td['abs_view_angle'] = td['terra_day_view_angle_deg'].abs()
td['diff_terra_minus_era5_mix'] = td['terra_day_mean_c'] - td['era5_lake_mix_temp_c']

print(f"\n[A. Statistik Waktu Lintas Lokal (view_time_hr) & Sudut Pandang (view_angle_deg)]")
for pref, label in [
    ('terra_day', 'MODIS Terra Siang'),
    ('aqua_day', 'MODIS Aqua Siang'),
    ('terra_night', 'MODIS Terra Malam'),
    ('aqua_night', 'MODIS Aqua Malam')
]:
    sub = merged[merged[f'{pref}_valid_pixels'] > 0].copy()
    sub['abs_ang'] = sub[f'{pref}_view_angle_deg'].abs()
    print(f"  - {label:18s} (N={len(sub):3d}): Jam Lintas = {sub[f'{pref}_view_time_hr'].mean():.2f} ± {sub[f'{pref}_view_time_hr'].std():.2f} WIB "
          f"(Min={sub[f'{pref}_view_time_hr'].min():.1f}, Max={sub[f'{pref}_view_time_hr'].max():.1f}) | "
          f"|Sudut Pandang| Mean={sub['abs_ang'].mean():.1f}°, Median Piksel={sub[f'{pref}_valid_pixels'].median():.0f}/22")

print("\n[B. Asosiasi Statistik Faktor Geometri/Spasial dengan Suhu MODIS Terra Siang (N=922)]")
print("  Catatan: Korelasi di bawah ini bersifat asosiatif pada sampel, bukan bukti kausalitas tunggal.")
print(f"  - Korelasi terra_day_mean_c vs |view_angle_deg| : r = {td['terra_day_mean_c'].corr(td['abs_view_angle']):+.3f}")
print(f"  - Korelasi terra_day_mean_c vs valid_pixels     : r = {td['terra_day_mean_c'].corr(td['terra_day_valid_pixels']):+.3f}")
print(f"  - Korelasi terra_day_std_c  vs |diff_from_ERA5| : r = {td['terra_day_std_c'].corr(td['diff_terra_minus_era5_mix'].abs()):+.3f}")

# Compare primary QA/geometry filters; keep max_c and >=24C strictly labeled as secondary sensitivity checks
print("\n[C. Perbandingan Bertahap Filter Kualitas Cakupan & Geometri pada MODIS Siang]")
filter_rows = []

for pref, sensor_label in [('terra_day', 'MODIS Terra Siang (~10:30 WIB)'), ('aqua_day', 'MODIS Aqua Siang (~13:30 WIB)')]:
    base = merged[merged[f'{pref}_valid_pixels'] >= 1].copy()
    base['abs_angle'] = base[f'{pref}_view_angle_deg'].abs()
    
    schemes = [
        ('1_Utama: Semua Hari Lolos QC (>=1 piksel)', base, f'{pref}_mean_c'),
        ('2_Utama: Cakupan Piksel Memadai (>=15 dari 22 piksel)', base[base[f'{pref}_valid_pixels'] >= 15], f'{pref}_mean_c'),
        ('3_Utama: Cakupan Penuh (22/22 piksel) & Homogen (std <= 0.5 C)', base[(base[f'{pref}_valid_pixels'] == 22) & (base[f'{pref}_std_c'] <= 0.5)], f'{pref}_mean_c'),
        ('4_Utama: Cakupan >=15, Homogen (std <= 0.5 C), Near-Nadir (|angle| <= 30 deg)', base[(base[f'{pref}_valid_pixels'] >= 15) & (base[f'{pref}_std_c'] <= 0.5) & (base['abs_angle'] <= 30)], f'{pref}_mean_c'),
        ('5_Sensitivitas_Sekunder: Nilai Maksimum ROI (max_c, >=15 piksel, |angle| <= 30 deg)', base[(base[f'{pref}_valid_pixels'] >= 15) & (base['abs_angle'] <= 30)], f'{pref}_max_c'),
        ('6_Sensitivitas_Sekunder: Potong Ambang >=24 C (Bukan Standar Validasi Utama)', base[(base[f'{pref}_valid_pixels'] >= 15) & (base['abs_angle'] <= 30) & (base[f'{pref}_mean_c'] >= 24.0)], f'{pref}_mean_c'),
    ]
    
    for s_name, df_s, val_col in schemes:
        n_obs = len(df_s)
        if n_obs > 2:
            mean_val = df_s[val_col].mean()
            min_val = df_s[val_col].min()
            max_val = df_s[val_col].max()
            bias_era5 = (df_s[val_col] - df_s['era5_lake_mix_temp_c']).mean()
            mae_era5 = (df_s[val_col] - df_s['era5_lake_mix_temp_c']).abs().mean()
            r_era5_mix = df_s[val_col].corr(df_s['era5_lake_mix_temp_c'])
            r_era5_skin = df_s[val_col].corr(df_s['era5_skin_temp_c'])
            r_air = df_s[val_col].corr(df_s['air_temp_mean_c'])
        else:
            mean_val = min_val = max_val = bias_era5 = mae_era5 = r_era5_mix = r_era5_skin = r_air = np.nan
        filter_rows.append({
            'Sensor': sensor_label,
            'Skema_Audit_Geometri_Piksel': s_name,
            'N': n_obs,
            'Mean_Sat_C': mean_val,
            'Min_Sat_C': min_val,
            'Max_Sat_C': max_val,
            'Bias_vs_ERA5_Mix_C': bias_era5,
            'MAE_vs_ERA5_Mix_C': mae_era5,
            'r_vs_ERA5_Mix': r_era5_mix,
            'r_vs_ERA5_Skin': r_era5_skin
        })

audit_modis_detail_df = pd.DataFrame(filter_rows)
audit_modis_detail_df.to_csv(BASE_DIR / 'hasil_audit_mendalam_modis_dan_grid.csv', index=False)
print(audit_modis_detail_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# Also check monthly / seasonal aggregation correlation between MODIS Terra/Aqua (>=15 pixels) and ERA5
print("\n[D. Perbandingan Skala Bulanan (Monthly Mean) MODIS (>=15 piksel, >=24 C) vs ERA5-Land]")
for pref, label in [('terra_day', 'MODIS Terra Siang'), ('aqua_day', 'MODIS Aqua Siang')]:
    sub = merged[(merged[f'{pref}_valid_pixels'] >= 15) & (merged[f'{pref}_mean_c'] >= 24.0)].copy()
    sub['ym'] = sub['date'].dt.to_period('M')
    m_grp = sub.groupby('ym').agg(
        n_days=(f'{pref}_mean_c', 'count'),
        sat_mean=(f'{pref}_mean_c', 'mean'),
        era5_mix=('era5_lake_mix_temp_c', 'mean'),
        era5_skin=('era5_skin_temp_c', 'mean')
    )
    m_grp_3 = m_grp[m_grp['n_days'] >= 3]
    print(f"  - {label:18s} | Semua Bulan (N={len(m_grp)} bln): r_mix={m_grp['sat_mean'].corr(m_grp['era5_mix']):+.3f} | "
          f"Bulan >=3 obs (N={len(m_grp_3)} bln): r_mix={m_grp_3['sat_mean'].corr(m_grp_3['era5_mix']):+.3f}, MAE={(m_grp_3['sat_mean'] - m_grp_3['era5_mix']).abs().mean():.2f} C")

# ==============================================================================
# 3. EVALUASI KHUSUS KESESUAIAN SUHU ERA5 vs MODIS TERRA SIANG (SENSITIVITAS PIKSEL)
# ==============================================================================
print("\n" + "=" * 88)
print("3. EVALUASI KHUSUS KESESUAIAN SUHU ERA5 vs MODIS TERRA SIANG (ANALISIS SENSITIVITAS)")
print("=" * 88)

terra_sens_rows = []
for k in [1, 5, 10, 15, 18, 20, 22]:
    s = td[td['terra_day_valid_pixels'] >= k].copy()
    diff_mix = s['terra_day_mean_c'] - s['era5_lake_mix_temp_c']
    diff_skin = s['terra_day_mean_c'] - s['era5_skin_temp_c']
    terra_sens_rows.append({
        'Kategori_Sensitivitas': f'1_Ambang_Piksel >= {k:2d}/22',
        'Syarat_Filter': f'terra_day_valid_pixels >= {k}',
        'N_Hari': len(s),
        'Persen_Hari_10Thn': len(s) / len(merged) * 100.0,
        'Terra_Mean_C': s['terra_day_mean_c'].mean(),
        'Terra_Std_Temporal_C': s['terra_day_mean_c'].std(),
        'Terra_Min_C': s['terra_day_mean_c'].min(),
        'Terra_Max_C': s['terra_day_mean_c'].max(),
        'ERA5_Mix_Mean_C': s['era5_lake_mix_temp_c'].mean(),
        'Bias_Terra_minus_ERA5_Mix_C': diff_mix.mean(),
        'MAE_vs_ERA5_Mix_C': diff_mix.abs().mean(),
        'RMSE_vs_ERA5_Mix_C': np.sqrt((diff_mix ** 2).mean()),
        'Pearson_r_vs_ERA5_Mix': s['terra_day_mean_c'].corr(s['era5_lake_mix_temp_c'], method='pearson'),
        'Spearman_rho_vs_ERA5_Mix': s['terra_day_mean_c'].corr(s['era5_lake_mix_temp_c'], method='spearman'),
        'Pearson_r_vs_ERA5_Skin': s['terra_day_mean_c'].corr(s['era5_skin_temp_c'], method='pearson'),
        'Proporsi_Dibawah_24C_Pct': (s['terra_day_mean_c'] < 24.0).mean() * 100.0
    })

# Tambahkan variasi kualitas pada ambang utama >= 15 piksel
s15 = td[td['terra_day_valid_pixels'] >= 15].copy()
extra_checks = [
    ('2_Fokus_15Piksel + Homogen_Spasial (std<=0.5C)', 'valid_pixels>=15 & std_c<=0.5', s15[s15['terra_day_std_c'] <= 0.5]),
    ('3_Fokus_15Piksel + Near_Nadir (|angle|<=30deg)', 'valid_pixels>=15 & |angle|<=30', s15[s15['abs_view_angle'] <= 30]),
    ('4_Fokus_15Piksel + Homogen + Near_Nadir', 'valid_pixels>=15 & std_c<=0.5 & |angle|<=30', s15[(s15['terra_day_std_c'] <= 0.5) & (s15['abs_view_angle'] <= 30)]),
    ('5_Fokus_15Piksel_Periode_Uji_2024_2025', 'valid_pixels>=15 & year in 2024-2025', s15[s15['year'] >= 2024]),
]

for kat, syarat, s in extra_checks:
    diff_mix = s['terra_day_mean_c'] - s['era5_lake_mix_temp_c']
    terra_sens_rows.append({
        'Kategori_Sensitivitas': kat,
        'Syarat_Filter': syarat,
        'N_Hari': len(s),
        'Persen_Hari_10Thn': len(s) / len(merged) * 100.0,
        'Terra_Mean_C': s['terra_day_mean_c'].mean(),
        'Terra_Std_Temporal_C': s['terra_day_mean_c'].std(),
        'Terra_Min_C': s['terra_day_mean_c'].min(),
        'Terra_Max_C': s['terra_day_mean_c'].max(),
        'ERA5_Mix_Mean_C': s['era5_lake_mix_temp_c'].mean(),
        'Bias_Terra_minus_ERA5_Mix_C': diff_mix.mean(),
        'MAE_vs_ERA5_Mix_C': diff_mix.abs().mean(),
        'RMSE_vs_ERA5_Mix_C': np.sqrt((diff_mix ** 2).mean()),
        'Pearson_r_vs_ERA5_Mix': s['terra_day_mean_c'].corr(s['era5_lake_mix_temp_c'], method='pearson'),
        'Spearman_rho_vs_ERA5_Mix': s['terra_day_mean_c'].corr(s['era5_lake_mix_temp_c'], method='spearman'),
        'Pearson_r_vs_ERA5_Skin': s['terra_day_mean_c'].corr(s['era5_skin_temp_c'], method='pearson'),
        'Proporsi_Dibawah_24C_Pct': (s['terra_day_mean_c'] < 24.0).mean() * 100.0
    })

terra_sens_df = pd.DataFrame(terra_sens_rows)
terra_sens_df.to_csv(BASE_DIR / 'evaluasi_khusus_era5_vs_modis_terra.csv', index=False)
print(terra_sens_df[['Kategori_Sensitivitas', 'N_Hari', 'Terra_Mean_C', 'Terra_Min_C', 'Terra_Max_C', 'Bias_Terra_minus_ERA5_Mix_C', 'MAE_vs_ERA5_Mix_C', 'RMSE_vs_ERA5_Mix_C', 'Pearson_r_vs_ERA5_Mix', 'Spearman_rho_vs_ERA5_Mix', 'Proporsi_Dibawah_24C_Pct']].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

