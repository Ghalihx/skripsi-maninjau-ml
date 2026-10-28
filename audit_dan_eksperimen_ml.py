import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor

# ==============================================================================
# 0. PORTABLE DIRECTORY SETUP & LEAP-YEAR CONSISTENT CALENDAR INDEX
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DAILY_CSV = BASE_DIR / 'maninjau_daily_gee_2016_2025.csv'
LANDSAT_CSV = BASE_DIR / 'maninjau_landsat_lst_2016_2025.csv'
UPWELLING_CSV = BASE_DIR / 'inventaris_upwelling_maninjau_2016_2025.csv'

daily_raw = pd.read_csv(DAILY_CSV)
landsat_raw = pd.read_csv(LANDSAT_CSV)
upwelling_raw = pd.read_csv(UPWELLING_CSV)

daily_raw['date'] = pd.to_datetime(daily_raw['date'])
daily_raw = daily_raw.sort_values('date').reset_index(drop=True)

landsat_raw['date'] = pd.to_datetime(landsat_raw['date'])
landsat_raw = landsat_raw.sort_values('date').reset_index(drop=True)

def compute_consistent_cal_doy_365(dates: pd.Series) -> pd.Series:
    """
    Menghasilkan indeks kalender 1..365 yang konsisten antara tahun kabisat
    dan non-kabisat berdasarkan (bulan, tanggal).
    Tanggal 29 Februari dipetakan ke indeks 59 (sama dengan 28 Februari),
    sehingga 1 Maret selalu bernilai 60 dan 31 Desember selalu bernilai 365.
    """
    ref_year = 2021  # Tahun non-kabisat sebagai acuan kalender 365 hari
    months = dates.dt.month
    days = dates.dt.day
    adj_days = np.where((months == 2) & (days == 29), 28, days)
    ref_dates = pd.to_datetime({'year': ref_year, 'month': months, 'day': adj_days})
    return ref_dates.dt.dayofyear

daily_raw['cal_doy_365'] = compute_consistent_cal_doy_365(daily_raw['date'])

print("=" * 88)
print("1. AUDIT REPRESENTASI FISIK ERA5-LAND & ANALISIS SENSITIVITAS SATELIT")
print("=" * 88)
print(f"Lokasi Folder Kerja : {BASE_DIR}")
print(f"Total Baris Harian  : {len(daily_raw)} hari ({daily_raw['date'].min().date()} s/d {daily_raw['date'].max().date()})")
print("Batasan Klaim Utama : Evaluasi model mengukur akurasi prediksi terhadap keluaran")
print("                      reanalisis ERA5-Land (era5_lake_mix_temp_c), BUKAN akurasi")
print("                      suhu air Danau Maninjau yang tervalidasi pengamatan langsung.")

# Internal ERA5-Land relationships
daily_raw['delta_lake_temp_1d'] = daily_raw['era5_lake_mix_temp_c'].diff(1)
print("\n[A. Karakteristik Internal ERA5-Land (2016-2025)]")
print(f"  - era5_lake_mix_temp_c : Mean={daily_raw['era5_lake_mix_temp_c'].mean():.2f} C, Std={daily_raw['era5_lake_mix_temp_c'].std():.2f} C, Min={daily_raw['era5_lake_mix_temp_c'].min():.2f} C, Max={daily_raw['era5_lake_mix_temp_c'].max():.2f} C")
print(f"  - era5_skin_temp_c     : Mean={daily_raw['era5_skin_temp_c'].mean():.2f} C, Std={daily_raw['era5_skin_temp_c'].std():.2f} C, Min={daily_raw['era5_skin_temp_c'].min():.2f} C, Max={daily_raw['era5_skin_temp_c'].max():.2f} C")
print(f"  - air_temp_mean_c      : Mean={daily_raw['air_temp_mean_c'].mean():.2f} C, Std={daily_raw['air_temp_mean_c'].std():.2f} C, Min={daily_raw['air_temp_mean_c'].min():.2f} C, Max={daily_raw['air_temp_mean_c'].max():.2f} C")
print(f"  - r(lake_mix, skin)    = {daily_raw['era5_lake_mix_temp_c'].corr(daily_raw['era5_skin_temp_c']):+.3f}")
print(f"  - r(lake_mix, air)     = {daily_raw['era5_lake_mix_temp_c'].corr(daily_raw['air_temp_mean_c']):+.3f}")
print(f"  - r(dT/dt, solar_rad)  = {daily_raw['delta_lake_temp_1d'].corr(daily_raw['solar_net_rad_mjm2']):+.3f}")
print(f"  - r(dT/dt, air_temp)   = {daily_raw['delta_lake_temp_1d'].corr(daily_raw['air_temp_mean_c']):+.3f}")
print(f"  - r(dT/dt, precip)     = {daily_raw['delta_lake_temp_1d'].corr(daily_raw['precip_era5_mm']):+.3f}")
print(f"  - r(dT/dt, wind_speed) = {daily_raw['delta_lake_temp_1d'].corr(daily_raw['wind_speed_10m']):+.3f}")

# Satellite Sensitivity Analysis with Overpass Time & Mean Bias
print("\n[B. Analisis Sensitivitas Kualitas Data Satelit (MODIS & Landsat vs ERA5-Land)]")
overpass_info = {
    'modis_terra_day_c': '~10:30 WIB (Siang)',
    'modis_aqua_day_c': '~13:30 WIB (Siang)',
    'modis_terra_night_c': '~22:30 WIB (Malam)',
    'modis_aqua_night_c': '~01:30 WIB (Malam)',
    'landsat_8_9_lst_c': '~10:00 WIB (Pagi)'
}
sat_audit_rows = []

for col in ['modis_terra_day_c', 'modis_aqua_day_c', 'modis_terra_night_c', 'modis_aqua_night_c']:
    sub = daily_raw[['date', 'era5_lake_mix_temp_c', 'era5_skin_temp_c', col]].dropna()
    q1, q3 = sub[col].quantile(0.25), sub[col].quantile(0.75)
    iqr = q3 - q1
    sub_iqr = sub[(sub[col] >= q1 - 1.5 * iqr) & (sub[col] <= q3 + 1.5 * iqr)]
    sub_lit = sub[(sub[col] >= 24.0) & (sub[col] <= 32.0)]
    
    for label, df_s in [
        ('1_Raw_GEE_QC', sub),
        ('2_Stat_NonOutlier_IQR', sub_iqr),
        ('3_Sensitivitas_Rentang_Fisik_24_32C', sub_lit)
    ]:
        r_mix = df_s[col].corr(df_s['era5_lake_mix_temp_c']) if len(df_s) > 2 else np.nan
        r_skin = df_s[col].corr(df_s['era5_skin_temp_c']) if len(df_s) > 2 else np.nan
        bias_mix = (df_s[col] - df_s['era5_lake_mix_temp_c']).mean() if len(df_s) > 0 else np.nan
        mae_mix = (df_s[col] - df_s['era5_lake_mix_temp_c']).abs().mean() if len(df_s) > 0 else np.nan
        sat_audit_rows.append({
            'Sensor': col,
            'Waktu_Lintas': overpass_info[col],
            'Skema_Filter': label,
            'N': len(df_s),
            'Mean_C': df_s[col].mean(),
            'Min_C': df_s[col].min(),
            'Max_C': df_s[col].max(),
            'Bias_vs_ERA5_LakeMix_C': bias_mix,
            'MAE_vs_ERA5_LakeMix_C': mae_mix,
            'r_vs_ERA5_LakeMix': r_mix,
            'r_vs_ERA5_Skin': r_skin
        })

landsat_merged = pd.merge(
    landsat_raw,
    daily_raw[['date', 'era5_lake_mix_temp_c', 'era5_skin_temp_c']],
    on='date',
    how='left'
)
ls_q1, ls_q3 = landsat_merged['landsat_lst_mean_c'].quantile(0.25), landsat_merged['landsat_lst_mean_c'].quantile(0.75)
ls_iqr = ls_q3 - ls_q1

ls_schemes = [
    ('1_Raw_GEE_QA (pixels>=100, ST_QA<3K)', landsat_merged),
    ('2_QA_Ketat (pixels>=5000, std<0.5C, cloud<30%)', landsat_merged[
        (landsat_merged['valid_water_pixels'] >= 5000) &
        (landsat_merged['landsat_lst_std_c'] < 0.5) &
        (landsat_merged['cloud_cover_scene'] < 30.0)
    ]),
    ('3_Stat_NonOutlier_IQR', landsat_merged[
        (landsat_merged['landsat_lst_mean_c'] >= ls_q1 - 1.5 * ls_iqr) &
        (landsat_merged['landsat_lst_mean_c'] <= ls_q3 + 1.5 * ls_iqr)
    ]),
    ('4_Sensitivitas_Rentang_Fisik_24_32C', landsat_merged[
        (landsat_merged['landsat_lst_mean_c'] >= 24.0) &
        (landsat_merged['landsat_lst_mean_c'] <= 32.0)
    ])
]

for label, df_s in ls_schemes:
    col = 'landsat_lst_mean_c'
    sat_audit_rows.append({
        'Sensor': 'landsat_8_9_lst_c',
        'Waktu_Lintas': overpass_info['landsat_8_9_lst_c'],
        'Skema_Filter': label,
        'N': len(df_s),
        'Mean_C': df_s[col].mean(),
        'Min_C': df_s[col].min(),
        'Max_C': df_s[col].max(),
        'Bias_vs_ERA5_LakeMix_C': (df_s[col] - df_s['era5_lake_mix_temp_c']).mean() if len(df_s) > 0 else np.nan,
        'MAE_vs_ERA5_LakeMix_C': (df_s[col] - df_s['era5_lake_mix_temp_c']).abs().mean() if len(df_s) > 0 else np.nan,
        'r_vs_ERA5_LakeMix': df_s[col].corr(df_s['era5_lake_mix_temp_c']) if len(df_s) > 2 else np.nan,
        'r_vs_ERA5_Skin': df_s[col].corr(df_s['era5_skin_temp_c']) if len(df_s) > 2 else np.nan
    })

sat_audit_df = pd.DataFrame(sat_audit_rows)
sat_audit_df.to_csv(BASE_DIR / 'hasil_audit_sensitivitas_satelit.csv', index=False)
print(sat_audit_df[['Sensor', 'Waktu_Lintas', 'Skema_Filter', 'N', 'Mean_C', 'Bias_vs_ERA5_LakeMix_C', 'MAE_vs_ERA5_LakeMix_C', 'r_vs_ERA5_LakeMix']].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ==============================================================================
# 2. LEAK-FREE CLIMATOLOGY (365-DAY CONSISTENT) & FEATURE PIPELINE BUILDER
# ==============================================================================
def build_leak_free_dataset(raw_df: pd.DataFrame, clim_end_date: str, horizon: int):
    """
    Membangun fitur (<= t) dan target (t + horizon) menggunakan indeks kalender
    konsisten 365 hari (cal_doy_365, tanpa pergeseran tahun kabisat) di mana
    klimatologi dihitung HANYA dari data dengan date <= clim_end_date.
    """
    df = raw_df.copy().sort_values('date').reset_index(drop=True)
    clim_cutoff = pd.to_datetime(clim_end_date)
    
    ref_df = df[df['date'] <= clim_cutoff]
    doy_means = ref_df.groupby('cal_doy_365')['era5_lake_mix_temp_c'].mean()
    doy_means = doy_means.reindex(range(1, 366)).interpolate(method='linear')
    doy_padded = pd.concat([doy_means.iloc[-7:], doy_means, doy_means.iloc[:7]])
    doy_clim_smooth = doy_padded.rolling(15, center=True, min_periods=1).mean().iloc[7:-7]
    clim_map = doy_clim_smooth.to_dict()
    
    df['clim_doy_c'] = df['cal_doy_365'].map(clim_map)
    df['anom_doy_c'] = df['era5_lake_mix_temp_c'] - df['clim_doy_c']
    
    df['sin_doy'] = np.sin(2 * np.pi * df['cal_doy_365'] / 365.0)
    df['cos_doy'] = np.cos(2 * np.pi * df['cal_doy_365'] / 365.0)
    
    meteo_base = [
        'air_temp_mean_c', 'air_temp_min_c', 'air_temp_max_c',
        'u_wind_10m_ms', 'v_wind_10m_ms', 'wind_speed_10m',
        'solar_net_rad_mjm2', 'thermal_net_rad_mjm2',
        'surface_pressure_hpa', 'precip_era5_mm', 'precip_chirps_mm'
    ]
    
    feature_cols_meteo = ['sin_doy', 'cos_doy'] + meteo_base.copy()
    for col in ['air_temp_mean_c', 'wind_speed_10m', 'solar_net_rad_mjm2', 'thermal_net_rad_mjm2', 'precip_era5_mm', 'precip_chirps_mm']:
        for lag in [1, 2, 3, 7]:
            fname = f"{col}_lag{lag}"
            df[fname] = df[col].shift(lag)
            feature_cols_meteo.append(fname)
        for win in [3, 7, 14]:
            if 'precip' in col:
                fname = f"{col}_sum{win}d"
                df[fname] = df[col].rolling(win).sum()
            else:
                fname = f"{col}_mean{win}d"
                df[fname] = df[col].rolling(win).mean()
            feature_cols_meteo.append(fname)
            
    ar_cols = ['era5_lake_mix_temp_c', 'anom_doy_c']
    for lag in [1, 2, 3, 7]:
        df[f'lake_temp_lag{lag}'] = df['era5_lake_mix_temp_c'].shift(lag)
        df[f'anom_doy_lag{lag}'] = df['anom_doy_c'].shift(lag)
        ar_cols.extend([f'lake_temp_lag{lag}', f'anom_doy_lag{lag}'])
        
    for win in [3, 7, 14]:
        df[f'lake_temp_mean{win}d'] = df['era5_lake_mix_temp_c'].rolling(win).mean()
        df[f'lake_temp_std{win}d'] = df['era5_lake_mix_temp_c'].rolling(win).std()
        ar_cols.extend([f'lake_temp_mean{win}d', f'lake_temp_std{win}d'])
        
    df['lake_temp_diff1d'] = df['era5_lake_mix_temp_c'] - df['era5_lake_mix_temp_c'].shift(1)
    df['lake_temp_diff3d'] = df['era5_lake_mix_temp_c'] - df['era5_lake_mix_temp_c'].shift(3)
    ar_cols.extend(['lake_temp_diff1d', 'lake_temp_diff3d'])
    
    feature_cols_full = ar_cols + feature_cols_meteo
    
    df['target_date'] = df['date'].shift(-horizon)
    df['target_temp'] = df['era5_lake_mix_temp_c'].shift(-horizon)
    df['target_anom'] = df['anom_doy_c'].shift(-horizon)
    df['target_clim'] = df['clim_doy_c'].shift(-horizon)
    
    df_clean = df.dropna(subset=feature_cols_full + ['target_date', 'target_temp', 'target_anom']).copy().reset_index(drop=True)
    return df_clean, feature_cols_full, feature_cols_meteo, clim_map

def compute_metrics(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(root_mean_squared_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred))
    }

# ==============================================================================
# 3. ROLLING-ORIGIN CROSS-VALIDATION (ROCV) + INDEPENDENT TEST (2024-2025)
# ==============================================================================
print("\n" + "=" * 88)
print("2 & 3. ROLLING-ORIGIN CROSS-VALIDATION (2016-2023) & UJI INDEPENDEN (2024-2025)")
print("=" * 88)

# Tiga fold Expanding-Window / Rolling-Origin Cross-Validation pada periode pra-uji (2016-2023):
# Fold 1: Train 2016-2019 (Clim <= 2019) -> Val 2020-2021
# Fold 2: Train 2016-2020 (Clim <= 2020) -> Val 2021-2022
# Fold 3: Train 2016-2021 (Clim <= 2021) -> Val 2022-2023
rocv_folds = [
    ('Fold1_Val2020_2021', '2019-12-31', '2020-01-01', '2021-12-31'),
    ('Fold2_Val2021_2022', '2020-12-31', '2021-01-01', '2022-12-31'),
    ('Fold3_Val2022_2023', '2021-12-31', '2022-01-01', '2023-12-31'),
]

candidate_grids = {
    'Random Forest': [
        {'n_estimators': 200, 'max_depth': 8,  'min_samples_leaf': 5,  'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 5,  'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 3,  'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 10, 'random_state': 42, 'n_jobs': -1},
    ],
    'XGBoost': [
        {'n_estimators': 200, 'learning_rate': 0.03, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1},
        {'n_estimators': 400, 'learning_rate': 0.02, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1},
    ],
    'SVR': [
        {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.002,   'epsilon': 0.01},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.002,   'epsilon': 0.01},
        {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.005,   'epsilon': 0.01},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'epsilon': 0.02},
    ]
}

def make_estimator(algo_name, params):
    if algo_name == 'Random Forest':
        return RandomForestRegressor(**params)
    elif algo_name == 'XGBoost':
        return XGBRegressor(**params)
    elif algo_name == 'SVR':
        return Pipeline([('scaler', StandardScaler()), ('svr', SVR(**params))])
    raise ValueError(algo_name)

tuning_records = []
test_records = []
feature_importance_records = []
val_predictions_by_h = {}
test_predictions_by_h = {}

for h in [1, 3]:
    print(f"\n>>> HORIZON PREDIKSI: t+{h} <<<")
    
    # Bangun dataset per fold dengan klimatologi yang terisolasi pada masing-masing batas akhir train
    fold_datasets = {}
    for fold_name, tr_end, va_start, va_end in rocv_folds:
        df_f, f_full, f_meteo, _ = build_leak_free_dataset(daily_raw, clim_end_date=tr_end, horizon=h)
        tr_sub = df_f[(df_f['date'] <= tr_end) & (df_f['target_date'] <= tr_end)].copy()
        va_sub = df_f[(df_f['date'] >= va_start) & (df_f['date'] <= va_end) & (df_f['target_date'] <= va_end)].copy()
        fold_datasets[fold_name] = (tr_sub, va_sub)
        print(f"  [ROCV {fold_name} | Clim <= {tr_end}] Train N={len(tr_sub)}, Val N={len(va_sub)}")
        
    # Siapkan dataframe penyimpanan untuk Fold 3 (Validasi 2022-2023)
    _, val_sub_22_23 = fold_datasets['Fold3_Val2022_2023']
    val_pred_df = val_sub_22_23[['date', 'target_date', 'era5_lake_mix_temp_c', 'anom_doy_c', 'target_temp', 'target_anom', 'target_clim']].copy()
    val_pred_df['split_regime'] = 'Validation_Fold3_2022_2023 (Used_for_Hyperparameter_Selection_Clim2016_2021)'
    val_pred_df['pred_anom_Persistence_Anom'] = val_pred_df['anom_doy_c']
    val_pred_df['pred_anom_Persistence_RawTemp'] = val_pred_df['era5_lake_mix_temp_c'] - val_pred_df['target_clim']
    
    best_params_full = {}
    best_params_meteo = {}
    
    for regime_label, feat_list, best_dict in [
        ('1_Utama (Termal + Meteo <= t)', f_full, best_params_full),
        ('2_Ablasi_Murni_Meteo (Tanpa Suhu Danau)', f_meteo, best_params_meteo)
    ]:
        print(f"  -- Rolling-Origin CV (3 Folds) untuk {regime_label} --")
        for algo_name, grid in candidate_grids.items():
            best_rocv_rmse = float('inf')
            best_cfg = None
            best_fold3_pred = None
            
            for idx_cfg, cfg in enumerate(grid, 1):
                fold_rmses = []
                fold_maes = []
                fold_r2s = []
                fold3_pred_curr = None
                
                for fold_name, _, _, _ in rocv_folds:
                    tr_sub, va_sub = fold_datasets[fold_name]
                    est = make_estimator(algo_name, cfg)
                    est.fit(tr_sub[feat_list].values, tr_sub['target_anom'].values)
                    va_pred = est.predict(va_sub[feat_list].values)
                    m_va = compute_metrics(va_sub['target_anom'].values, va_pred)
                    fold_rmses.append(m_va['RMSE'])
                    fold_maes.append(m_va['MAE'])
                    fold_r2s.append(m_va['R2'])
                    if fold_name == 'Fold3_Val2022_2023':
                        fold3_pred_curr = va_pred
                        
                mean_rmse = float(np.mean(fold_rmses))
                std_rmse = float(np.std(fold_rmses))
                mean_mae = float(np.mean(fold_maes))
                mean_r2 = float(np.mean(fold_r2s))
                
                tuning_records.append({
                    'Horizon': f't+{h}',
                    'Rezim_Fitur': regime_label,
                    'Algoritma': algo_name,
                    'Kandidat_ID': f'{algo_name}_cfg{idx_cfg}',
                    'Konfigurasi': str(cfg),
                    'ROCV_Mean_MAE': mean_mae,
                    'ROCV_Mean_RMSE': mean_rmse,
                    'ROCV_Std_RMSE': std_rmse,
                    'ROCV_Mean_R2': mean_r2,
                    'Fold1_2020_2021_RMSE': fold_rmses[0],
                    'Fold2_2021_2022_RMSE': fold_rmses[1],
                    'Fold3_2022_2023_RMSE': fold_rmses[2],
                })
                
                # Pilih konfigurasi terbaik berdasarkan rata-rata RMSE Rolling-Origin CV (ROCV_Mean_RMSE)
                if mean_rmse < best_rocv_rmse:
                    best_rocv_rmse = mean_rmse
                    best_cfg = cfg
                    best_fold3_pred = fold3_pred_curr
                    
            best_dict[algo_name] = best_cfg
            if regime_label.startswith('1_Utama'):
                val_pred_df[f'pred_anom_{algo_name}'] = best_fold3_pred
            else:
                val_pred_df[f'pred_anom_meteo_{algo_name}'] = best_fold3_pred
            print(f"     * Best {algo_name:13s} | ROCV Mean RMSE={best_rocv_rmse:.4f} C | Config={best_cfg}")
            
    val_predictions_by_h[h] = val_pred_df
    val_pred_df.to_csv(BASE_DIR / f'prediksi_validasi_t{h}_2022_2023.csv', index=False)
    
    # --------------------------------------------------------------------------
    # TAHAP B: UJI INDEPENDEN FINAL (Klimatologi 2016-2023, Target TrainVal <= 2023-12-31)
    # --------------------------------------------------------------------------
    df_test_stage, f_full, f_meteo, _ = build_leak_free_dataset(
        daily_raw, clim_end_date='2023-12-31', horizon=h
    )
    
    trainval_sub = df_test_stage[
        (df_test_stage['date'] <= '2023-12-31') &
        (df_test_stage['target_date'] <= '2023-12-31')
    ].copy()
    
    test_sub = df_test_stage[
        (df_test_stage['date'] >= '2024-01-01') &
        (df_test_stage['target_date'] <= '2025-12-31')
    ].copy()
    
    print(f"  [Tahap Uji Independen | Klimatologi 2016-2023] Train+Val N={len(trainval_sub)} (target s/d {trainval_sub['target_date'].max().date()}), Test N={len(test_sub)} (target s/d {test_sub['target_date'].max().date()})")
    
    y_te_anom = test_sub['target_anom'].values
    y_te_temp = test_sub['target_temp'].values
    clim_te = test_sub['target_clim'].values
    
    # 1a. Baseline Persistence Anomali: A_hat(t+h) = A(t)  <=>  T_hat(t+h) = A(t) + T_clim(t+h)
    pers_anom_pred_anom = test_sub['anom_doy_c'].values
    pers_anom_pred_temp = pers_anom_pred_anom + clim_te
    m_pa_anom = compute_metrics(y_te_anom, pers_anom_pred_anom)
    m_pa_temp = compute_metrics(y_te_temp, pers_anom_pred_temp)
    
    test_records.append({
        'Horizon': f't+{h}',
        'Rezim_Fitur': '1_Utama (Termal + Meteo <= t)',
        'Model': 'Baseline Persistence (Anomali: A_t+h = A_t)',
        'Konfigurasi_Terpilih_ROCV': 'A_hat(t+h) = A(t)',
        'Test_Anom_MAE': m_pa_anom['MAE'],
        'Test_Anom_RMSE': m_pa_anom['RMSE'],
        'Test_Anom_R2': m_pa_anom['R2'],
        'Test_Temp_MAE': m_pa_temp['MAE'],
        'Test_Temp_RMSE': m_pa_temp['RMSE'],
        'Test_Temp_R2': m_pa_temp['R2']
    })
    
    # 1b. Baseline Persistence Suhu Mentah: T_hat(t+h) = T(t)  <=>  A_hat(t+h) = T(t) - T_clim(t+h)
    pers_raw_pred_temp = test_sub['era5_lake_mix_temp_c'].values
    pers_raw_pred_anom = pers_raw_pred_temp - clim_te
    m_pr_anom = compute_metrics(y_te_anom, pers_raw_pred_anom)
    m_pr_temp = compute_metrics(y_te_temp, pers_raw_pred_temp)
    
    test_records.append({
        'Horizon': f't+{h}',
        'Rezim_Fitur': '1_Utama (Termal + Meteo <= t)',
        'Model': 'Baseline Persistence (Suhu Mentah: T_t+h = T_t)',
        'Konfigurasi_Terpilih_ROCV': 'T_hat(t+h) = T(t)',
        'Test_Anom_MAE': m_pr_anom['MAE'],
        'Test_Anom_RMSE': m_pr_anom['RMSE'],
        'Test_Anom_R2': m_pr_anom['R2'],
        'Test_Temp_MAE': m_pr_temp['MAE'],
        'Test_Temp_RMSE': m_pr_temp['RMSE'],
        'Test_Temp_R2': m_pr_temp['R2']
    })
    
    test_pred_df = test_sub[['date', 'target_date', 'era5_lake_mix_temp_c', 'anom_doy_c', 'target_temp', 'target_anom', 'target_clim']].copy()
    test_pred_df['split_regime'] = 'Independent_Out_of_Sample_Test_2024_2025 (Clim2016_2023)'
    test_pred_df['pred_anom_Persistence_Anom'] = pers_anom_pred_anom
    test_pred_df['pred_anom_Persistence_RawTemp'] = pers_raw_pred_anom
    
    for regime_label, feat_list, best_dict, col_prefix, short_reg in [
        ('1_Utama (Termal + Meteo <= t)', f_full, best_params_full, 'pred_anom_', '1_Utama'),
        ('2_Ablasi_Murni_Meteo (Tanpa Suhu Danau)', f_meteo, best_params_meteo, 'pred_anom_meteo_', '2_Ablasi_Murni_Meteo')
    ]:
        for algo_name, best_cfg in best_dict.items():
            est = make_estimator(algo_name, best_cfg)
            est.fit(trainval_sub[feat_list].values, trainval_sub['target_anom'].values)
            pred_anom = est.predict(test_sub[feat_list].values)
            pred_temp = pred_anom + clim_te
            
            m_anom = compute_metrics(y_te_anom, pred_anom)
            m_temp = compute_metrics(y_te_temp, pred_temp)
            
            test_records.append({
                'Horizon': f't+{h}',
                'Rezim_Fitur': regime_label,
                'Model': algo_name,
                'Konfigurasi_Terpilih_ROCV': str(best_cfg),
                'Test_Anom_MAE': m_anom['MAE'],
                'Test_Anom_RMSE': m_anom['RMSE'],
                'Test_Anom_R2': m_anom['R2'],
                'Test_Temp_MAE': m_temp['MAE'],
                'Test_Temp_RMSE': m_temp['RMSE'],
                'Test_Temp_R2': m_temp['R2']
            })
            test_pred_df[f'{col_prefix}{algo_name}'] = pred_anom
            
            if algo_name in ['Random Forest', 'XGBoost']:
                imp = pd.Series(est.feature_importances_, index=feat_list).sort_values(ascending=False)
                for rank_i, (fname, fval) in enumerate(imp.head(10).items(), 1):
                    feature_importance_records.append({
                        'Horizon': f't+{h}',
                        'Rezim_Fitur': short_reg,
                        'Model': algo_name,
                        'Rank': rank_i,
                        'Fitur': fname,
                        'Importance': float(fval)
                    })
                    
    test_predictions_by_h[h] = test_pred_df
    test_pred_df.to_csv(BASE_DIR / f'prediksi_uji_independen_t{h}_2024_2025.csv', index=False)

tuning_df = pd.DataFrame(tuning_records)
tuning_df.to_csv(BASE_DIR / 'hasil_tuning_validasi_2022_2023.csv', index=False)

test_results_df = pd.DataFrame(test_records)
test_results_df.to_csv(BASE_DIR / 'hasil_evaluasi_test_2024_2025.csv', index=False)

feat_imp_df = pd.DataFrame(feature_importance_records)
feat_imp_df.to_csv(BASE_DIR / 'hasil_feature_importance.csv', index=False)

print("\n[Ringkasan Hasil Evaluasi pada Data Uji Independen 2024-2025]")
print(test_results_df[['Horizon', 'Rezim_Fitur', 'Model', 'Test_Anom_MAE', 'Test_Anom_RMSE', 'Test_Anom_R2', 'Test_Temp_MAE', 'Test_Temp_RMSE', 'Test_Temp_R2']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ==============================================================================
# 4. ANALISIS DESKRIPTIF-EKSPLORATIF EPISODE UPWELLING (TANPA PENUGASAN GANDA)
# ==============================================================================
print("\n" + "=" * 88)
print("4. ANALISIS DESKRIPTIF-EKSPLORATIF KEJADIAN UPWELLING (DIPISAH PER PANJANG JENDELA)")
print("=" * 88)

df_obs_clim21, _, _, _ = build_leak_free_dataset(daily_raw, clim_end_date='2021-12-31', horizon=1)
df_obs_clim23, _, _, _ = build_leak_free_dataset(daily_raw, clim_end_date='2023-12-31', horizon=1)

val_t1 = val_predictions_by_h[1].set_index('target_date')
val_t3 = val_predictions_by_h[3].set_index('target_date')
test_t1 = test_predictions_by_h[1].set_index('target_date')
test_t3 = test_predictions_by_h[3].set_index('target_date')

upwelling_eval_rows = []

for _, row in upwelling_raw.iterrows():
    rec_id = row['record_id']
    ep = row['episode_group']
    prec = row['date_precision']
    s_date = pd.to_datetime(row['start_date'])
    e_date = pd.to_datetime(row['end_date'])
    rep_date = pd.to_datetime(row['reported_event_date']) if pd.notna(row['reported_event_date']) and str(row['reported_event_date']) != 'NA' else pd.NaT
    
    yr = s_date.year
    if yr <= 2021:
        eval_regime = '1_Train_Historical_ERA5_Only (Clim_2016_2021)'
        active_obs_df = df_obs_clim21
        pred_source_t1, pred_source_t3 = None, None
    elif yr <= 2023:
        eval_regime = '2_Validation_Set (Clim_2016_2021)'
        active_obs_df = df_obs_clim21
        pred_source_t1, pred_source_t3 = val_t1, val_t3
    else:
        eval_regime = '3_Independent_Test_Set (Clim_2016_2023)'
        active_obs_df = df_obs_clim23
        pred_source_t1, pred_source_t3 = test_t1, test_t3
        
    if prec == 'Exact_Day' and pd.notna(rep_date):
        w_start = rep_date - pd.Timedelta(days=3)
        w_end = rep_date + pd.Timedelta(days=1)
        window_rule = 'Jendela_Seragam_5Hari (H-3 s/d H+1)'
    elif prec in ['Multi_Day_Episode', 'Multi_Day_Observation']:
        w_start = s_date
        w_end = e_date
        window_rule = f'Episode_MultiHari ({(e_date - s_date).days + 1} hari)'
    else:
        w_start = s_date
        w_end = e_date
        window_rule = f'Agregat_Bulanan ({(e_date - s_date).days + 1} hari)'
        
    # Satu penugasan bersih tanpa duplikasi
    sub_obs = active_obs_df[
        (active_obs_df['date'] >= w_start) &
        (active_obs_df['date'] <= w_end)
    ]
    
    if prec == 'Exact_Day' and pd.notna(rep_date):
        day_match = active_obs_df[active_obs_df['date'] == rep_date]
        event_day_anom = float(day_match['anom_doy_c'].iloc[0]) if len(day_match) > 0 else np.nan
    else:
        event_day_anom = np.nan
        
    if pred_source_t1 is not None:
        valid_d1 = [d for d in sub_obs['date'] if d in pred_source_t1.index]
        xgb_min_t1 = float(pred_source_t1.loc[valid_d1, 'pred_anom_XGBoost'].min()) if valid_d1 else np.nan
        xgb_mean_t1 = float(pred_source_t1.loc[valid_d1, 'pred_anom_XGBoost'].mean()) if valid_d1 else np.nan
    else:
        xgb_min_t1, xgb_mean_t1 = np.nan, np.nan
        
    if pred_source_t3 is not None:
        valid_d3 = [d for d in sub_obs['date'] if d in pred_source_t3.index]
        xgb_min_t3 = float(pred_source_t3.loc[valid_d3, 'pred_anom_XGBoost'].min()) if valid_d3 else np.nan
        xgb_mean_t3 = float(pred_source_t3.loc[valid_d3, 'pred_anom_XGBoost'].mean()) if valid_d3 else np.nan
    else:
        xgb_min_t3, xgb_mean_t3 = np.nan, np.nan
        
    upwelling_eval_rows.append({
        'record_id': rec_id,
        'episode_group': ep,
        'date_precision': prec,
        'window_days': int(len(sub_obs)),
        'window_rule': window_rule,
        'eval_regime': eval_regime,
        'fish_loss_tons': row['fish_loss_tons'],
        'obs_min_temp_c': float(sub_obs['era5_lake_mix_temp_c'].min()),
        'obs_event_day_anom_c': event_day_anom,
        'obs_mean_anom_c': float(sub_obs['anom_doy_c'].mean()),
        'obs_min_anom_c': float(sub_obs['anom_doy_c'].min()),
        'obs_max_3d_drop_c': float(sub_obs['lake_temp_diff3d'].min()),
        'pred_xgb_mean_anom_t1_c': xgb_mean_t1,
        'pred_xgb_min_anom_t1_c': xgb_min_t1,
        'pred_xgb_mean_anom_t3_c': xgb_mean_t3,
        'pred_xgb_min_anom_t3_c': xgb_min_t3,
        'max_rain_era5_mm': float(sub_obs['precip_era5_mm'].max()),
        'min_solar_mjm2': float(sub_obs['solar_net_rad_mjm2'].min()),
        'max_wind_ms': float(sub_obs['wind_speed_10m'].max())
    })

upwelling_eval_df = pd.DataFrame(upwelling_eval_rows)
upwelling_eval_df.to_csv(BASE_DIR / 'hasil_evaluasi_eksploratif_upwelling.csv', index=False)

print("\n[A. Kelompok 1: Catatan Harian Spesifik (Exact_Day, N=10, Jendela Seragam 5 Hari)]")
exact_df = upwelling_eval_df[upwelling_eval_df['date_precision'] == 'Exact_Day']
print(exact_df[['record_id', 'window_days', 'eval_regime', 'obs_event_day_anom_c', 'obs_mean_anom_c', 'obs_min_anom_c', 'obs_max_3d_drop_c', 'pred_xgb_mean_anom_t1_c']].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n[B. Kelompok 2: Episode Multi-Hari (Multi_Day_Episode & Observation, N=3, Durasi Bervariasi)]")
multi_df = upwelling_eval_df[upwelling_eval_df['date_precision'].isin(['Multi_Day_Episode', 'Multi_Day_Observation'])]
print(multi_df[['record_id', 'window_days', 'eval_regime', 'obs_mean_anom_c', 'obs_min_anom_c', 'obs_max_3d_drop_c', 'pred_xgb_mean_anom_t1_c', 'pred_xgb_min_anom_t1_c']].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n[C. Kelompok 3: Catatan Presisi Bulanan Saja (Month_Only, N=2, Dikecualikan dari Evaluasi Harian)]")
month_df = upwelling_eval_df[upwelling_eval_df['date_precision'] == 'Month_Only']
print(month_df[['record_id', 'window_days', 'eval_regime', 'obs_mean_anom_c', 'obs_min_anom_c', 'obs_max_3d_drop_c']].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

print("\n[SELESAI] Seluruh file CSV hasil perbaikan telah diperbarui di:", BASE_DIR)
