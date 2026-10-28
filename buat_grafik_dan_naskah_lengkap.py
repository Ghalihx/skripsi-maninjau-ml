import re
import shutil
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / 'Gambar_Skripsi'
FIG_DIR.mkdir(exist_ok=True)

def resolve_latest_source_file(base_dir: Path, stem: str, suffix: str, required_markers: list, explicit_name: str = None) -> Path:
    """
    Menemukan file sumber terbaru secara deterministik:
    1. Jika terdapat varian bernomor hasil unggahan ulang (misal (5) vs (4)),
       pengurutan mendahulukan: (a) lolos verifikasi seluruh required_markers,
       (b) nomor versi tertinggi (ver_num), baru kemudian (c) waktu modifikasi (mtime),
       sehingga file versi (4) yang disalin belakangan TIDAK mengalahkan versi (5).
    2. Jika tidak ada varian bernomor, menggunakan nama file kanonik eksplisit (explicit_name).
    """
    candidates = []
    pattern = re.compile(rf"^{re.escape(stem)}(?:\s*\((\d+)\))?{re.escape(suffix)}$", re.IGNORECASE)
    for p in base_dir.iterdir():
        if p.is_file():
            m = pattern.match(p.name)
            if m:
                ver_num = int(m.group(1)) if m.group(1) is not None else 0
                txt = p.read_text(encoding='utf-8', errors='ignore')
                has_all_markers = all(marker in txt for marker in required_markers)
                # Urutan tuple: (has_all_markers, ver_num, mtime, path)
                candidates.append((has_all_markers, ver_num, p.stat().st_mtime, p))
    if not candidates and explicit_name:
        explicit_path = base_dir / explicit_name
        if explicit_path.is_file():
            txt = explicit_path.read_text(encoding='utf-8', errors='ignore')
            has_all_markers = all(marker in txt for marker in required_markers)
            candidates.append((has_all_markers, 0, explicit_path.stat().st_mtime, explicit_path))
    if not candidates:
        raise FileNotFoundError(f"Tidak ditemukan file sumber {stem}{suffix} di {base_dir}")
    # Urutkan berdasarkan: (1) lolos verifikasi marker revisi terbaru, (2) nomor versi tertinggi (ver_num), (3) mtime
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    best_has_markers, best_ver, _, best_path = candidates[0]
    if not best_has_markers:
        missing = [m for m in required_markers if m not in best_path.read_text(encoding='utf-8', errors='ignore')]
        raise RuntimeError(f"File {best_path.name} belum memuat penanda revisi final: {missing}")
    print(f"[VERIFIKASI SUMBER] Menggunakan file terverifikasi: {best_path.name} (versi={best_ver})")
    return best_path

def save_dual_figure(fig, filename: str):
    """
    Menyimpan grafik 300 DPI ke dalam sub-folder Gambar_Skripsi/ DAN menyalinnya
    ke folder utama (BASE_DIR) agar file PNG langsung tersedia saat diperiksa.
    """
    target_sub = FIG_DIR / filename
    target_root = BASE_DIR / filename
    fig.savefig(target_sub, dpi=300, bbox_inches='tight')
    shutil.copyfile(target_sub, target_root)

# Set publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Load verified CSVs using resolve_latest_source_file so numbered revisions are respected
inv_path = resolve_latest_source_file(
    BASE_DIR, 'inventaris_upwelling_maninjau_2016_2025', '.csv',
    required_markers=['130.0', '"Kematian ikan dimulai sore hari 11 Februari 2022'],
    explicit_name='inventaris_upwelling_maninjau_2016_2025.csv'
)
upw_path = resolve_latest_source_file(
    BASE_DIR, 'hasil_evaluasi_eksploratif_upwelling', '.csv',
    required_markers=['130.0'],
    explicit_name='hasil_evaluasi_eksploratif_upwelling.csv'
)
inv_df = pd.read_csv(inv_path)
upw_eval = pd.read_csv(upw_path)
assert inv_df.shape == (15, 15), f"Format CSV inventaris tidak sesuai: {inv_df.shape}"
assert (inv_df['fish_loss_tons'].values == upw_eval['fish_loss_tons'].values).all(), "Kolom fish_loss_tons antara inventaris dan hasil evaluasi belum sinkron!"
print(f"[VERIFIKASI CSV] inventaris ({inv_df.shape[0]} baris, {inv_df.shape[1]} kolom) & evaluasi upwelling sinkron 15/15 catatan.")

modis_sens = pd.read_csv(BASE_DIR / 'evaluasi_khusus_era5_vs_modis_terra.csv')
test_res = pd.read_csv(BASE_DIR / 'hasil_evaluasi_test_2024_2025.csv')
feat_imp = pd.read_csv(BASE_DIR / 'hasil_feature_importance.csv')
pred_t1 = pd.read_csv(BASE_DIR / 'prediksi_uji_independen_t1_2024_2025.csv', parse_dates=['date', 'target_date'])
pred_t3 = pd.read_csv(BASE_DIR / 'prediksi_uji_independen_t3_2024_2025.csv', parse_dates=['date', 'target_date'])

# ==============================================================================
# GRAFIK 1 (Gambar 2 pada Naskah): AUDIT SENSITIVITAS ERA5 vs MODIS TERRA SIANG
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), dpi=300)

sel_cats = [
    '1_Ambang_Piksel >=  1/22',
    '1_Ambang_Piksel >= 10/22',
    '1_Ambang_Piksel >= 15/22',
    '1_Ambang_Piksel >= 22/22',
    '2_Fokus_15Piksel + Homogen_Spasial (std<=0.5C)',
    '4_Fokus_15Piksel + Homogen + Near_Nadir'
]
labels_g1 = [
    '≥1 piksel\n(N=922)',
    '≥10 piksel\n(N=735)',
    '≥15 piksel\n(N=641)',
    '22/22 piksel\n(N=435)',
    '≥15 piksel +\nstd ≤ 0,5°C\n(N=275)',
    '≥15 piksel +\nstd ≤ 0,5°C +\n|angle| ≤ 30°\n(N=117)'
]
sub_g1 = modis_sens.set_index('Kategori_Sensitivitas').loc[sel_cats].reset_index()

x = np.arange(len(labels_g1))
width = 0.36

# Panel (a): MAE dan |Bias Dingin| (°C)
bars1 = axes[0].bar(x - width/2, sub_g1['MAE_vs_ERA5_Mix_C'], width, label='MAE terhadap ERA5 (°C)', color='#2b5c8f', alpha=0.92)
bars2 = axes[0].bar(x + width/2, sub_g1['Bias_Terra_minus_ERA5_Mix_C'].abs(), width, label='|Bias Dingin| terhadap ERA5 (°C)', color='#d95f02', alpha=0.88)

axes[0].set_title('(a) Penurunan Galat (MAE) dan |Bias Dingin| (°C)', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Galat / Selisih Suhu (°C)', fontsize=9.8, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels_g1, fontsize=8.3)
axes[0].set_ylim(0, 2.85)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0].legend(loc='upper right', frameon=True, fontsize=8.5)

for b in bars1:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.04, f"{b.get_height():.2f}°C", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#1d3f63')
for b in bars2:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.04, f"{b.get_height():.2f}°C", ha='center', va='bottom', fontsize=7.8, color='#993d00')

# Panel (b): Korelasi Pearson (r) dan Spearman (rho)
bars_r = axes[1].bar(x - width/2, sub_g1['Pearson_r_vs_ERA5_Mix'], width, label='Korelasi Pearson (r)', color='#1b9e77', alpha=0.92)
bars_rho = axes[1].bar(x + width/2, sub_g1['Spearman_rho_vs_ERA5_Mix'], width, label='Korelasi Spearman (ρ)', color='#7570b3', alpha=0.88)

axes[1].set_title('(b) Korelasi Harian terhadap ERA5-Land (r dan ρ)', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Koefisien Korelasi', fontsize=9.8, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels_g1, fontsize=8.3)
axes[1].set_ylim(0, 0.40)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
axes[1].legend(loc='upper left', frameon=True, fontsize=8.5)

for b in bars_r:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.006, f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=7.8, fontweight='bold', color='#0f6e52')
for b in bars_rho:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.006, f"{b.get_height():.3f}", ha='center', va='bottom', fontsize=7.8, color='#4a448a')

plt.tight_layout()
save_dual_figure(fig, 'Gambar_1_Audit_ERA5_vs_MODIS_Terra.png')
plt.close()

# ==============================================================================
# GRAFIK 2 (Gambar 3 pada Naskah): KOMPARASI RMSE MODEL PADA DATA UJI INDEPENDEN
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=300)

# Panel A: Skema Utama (73 Fitur: Termal + Meteo <= t)
u_t1 = test_res[(test_res['Horizon'] == 't+1') & (test_res['Rezim_Fitur'].str.startswith('1_Utama'))]
u_t3 = test_res[(test_res['Horizon'] == 't+3') & (test_res['Rezim_Fitur'].str.startswith('1_Utama'))]

model_names_short = ['Persistence\n(Suhu T_t)', 'Persistence\n(Anomali A_t)', 'Random\nForest', 'SVR\n(RBF)', 'XGBoost']
order_models = [
    'Baseline Persistence (Suhu Mentah: T_t+h = T_t)',
    'Baseline Persistence (Anomali: A_t+h = A_t)',
    'Random Forest',
    'SVR',
    'XGBoost'
]
u_t1 = u_t1.set_index('Model').loc[order_models].reset_index()
u_t3 = u_t3.set_index('Model').loc[order_models].reset_index()

x_m = np.arange(len(model_names_short))
w = 0.36
b_t1 = axes[0].bar(x_m - w/2, u_t1['Test_Anom_RMSE'], w, label='Horizon t+1 (1 Hari)', color='#2b5c8f')
b_t3 = axes[0].bar(x_m + w/2, u_t3['Test_Anom_RMSE'], w, label='Horizon t+3 (3 Hari)', color='#e66101')
axes[0].set_title('(a) Skema Utama (73 Fitur: Termal Historis + Meteorologi ≤ t)', fontsize=10, fontweight='bold')
axes[0].set_ylabel('RMSE Anomali & Suhu pada Data Uji 2024–2025 (°C)', fontsize=9.5, fontweight='bold')
axes[0].set_xticks(x_m)
axes[0].set_xticklabels(model_names_short, fontsize=8.8)
axes[0].set_ylim(0, 0.26)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
axes[0].legend(fontsize=8.5)

for b in b_t1:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{b.get_height():.4f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for b in b_t3:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.004, f"{b.get_height():.4f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

# Panel B: Skema Ablasi Murni Meteorologi (55 Fitur Tanpa Suhu Danau)
a_t1 = test_res[(test_res['Horizon'] == 't+1') & (test_res['Rezim_Fitur'].str.startswith('2_Ablasi'))].set_index('Model').loc[['Random Forest', 'SVR', 'XGBoost']].reset_index()
a_t3 = test_res[(test_res['Horizon'] == 't+3') & (test_res['Rezim_Fitur'].str.startswith('2_Ablasi'))].set_index('Model').loc[['Random Forest', 'SVR', 'XGBoost']].reset_index()

x_a = np.arange(3)
ba_t1 = axes[1].bar(x_a - w/2, a_t1['Test_Anom_RMSE'], w, label='Horizon t+1 (55 Fitur Meteo)', color='#386cb0')
ba_t3 = axes[1].bar(x_a + w/2, a_t3['Test_Anom_RMSE'], w, label='Horizon t+3 (55 Fitur Meteo)', color='#f0027f')
axes[1].set_title('(b) Skema Ablasi (55 Fitur Murni Meteorologi ≤ t Tanpa Suhu Danau)', fontsize=10, fontweight='bold')
axes[1].set_ylabel('RMSE Anomali & Suhu pada Data Uji 2024–2025 (°C)', fontsize=9.5, fontweight='bold')
axes[1].set_xticks(x_a)
axes[1].set_xticklabels(['Random Forest\n(Meteo)', 'SVR (RBF)\n(Meteo)', 'XGBoost\n(Meteo)'], fontsize=9)
axes[1].set_ylim(0, 0.36)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
axes[1].legend(fontsize=8.5)

for b in ba_t1:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{b.get_height():.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
for b in ba_t3:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{b.get_height():.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
save_dual_figure(fig, 'Gambar_2_Komparasi_RMSE_Uji_2024_2025.png')
plt.close()

# ==============================================================================
# GRAFIK 3 (Gambar 4 pada Naskah): DERET WAKTU PREDIKSI DATA UJI INDEPENDEN
# ==============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.0), dpi=300, sharex=True)

# Panel A: Horizon t+1
axes[0].axhline(0, color='#666666', linestyle='--', linewidth=0.9)
axes[0].plot(pred_t1['target_date'], pred_t1['target_anom'], label='Target Referensi ERA5-Land (A_t+1)', color='black', linewidth=1.5, alpha=0.85)
axes[0].plot(pred_t1['target_date'], pred_t1['pred_anom_XGBoost'], label='Prediksi XGBoost (RMSE=0,1063°C, R²=0,9281)', color='#e41a1c', linewidth=1.2, alpha=0.9)
axes[0].plot(pred_t1['target_date'], pred_t1['pred_anom_SVR'], label='Prediksi SVR (RMSE=0,1215°C, R²=0,9061)', color='#377eb8', linewidth=1.0, alpha=0.75)
axes[0].axvspan(pd.to_datetime('2025-01-10'), pd.to_datetime('2025-01-14'), color='#feb24c', alpha=0.35, label='Kandidat REC-13 (13 Jan 2025: Anomali Positif +0,29°C)')
axes[0].axvspan(pd.to_datetime('2025-11-23'), pd.to_datetime('2025-12-20'), color='#9ecae1', alpha=0.45, label='Kandidat REC-14 & REC-15 (Nov–Des 2025: Anomali Dingin -1,11°C)')
axes[0].set_ylabel('Anomali Suhu (°C)', fontsize=10, fontweight='bold')
axes[0].set_title('(a) Prediksi Anomali Suhu Lapisan Tercampur ERA5-Land Horizon t+1 (73 Fitur, Periode Uji 2024–2025)', fontsize=10, fontweight='bold')
axes[0].legend(loc='lower left', ncol=2, fontsize=8.2, frameon=True)
axes[0].grid(True, linestyle=':', alpha=0.4)

# Panel B: Horizon t+3
axes[1].axhline(0, color='#666666', linestyle='--', linewidth=0.9)
axes[1].plot(pred_t3['target_date'], pred_t3['target_anom'], label='Target Referensi ERA5-Land (A_t+3)', color='black', linewidth=1.5, alpha=0.85)
axes[1].plot(pred_t3['target_date'], pred_t3['pred_anom_XGBoost'], label='Prediksi XGBoost (RMSE=0,1952°C, R²=0,7574)', color='#e41a1c', linewidth=1.2, alpha=0.9)
axes[1].plot(pred_t3['target_date'], pred_t3['pred_anom_SVR'], label='Prediksi SVR (MAE=0,1444°C, RMSE=0,1980°C)', color='#377eb8', linewidth=1.0, alpha=0.75)
axes[1].axvspan(pd.to_datetime('2025-01-10'), pd.to_datetime('2025-01-14'), color='#feb24c', alpha=0.35, label='Kandidat REC-13 (Jan 2025)')
axes[1].axvspan(pd.to_datetime('2025-11-23'), pd.to_datetime('2025-12-20'), color='#9ecae1', alpha=0.45, label='Kandidat REC-14 & REC-15 (Nov–Des 2025)')
axes[1].set_ylabel('Anomali Suhu (°C)', fontsize=10, fontweight='bold')
axes[1].set_xlabel('Tanggal Target (Periode Uji Independen 2024–2025)', fontsize=10, fontweight='bold')
axes[1].set_title('(b) Prediksi Anomali Suhu Lapisan Tercampur ERA5-Land Horizon t+3 (73 Fitur, Periode Uji 2024–2025)', fontsize=10, fontweight='bold')
axes[1].legend(loc='lower left', ncol=2, fontsize=8.2, frameon=True)
axes[1].grid(True, linestyle=':', alpha=0.4)
axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
save_dual_figure(fig, 'Gambar_3_Deret_Waktu_Uji_Independen_2024_2025.png')
plt.close()

# ==============================================================================
# GRAFIK 4 (Gambar 5 pada Naskah): FEATURE IMPORTANCE SKEMA UTAMA vs SKEMA ABLASI
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)

fi_u = feat_imp[(feat_imp['Horizon'] == 't+1') & (feat_imp['Rezim_Fitur'] == '1_Utama') & (feat_imp['Model'] == 'XGBoost')].sort_values('Importance', ascending=True).tail(8)
fi_a = feat_imp[(feat_imp['Horizon'] == 't+1') & (feat_imp['Rezim_Fitur'] == '2_Ablasi_Murni_Meteo') & (feat_imp['Model'] == 'XGBoost')].sort_values('Importance', ascending=True).tail(8)

axes[0].barh(fi_u['Fitur'], fi_u['Importance'], color='#2b5c8f')
axes[0].set_title('(a) Top-8 Fitur XGBoost — Skema Utama (73 Fitur, t+1)\n(Dominasi Memori Termal Hari Sebelumnya)', fontsize=9.8, fontweight='bold')
axes[0].set_xlabel('Kepentingan Relatif (Relative Gain Importance)', fontsize=9.5)
axes[0].set_xlim(0, 0.40)
axes[0].grid(axis='x', linestyle='--', alpha=0.4)
for idx, row in enumerate(fi_u.itertuples()):
    axes[0].text(row.Importance + 0.006, idx, f"{row.Importance:.4f}", va='center', fontsize=8.2, fontweight='bold')

axes[1].barh(fi_a['Fitur'], fi_a['Importance'], color='#d95f02')
axes[1].set_title('(b) Top-8 Fitur XGBoost — Skema Ablasi Meteo (55 Fitur, t+1)\n(Peran Rata-Rata Suhu Udara & Radiasi 7–14 Hari)', fontsize=9.8, fontweight='bold')
axes[1].set_xlabel('Kepentingan Relatif (Relative Gain Importance)', fontsize=9.5)
axes[1].set_xlim(0, 0.18)
axes[1].grid(axis='x', linestyle='--', alpha=0.4)
for idx, row in enumerate(fi_a.itertuples()):
    axes[1].text(row.Importance + 0.002, idx, f"{row.Importance:.4f}", va='center', fontsize=8.2, fontweight='bold')

plt.tight_layout()
save_dual_figure(fig, 'Gambar_4_Feature_Importance_XGBoost.png')
plt.close()

# ==============================================================================
# GRAFIK 5 (Gambar 6 pada Naskah): EVALUASI DESKRIPTIF CATATAN EXACT_DAY (N=10)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
ex = upw_eval[upw_eval['date_precision'] == 'Exact_Day'].copy().reset_index(drop=True)
x_e = np.arange(len(ex))
w_e = 0.28

b1 = ax.bar(x_e - w_e, ex['obs_event_day_anom_c'], w_e, label='Anomali ERA5 Hari Kejadian (t_0)', color='#1f78b4')
b2 = ax.bar(x_e, ex['obs_mean_anom_c'], w_e, label='Rata-Rata Anomali Jendela 5-Hari [t_0-3 .. t_0+1]', color='#33a02c')
b3 = ax.bar(x_e + w_e, ex['obs_min_anom_c'], w_e, label='Minimum Anomali Jendela 5-Hari [t_0-3 .. t_0+1]', color='#e31a1c')

ax.axhline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_xticks(x_e)
ax.set_xticklabels([f"{r.record_id}\n({r.episode_group.replace('EP-', '')})" for r in ex.itertuples()], fontsize=8.5)
ax.set_ylabel('Anomali Suhu Lapisan Tercampur ERA5-Land (°C)', fontsize=9.5, fontweight='bold')
ax.legend(fontsize=8.5, frameon=True)
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
save_dual_figure(fig, 'Gambar_5_Evaluasi_Deskriptif_Upwelling_ExactDay.png')
plt.close()

print("Berhasil membuat 5 gambar grafik resolusi tinggi (300 DPI) di:", FIG_DIR, "dan menyalinnya ke:", BASE_DIR)

# ==============================================================================
# PENGGABUNGAN NASKAH LENGKAP SKRIPSI (ABSTRAK + BAB I s/d BAB V + DAFTAR RUJUKAN)
# ==============================================================================
prop_path = resolve_latest_source_file(
    BASE_DIR,
    'PROPOSAL_SKRIPSI_MANINJAU',
    '.md',
    required_markers=['73 Fitur', '55 Fitur', '10.30536/ijreses.v19i2.13716', 'ANTARA. (2021a, 27 April)', 'ANTARA. (2025a, 19 Januari)', 'hujan-angin kencang', 'menunjukkan dominannya informasi riwayat suhu'],
    explicit_name='PROPOSAL_SKRIPSI_MANINJAU.md'
)
bab45_path = resolve_latest_source_file(
    BASE_DIR,
    'BAB_IV_DAN_V_SKRIPSI_MANINJAU',
    '.md',
    required_markers=['73 Fitur', '1,10', 'diduga berkaitan dengan perbedaan karakteristik fungsi kerugian', 'mendukung Hipotesis Pertama', 'menunjukkan dominannya informasi riwayat suhu'],
    explicit_name='BAB_IV_DAN_V_SKRIPSI_MANINJAU.md'
)

prop_text = prop_path.read_text(encoding='utf-8')
bab45_text = bab45_path.read_text(encoding='utf-8')

# Pisahkan Daftar Rujukan dari akhir PROPOSAL_SKRIPSI_MANINJAU.md agar berada di bagian paling akhir setelah BAB V
split_marker = '# DAFTAR RUJUKAN'
if split_marker in prop_text:
    bab123_part, ref_part = prop_text.split(split_marker, 1)
    ref_section = split_marker + ref_part
else:
    bab123_part = prop_text
    ref_section = ''

# Isi Abstrak disusun 1 alinea (1 paragraf), kalimat aktif, <= 250 kata, dan tepat 5 kata kunci
abstrak_body = (
    "Fluktuasi hidrometeorologi di kaldera Danau Maninjau berpotensi memicu anomali penurunan suhu lapisan "
    "permukaan yang berkaitan dengan pelemahan stratifikasi termal perairan, namun pemantauan harian berbasis "
    "satelit termal terhambat oleh tutupan awan tropis. Penelitian ini bertujuan mengevaluasi kesesuaian suhu "
    "reanalisis terhadap satelit serta membandingkan kinerja algoritma Random Forest, XGBoost, dan Support Vector "
    "Regression (SVR) terhadap dua model Baseline Persistence dalam memprediksi anomali suhu lapisan tercampur "
    "ERA5-Land pada horizon satu hari (t+1) dan tiga hari (t+3) berbasis Google Earth Engine periode 2016–2025 "
    "(3.653 hari). Metode penelitian mencakup audit spasial sel grid, analisis sensitivitas piksel MODIS dan "
    "Landsat, ekstraksi 73 fitur skema utama (termal historis dan meteorologi) serta 55 fitur skema ablasi murni "
    "meteorologi dengan indeks kalender 365 hari serta pengendalian kebocoran informasi masa depan melalui isolasi "
    "klimatologi dan pemisahan kronologis target, seleksi parameter melalui 3-Fold "
    "Rolling-Origin Cross-Validation (2016–2023), dan pengujian independen (2024–2025). Hasil penelitian "
    "menunjukkan bahwa sel grid utama ERA5-Land mencakup 52,20% perairan permanen, sedangkan penyaringan ketat "
    "MODIS Terra siang menurunkan MAE menjadi 1,10°C tetapi hanya menyisakan 3,2% hari pengamatan (r = 0,298). "
    "Pada pengujian independen skema utama, ketiga algoritma mengungguli kedua Baseline Persistence, di mana "
    "XGBoost menjadi model terbaik berdasarkan kriteria utama RMSE dan R² baik pada t+1 (RMSE = 0,1063°C; "
    "MAE = 0,0794°C; R² = 0,9281) maupun t+3 (RMSE = 0,1952°C; R² = 0,7574), sementara SVR mencatat MAE "
    "terendah pada t+3 (0,1444°C) dan unggul pada skema ablasi t+3 (R² = 0,5615). Evaluasi eksploratif "
    "menunjukkan bahwa anomali suhu ERA5-Land belum dapat dijadikan indikator tunggal kejadian upwelling."
)

word_count = len(abstrak_body.split())
assert word_count <= 250, f"Jumlah kata abstrak ({word_count}) melebihi batas 250 kata!"
print(f"Verifikasi Abstrak: 1 alinea, jumlah kata isi abstrak = {word_count} kata (batas <= 250 kata), 5 kata kunci.")

abstrak_section = f"""# NASKAH LENGKAP SKRIPSI (BAB I – BAB V)

**JUDUL SKRIPSI:**
**KOMPARASI ALGORITMA RANDOM FOREST, XGBOOST, DAN SUPPORT VECTOR REGRESSION UNTUK PREDIKSI ANOMALI SUHU LAPISAN TERCAMPUR REANALISIS ERA5-LAND DI KAWASAN DANAU MANINJAU BERBASIS GOOGLE EARTH ENGINE**

**Skema Penelitian:** Pengembangan Algoritma / Model / Analisis (Skema 1)
**Program Studi:** Ilmu Komputer (S1) — Fakultas Ilmu Komputer, Universitas Djuanda (2026)

---

## ABSTRAK

**Judul Skripsi:** Komparasi Algoritma Random Forest, XGBoost, dan Support Vector Regression untuk Prediksi Anomali Suhu Lapisan Tercampur Reanalisis ERA5-Land di Kawasan Danau Maninjau Berbasis Google Earth Engine
**Program Studi / Tahun:** Ilmu Komputer (S1), Fakultas Ilmu Komputer, Universitas Djuanda / 2026

{abstrak_body}

**Kata Kunci:** *Danau Maninjau, ERA5-Land, Google Earth Engine, Support Vector Regression, XGBoost*

---

## DAFTAR TABEL DAN DAFTAR GAMBAR

### Daftar Tabel
* **Tabel 1.** Ringkasan Penelitian Terdahulu (2022–2026)
* **Tabel 2.** Tahapan Metodologi Penelitian Pengembangan Model (Skema 1)
* **Tabel 3.** Rincian Eksak Konstruksi Fitur Input pada Skema Utama (73 Fitur) dan Skema Ablasi Murni Meteorologi (55 Fitur)
* **Tabel 4.** Jadwal Pelaksanaan Penelitian Skripsi
* **Tabel 5.** Ringkasan Ketersediaan dan Statistik Suhu Multi-Sumber di GEE (2016–2025)
* **Tabel 6.** Karakteristik Spasial dan Parameter Termal Rata-Rata (2016–2025) pada Empat Sel Grid ERA5-Land di Sekitar Danau Maninjau
* **Tabel 7.** Analisis Sensitivitas Kesesuaian Suhu Harian ERA5-Land (`era5_lake_mix_temp_c`) terhadap MODIS Siang dan Landsat 8/9 (2016–2025)
* **Tabel 8.** Konfigurasi Hyperparameter Terbaik Berdasarkan 3-Fold Rolling-Origin CV (2016–2023)
* **Tabel 9.** Perbandingan Kinerja Model pada Data Uji Independen (2024–2025) terhadap Target Reanalisis ERA5-Land
* **Tabel 10.** Lima Fitur dengan Kontribusi Relatif Tertinggi pada Skema Utama (73 Fitur) dan Skema Ablasi Meteorologi (55 Fitur)
* **Tabel 11.** Perilaku Deskriptif Anomali Termal Reanalisis ERA5-Land dan Prediksi XGBoost pada Catatan Harian Spesifik (`Exact_Day`, Jendela $H-3$ s/d $H+1$)
* **Tabel 12.** Perilaku Deskriptif Anomali Termal pada Episode Multi-Hari ($N=3$) dan Catatan Bulanan ($N=2$)

### Daftar Gambar
* **Gambar 1.** Bagan kerangka pemikiran penelitian komparasi algoritma regresi dan audit spasial-observasional di Danau Maninjau
* **Gambar 2.** Evaluasi kesesuaian suhu harian ERA5-Land terhadap MODIS Terra siang berdasarkan kriteria kualitas cakupan piksel dan geometri sensor (2016–2025)
* **Gambar 3.** Perbandingan RMSE model pada data uji independen (2024–2025) terhadap target reanalisis ERA5-Land untuk Skema Utama (73 fitur) dan Skema Ablasi (55 fitur)
* **Gambar 4.** Deret waktu anomali suhu referensi ERA5-Land terhadap prediksi model pada data uji independen (2024–2025) untuk horizon t+1 dan t+3
* **Gambar 5.** Perbandingan kontribusi fitur relatif (feature importance) XGBoost pada Skema Utama (73 fitur) dan Skema Ablasi Murni Meteorologi (55 fitur) untuk horizon t+1
* **Gambar 6.** Distribusi deskriptif anomali suhu reanalisis ERA5-Land pada 10 catatan harian spesifik (Exact_Day) periode 2016–2025

---
"""

# Bersihkan header duplikat di bab123_part
bab1_start = bab123_part.find('# BAB I — PENDAHULUAN')
bab123_clean = bab123_part[bab1_start:] if bab1_start != -1 else bab123_part

lampiran_section = """# LAMPIRAN PENDUKUNG (SKEMA 1: PENGEMBANGAN ALGORITMA / MODEL)

## Lampiran I. Daftar Artefak Dataset dan Kode Program Penelitian
1. `gee_extract_maninjau.js` — Skrip ekstraksi deret waktu harian *ERA5-Land*, *CHIRPS*, *MODIS Terra/Aqua*, dan *Landsat 8/9* di *Google Earth Engine*.
2. `gee_audit_grid_dan_modis.js` — Skrip audit geometri poligon sel grid *ERA5-Land*, fraksi air permanen `JRC Global Surface Water` (30 m), serta atribut kualitas piksel *MODIS* (`valid_pixels`, `std_c`, `view_time_hr`, `view_angle_deg`).
3. `audit_dan_eksperimen_ml.py` — Skrip utama pemodelan *machine learning* dengan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi dan pemisahan kronologis target (indeks kalender konsisten 365 hari, konstruksi 73 fitur Skema Utama dan 55 fitur Skema Ablasi, seleksi *hyperparameter* *3-Fold Rolling-Origin Cross-Validation*, pengujian independen terhadap dua *Baseline Persistence*, dan evaluasi eksploratif catatan *upwelling*).
4. `analisis_audit_lanjutan_gee.py` — Skrip analisis sensitivitas kesesuaian suhu *ERA5-Land* terhadap *MODIS Terra/Aqua* dan *Landsat 8/9*.
5. `buat_grafik_dan_naskah_lengkap.py` — Skrip pembangkit grafik publikasi 300 DPI (`Gambar_Skripsi/` dan folder utama) serta penggabungan naskah lengkap dengan verifikasi otomatis file sumber terbaru.

---

## Lampiran II. Dokumentasi Metadata Bibliografi Pustaka Jurnal dan Sumber Berita Sekunder
Tabel berikut mendokumentasikan **metadata bibliografi** (penulis, nama jurnal/penerbit, volume/nomor, halaman/nomor artikel, serta kode DOI laman penerbit) untuk 11 artikel jurnal ilmiah primer yang dirujuk dalam naskah ini:

| No | Sitasi dalam Teks | Penulis Lengkap | Jurnal & Penerbit | Vol(No), Halaman / Nomor Artikel | DOI Laman Penerbit & Kanal Publikasi |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | Aldiansyah dan Saputra (2022) | Septianto Aldiansyah, Randi Adrian Saputra | *International Journal of Remote Sensing and Earth Sciences (IJReSES)* (Penerbit BRIN) | Vol. 19(2), hal. 197–210 *(Edisi 2022)* | DOI Penerbit BRIN: `10.30536/ijreses.v19i2.13716` *(Rekaman Garuda memuat DOI lama `10.30536/j.ijreses.2022.v19.a3803`)* — Jurnal Nasional (BRIN) |
| 2 | Fajri dkk. (2026) | Muhammad Fajri, Muhammad Rama Kencana, Wendi Marsa Putra, Muhammad Thoriq | *Jurnal Nasional Teknologi dan Sistem Informasi (TEKNOSI)* (Universitas Andalas) | Vol. 12(2), hal. 276–283 | DOI: `10.25077/TEKNOSI.v12i2.2026.276-283` — Jurnal Nasional (Universitas Andalas) |
| 3 | Hamze-Ziabari dkk. (2023) | Seyed Mahmood Hamze-Ziabari, Ulrich Lemmin, Mehrshad Foroughan, Rafael S. Reiss, D. Andrew Barry | *Journal of Geophysical Research: Oceans* (Wiley / AGU) | Vol. 128(7), e2022JC019592 | DOI: `10.1029/2022JC019592` — Jurnal Internasional (Wiley / AGU) |
| 4 | Komala dkk. (2024) | Puti Sri Komala, Zulkarnaini Zulkarnaini, Roselyn Indah Kurniati, Mhd Fauzi | *Journal of Water and Land Development* (Polish Academy of Sciences) | No. 60, hal. 71–78 | DOI: `10.24425/jwld.2024.149109` — Jurnal Internasional (Polish Academy of Sciences) |
| 5 | Li dkk. (2024) | Zhenghao Li, Zhijie Zhang, Shengqing Xiong, Wanchang Zhang, Rui Li | *Remote Sensing* (MDPI) | Vol. 16(17), 3220 (hal. 1–22) | DOI: `10.3390/rs16173220` — Jurnal Internasional (MDPI) |
| 6 | Naumenko dkk. (2025) | Mikhail A. Naumenko, V. V. Guzivaty, N. M. Budnev, Elena S. Troitskaya | *Izvestiya, Atmospheric and Oceanic Physics* (Pleiades Publishing / Springer) | Vol. 61(11), hal. 1446–1454 *(Volume tahun 2025; publikasi daring Springer Mei 2026)* | DOI: `10.1134/S0001433826700210` *(Edisi terjemahan resmi dari Issledovanie Zemli iz Kosmosa 2025 No. 6 hal. 54–65)* — Jurnal Internasional (Springer) |
| 7 | Putri dkk. (2024) | Mutiara Rachmat Putri, Taofik Jasalesmana, Mirzam Abdurrachman, Cynthia Henny, Sulung Nomosatryo, Alif Shidqie Albani | *LIMNOTEK: Perairan Darat Tropis di Indonesia* (Penerbit BRIN) | Vol. 30(1), hal. 1–11 | DOI: `10.55981/limnotek.2024.2203` — Jurnal Nasional (BRIN) |
| 8 | Sunaryani (2023) | Astried Sunaryani | *Jurnal Teknologi Lingkungan* (Penerbit BRIN) | Vol. 24(1), hal. 21–27 | DOI: `10.55981/jtl.2023.246` — Jurnal Nasional (BRIN) |
| 9 | Syakir dkk. (2025) | Fakhrus Syakir, Muhammad Irhamsyah, Melinda, Yunidar, Zulhelmi, Rizka Miftahujjannah | *JITK (Jurnal Ilmu Pengetahuan dan Teknologi Komputer)* (Universitas Nusa Mandiri) | Vol. 11(2), hal. 580–590 | DOI: `10.33480/jitk.v11i2.6665` — Jurnal Nasional (Universitas Nusa Mandiri) |
| 10 | Ulhaq dkk. (2025) | M. Z. Ulhaq, M. Farid, Z. I. Aziza, T. M. F. Nuzullah, F. Syakir, N. R. Sasmita | *Theoretical and Applied Climatology* (Springer) | Vol. 156(9), 463 | DOI: `10.1007/s00704-025-05717-3` — Jurnal Internasional (Springer) |
| 11 | Valbuena dkk. (2022) | Sergio A. Valbuena, Fabián A. Bombardelli, Alicia Cortés, John L. Largier, Derek C. Roberts, Alexander L. Forrest, S. Geoffrey Schladow | *Water Resources Research* (Wiley / AGU) | Vol. 58(3), e2021WR030666 | DOI: `10.1029/2021WR030666` — Jurnal Internasional (Wiley / AGU) |

**Pemisahan Ruang Lingkup Verifikasi Metadata dan Isi Kutipan:**
1. **Ruang Lingkup Verifikasi Metadata Bibliografi:** Pemeriksaan pada tabel di atas mencakup keberadaan publikasi, kesesuaian nama penulis, judul resmi pada laman penerbit, nama jurnal, volume/nomor, halaman atau nomor artikel, serta kode DOI berdasarkan laman penerbit resmi dan/atau portal Garuda (termasuk pencatatan perbedaan DOI lama di Garuda vs portal BRIN untuk Aldiansyah & Saputra [2022] serta tahun volume vs tanggal terbit daring untuk Naumenko dkk. [2025]).
2. **Batasan terhadap Isi Kutipan Teks Lengkap dan Status Indeksasi:** Verifikasi metadata bibliografi di atas **terpisah dari pemeriksaan isi teks lengkap (*full-text PDF*) halaman demi halaman**. Oleh sebab itu, rujukan terhadap artikel jurnal di dalam BAB I hingga BAB IV dibatasi pada parafrasa konseptual umum `(Penulis, Tahun)` yang selaras dengan ringkasan/abstrak resmi publikasi tanpa mengklaim kutipan nomor halaman spesifik, sedangkan seluruh angka hasil pemodelan dan evaluasi dalam naskah ini berasal dari pengolahan data GEE dan eksperimen penulis. Selain itu, tabel ini tidak mengklaim peringkat akreditasi SINTA atau kuartil Scopus tertentu tanpa lampiran sertifikat/rekaman indeksasi historis pada tahun terbit masing-masing artikel.
3. **Ketertelusuran Berita Sekunder LKBN ANTARA:** Seluruh angka laporan kematian ikan KJA periode 2021–2025 yang dirujuk pada BAB I (`ANTARA, 2021a, 2021b, 2022, 2023, 2025a, 2025b, 2025c`) telah dicantumkan satu per satu pada bagian B Daftar Rujukan lengkap dengan tanggal terbit, judul berita, institusi sumber (DKPP Kabupaten Agam / LKBN ANTARA), serta tautan URL spesifik yang selaras dengan berkas `inventaris_upwelling_maninjau_2016_2025.csv`.
"""

full_manuscript = (
    abstrak_section.strip() + "\n\n" +
    bab123_clean.strip() + "\n\n---\n\n" +
    bab45_text.strip() + "\n\n---\n\n" +
    ref_section.strip() + "\n\n---\n\n" +
    lampiran_section.strip() + "\n"
)

full_md_path = BASE_DIR / 'NASKAH_LENGKAP_SKRIPSI_MANINJAU_BAB_I_SAMPAI_V.md'
full_md_path.write_text(full_manuscript, encoding='utf-8')
print("Berhasil menyusun naskah lengkap Markdown (Abstrak + BAB I-V + Daftar Rujukan + Lampiran) di:", full_md_path)
