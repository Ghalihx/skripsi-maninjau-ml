"""Reproduksi eksperimen skripsi dari CSV pengguna; kode asli tidak dimodifikasi."""
from pathlib import Path
import io
import zipfile
import subprocess
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parent
DAILY = 'maninjau_daily_gee_2016_2025.csv'
LANDSAT = 'maninjau_landsat_lst_2016_2025.csv'
EVENT = 'inventaris_upwelling_maninjau_2016_2025.csv'
REQUIRED_DAILY = {
    'date','era5_lake_mix_temp_c','era5_skin_temp_c',
    'air_temp_mean_c','air_temp_min_c','air_temp_max_c',
    'u_wind_10m_ms','v_wind_10m_ms','wind_speed_10m',
    'solar_net_rad_mjm2','thermal_net_rad_mjm2',
    'surface_pressure_hpa','precip_era5_mm','precip_chirps_mm',
    'modis_terra_day_c','modis_aqua_day_c','modis_terra_night_c','modis_aqua_night_c',
}

def validate_inputs(daily_bytes, landsat_bytes, events_bytes):
    daily = pd.read_csv(io.BytesIO(daily_bytes))
    landsat = pd.read_csv(io.BytesIO(landsat_bytes))
    events = pd.read_csv(io.BytesIO(events_bytes))
    missing = REQUIRED_DAILY - set(daily.columns)
    if missing:
        raise ValueError('Kolom dataset harian belum lengkap: '+', '.join(sorted(missing)))
    dates = pd.to_datetime(daily['date'], errors='coerce')
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError('Kolom date mengandung tanggal tidak valid atau duplikat.')
    if not dates.is_monotonic_increasing:
        raise ValueError('Urutkan dataset harian menurut date secara naik.')
    if len(daily) != 3653 or str(dates.iloc[0].date()) != '2016-01-01' or str(dates.iloc[-1].date()) != '2025-12-31':
        raise ValueError('Reproduksi skripsi memerlukan 3.653 hari pada 2016-01-01 hingga 2025-12-31, bukan sembarang rentang waktu.')
    if (dates.diff().iloc[1:].dt.days != 1).any():
        raise ValueError('Tanggal harian harus berurutan tanpa hari yang hilang.')
    if daily['era5_lake_mix_temp_c'].isna().any():
        raise ValueError('Target era5_lake_mix_temp_c tidak boleh memiliki nilai kosong.')
    for c in ['date','landsat_lst_mean_c']:
        if c not in landsat:
            raise ValueError('CSV Landsat tidak memiliki kolom '+c)
    expected_cols={'record_id','date_precision','start_date','end_date','fish_loss_tons','reported_event_date','episode_group'}
    if not expected_cols.issubset(events.columns):
        raise ValueError('Kolom CSV inventaris kurang: '+str(sorted(expected_cols-set(events.columns))))
    if len(events) != 15:
        raise ValueError('Penelitian ini menggunakan tepat 15 catatan historis; unggah inventaris yang konsisten.')
    if events['record_id'].duplicated().any():
        raise ValueError('record_id pada inventaris tidak boleh duplikat.')
    return daily, landsat, events

def execute(daily_bytes, landsat_bytes, events_bytes, timeout_s=3600):
    """Output ZIP di memori; setiap sesi menyimpan hasilnya di st.session_state saja."""
    validate_inputs(daily_bytes, landsat_bytes, events_bytes)
    with tempfile.TemporaryDirectory(prefix='maninjau_ml_') as temp:
        work=Path(temp)
        (work/DAILY).write_bytes(daily_bytes)
        (work/LANDSAT).write_bytes(landsat_bytes)
        (work/EVENT).write_bytes(events_bytes)
        (work/'audit_dan_eksperimen_ml.py').write_bytes((ROOT/'audit_dan_eksperimen_ml.py').read_bytes())
        try:
            proc = subprocess.run([sys.executable, '-u', 'audit_dan_eksperimen_ml.py'], cwd=str(work),
                                  capture_output=True,text=True,timeout=timeout_s,check=False)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError('Batas waktu pelatihan terlampaui. Jalankan versi lengkap di Colab/laptop jika server Streamlit membatasi waktu atau memori.') from e
        logs = (proc.stdout or '') + '\n' + (proc.stderr or '')
        if proc.returncode != 0:
            raise RuntimeError('Skrip eksperimen gagal:\n'+logs[-5000:])
        expected=['hasil_evaluasi_test_2024_2025.csv','hasil_tuning_validasi_2022_2023.csv',
                  'hasil_feature_importance.csv','prediksi_uji_independen_t1_2024_2025.csv',
                  'prediksi_uji_independen_t3_2024_2025.csv','hasil_evaluasi_eksploratif_upwelling.csv']
        missing=[f for f in expected if not (work/f).exists()]
        if missing:
            raise RuntimeError('Skrip selesai tanpa berkas wajib: '+', '.join(missing))
        results={name:pd.read_csv(work/name) for name in expected}
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for f in work.glob('*.csv'):
                if f.name not in (DAILY,LANDSAT,EVENT):
                    z.write(f,arcname='hasil/'+f.name)
            z.writestr('hasil/log_pelatihan.txt',logs)
            z.writestr('README_HASIL.txt','Seluruh output dihasilkan dengan audit_dan_eksperimen_ml.py dari dataset yang diunggah. Target adalah keluaran reanalisis ERA5-Land/FLake, bukan deteksi upwelling. Periksa kesesuaian angka dengan BAB IV sebelum dipublikasikan.')
        return results,buf.getvalue(),logs
