"""Halaman khusus training; paparan metode rinci tetap berada di naskah skripsi."""
from pathlib import Path
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from training_runner import execute, validate_inputs, ROOT, DAILY, LANDSAT, EVENT


def show_training_page():
    st.title('Latih ulang model dari dataset CSV')
    st.caption('Unggah data → validasi → jalankan eksperimen → periksa hasil → unduh CSV. Rincian metode ada di naskah skripsi.')
    st.warning('Target: anomali suhu lapisan tercampur **ERA5-Land/FLake**, bukan suhu danau terukur atau deteksi upwelling. Proses memakai pembagian tahun penelitian 2016–2025.')
    daily_upload=st.file_uploader('Dataset harian GEE (wajib)',type='csv',key='upload_daily')
    with st.expander('Data pendamping (opsional: gunakan contoh penelitian jika tidak diunggah)',expanded=False):
        landsat_upload=st.file_uploader('CSV Landsat (opsional)',type='csv',key='upload_landsat')
        events_upload=st.file_uploader('CSV inventaris 15 catatan (opsional)',type='csv',key='upload_events')
        st.caption('Bila tidak diunggah, dua file pendamping bawaan penelitian dipakai untuk audit satelit dan evaluasi deskriptif. Jangan pakai pendamping ini untuk mengklaim dataset baru sudah tervalidasi.')
    with st.expander('Unduh contoh format dataset',expanded=False):
        for filename,caption in [(DAILY,'Dataset harian contoh (3.653 baris)'),(LANDSAT,'Data Landsat penelitian'),(EVENT,'Inventaris 15 catatan penelitian')]:
            p=ROOT/'contoh_data'/filename
            st.download_button('⬇ '+caption,data=p.read_bytes(),file_name=filename,mime='text/csv',key='example_'+filename)
    if daily_upload is None:
        st.info('Unggah dataset harian terlebih dahulu untuk mengaktifkan pelatihan. Data harian harus memiliki kolom sesuai skema penelitian.')
        return
    daily_bytes=daily_upload.getvalue()
    landsat_bytes=landsat_upload.getvalue() if landsat_upload else (ROOT/'contoh_data'/LANDSAT).read_bytes()
    events_bytes=events_upload.getvalue() if events_upload else (ROOT/'contoh_data'/EVENT).read_bytes()
    try:
        d,l,e=validate_inputs(daily_bytes,landsat_bytes,events_bytes)
    except Exception as exc:
        st.error('Validasi dataset gagal: '+str(exc))
        return
    x,y,z=st.columns(3)
    x.metric('Hari tersedia',f'{len(d):,}'.replace(',','.'))
    y.metric('Landsat',f'{len(l)} baris')
    z.metric('Catatan historis',f'{len(e)}')
    st.success('Validasi skema dataset berhasil. Periode latih/validasi: 2016–2023; periode uji: 2024–2025.')
    with st.expander('Pratinjau CSV harian'):
        st.dataframe(d.head(12),use_container_width=True,hide_index=True)
    st.caption('Eksperimen penuh: 2 horizon × 2 skema × 3 algoritma × 4 kandidat parameter × 3 fold, lalu model final dan dua baseline. Proses dapat berat dan dibatasi sumber daya server GitHub/Streamlit Cloud.')
    if st.button('▶️ Mulai latih ulang seluruh model',type='primary',use_container_width=True):
        st.session_state.pop('training_result',None)
        with st.spinner('Melatih seluruh kandidat pada 3-fold ROCV dan menguji 2024–2025. Jangan memuat ulang halaman sampai selesai.'):
            try:
                result,archive,logs=execute(daily_bytes,landsat_bytes,events_bytes)
                st.session_state['training_result']={'outputs':result,'archive':archive,'logs':logs,'source_name':daily_upload.name}
            except Exception as exc:
                st.error(str(exc))
                st.info('Bila pelatihan gagal karena keterbatasan server, jalankan eksperimen lengkap di Google Colab atau laptop.')
                return
    session=st.session_state.get('training_result')
    if session is None:
        return
    st.divider()
    st.success('Pelatihan ulang selesai. Di bawah ini hasil sesi ini, bukan angka arsip bawaan dashboard.')
    st.caption('Sumber sesi: '+str(session['source_name']))
    results=session['outputs']
    metrics=results['hasil_evaluasi_test_2024_2025.csv']
    tuning=results['hasil_tuning_validasi_2022_2023.csv']
    fi=results['hasil_feature_importance.csv']
    st.download_button('⬇ Unduh seluruh hasil pelatihan (ZIP)',session['archive'],file_name='HASIL_LATIH_ULANG_MANINJAU.zip',mime='application/zip',type='primary')
    st.subheader('Hasil pengujian independen 2024–2025')
    st.dataframe(metrics,hide_index=True,use_container_width=True)
    st.download_button('Unduh tabel evaluasi',metrics.to_csv(index=False).encode('utf-8-sig'),file_name='hasil_evaluasi_test_2024_2025.csv',mime='text/csv')
    u=metrics[metrics['Rezim_Fitur'].astype(str).str.startswith('1_')].copy()
    if not u.empty:
        fig=px.bar(u,x='Model',y='Test_Anom_RMSE',color='Horizon',barmode='group',title='RMSE skema utama — hasil pelatihan sesi ini',labels={'Test_Anom_RMSE':'RMSE (°C)'})
        st.plotly_chart(fig,use_container_width=True)
    st.subheader('Deret waktu: referensi ERA5-Land vs prediksi')
    horizon=st.radio('Horizon hasil baru',['t+1','t+3'],horizontal=True)
    pred=results[f'prediksi_uji_independen_{"t1" if horizon=="t+1" else "t3"}_2024_2025.csv']
    if {'target_date','target_anom','pred_anom_XGBoost','pred_anom_SVR'}.issubset(pred.columns):
        series=pred[['target_date','target_anom','pred_anom_XGBoost','pred_anom_SVR']].copy()
        series['target_date']=pd.to_datetime(series['target_date'])
        fig2=px.line(series,x='target_date',y=['target_anom','pred_anom_XGBoost','pred_anom_SVR'],title='Periode uji 2024–2025')
        st.plotly_chart(fig2,use_container_width=True)
    with st.expander('Tuning dan feature importance'):
        st.dataframe(tuning,hide_index=True,use_container_width=True)
        st.dataframe(fi,hide_index=True,use_container_width=True)
    with st.expander('Log eksperimen'):
        st.code(session['logs'][-18000:],language='text')
    st.info('Periksa selisih hasil sesi ini terhadap Tabel BAB IV sebelum mengubah isi skripsi. Halaman dashboard lainnya masih menampilkan hasil arsip sebelumnya.')
