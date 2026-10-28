# Dashboard baca-saja hasil eksperimen skripsi Maninjau (data CSV asli).
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Maninjau | Analisis ERA5-Land", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
COLORS = {"XGBoost":"#0D9488", "Random Forest":"#2563EB", "SVR":"#EA580C", "Persistence Anomali":"#64748B", "Persistence Suhu":"#A855F7"}

@st.cache_data(show_spinner=False)
def read_csv(filename):
    path = DATA / filename
    if not path.is_file():
        st.error(f"File data tidak ditemukan: {filename}. Pastikan folder data/ ikut diunggah ke GitHub.")
        st.stop()
    return pd.read_csv(path)

@st.cache_data(show_spinner=False)
def load_everything():
    return {name:read_csv(name) for name in [
        "evaluasi_model.csv","tuning_rocv.csv","feature_importance.csv","prediksi_t1.csv","prediksi_t3.csv",
        "audit_satelit.csv","evaluasi_upwelling.csv","inventaris_upwelling.csv","audit_grid.csv"]}

files = load_everything()
metric = files['evaluasi_model.csv']
tuning = files['tuning_rocv.csv']
fi = files['feature_importance.csv']
sat = files['audit_satelit.csv']
upw = files['evaluasi_upwelling.csv']
inv = files['inventaris_upwelling.csv']

st.markdown('''<style>
.block-container {padding-top:1.6rem; max-width:1450px}
h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:15px;background:rgba(128,128,128,.045)}
.smallnote {color:#64748b;font-size:.88rem}
</style>''', unsafe_allow_html=True)

st.sidebar.title("🌊 Danau Maninjau")
st.sidebar.caption("Dashboard hasil penelitian • 2016–2025")
page = st.sidebar.radio("Navigasi", [
    "Ringkasan", "Komparasi Model", "Prediksi Deret Waktu", "Validasi ROCV", "Feature Importance",
    "Audit Satelit", "Catatan Historis", "Tentang Data"], index=0)
st.sidebar.divider()
st.sidebar.info("**Target model adalah anomali suhu keluaran ERA5-Land/FLake**, bukan pengukuran in-situ atau deteksi upwelling. Seluruh grafik diturunkan dari CSV eksperimen yang disertakan.")
st.sidebar.caption("Skema 1 · Komparasi RF, XGBoost, dan SVR · Data uji 2024–2025")

MODEL_MAP = {
 "XGBoost":"XGBoost", "Random Forest":"Random Forest", "SVR":"SVR",
 "Baseline Persistence (Anomali: A_t+h = A_t)":"Persistence Anomali",
 "Baseline Persistence (Suhu Mentah: T_t+h = T_t)":"Persistence Suhu",
}
def label_model(value):
    return MODEL_MAP.get(value,value)
def scheme_label(s):
    return "Utama · 73 fitur" if str(s).startswith('1_') else "Ablasi · 55 fitur"
def prep_metric():
    x=metric.copy()
    x['Model Tampilan'] = x['Model'].map(label_model)
    x['Skema'] = x['Rezim_Fitur'].map(scheme_label)
    return x

def fmtc(x,nd=4):
    return (f"{x:,.{nd}f}".replace(',','Q').replace('.',',').replace('Q','.'))

def download_df(frame, filename):
    st.download_button("⬇️ Unduh tabel CSV", frame.to_csv(index=False).encode('utf-8-sig'), file_name=filename, mime='text/csv')

def plot_style(fig, height=480):
    fig.update_layout(height=height, margin=dict(l=20,r=20,t=55,b=30), legend_title_text='', hovermode='x unified')
    return fig

if page == "Ringkasan":
    st.title("Analisis Prediksi Anomali Suhu ERA5-Land")
    st.caption("Kawasan Danau Maninjau · eksperimen retrospektif 2016–2025 · pengujian independen 2024–2025")
    x=prep_metric()
    main=x[(x['Skema']=="Utama · 73 fitur") & (x['Model Tampilan']=="XGBoost")].set_index('Horizon')
    a,b,c,d=st.columns(4)
    a.metric("Jumlah hari reanalisis", "3.653")
    b.metric("RMSE XGBoost · t+1",f"{fmtc(main.loc['t+1','Test_Anom_RMSE'])} °C")
    c.metric("R² XGBoost · t+1",fmtc(main.loc['t+1','Test_Anom_R2']))
    d.metric("RMSE XGBoost · t+3",f"{fmtc(main.loc['t+3','Test_Anom_RMSE'])} °C")
    st.markdown('### Temuan kunci')
    col1,col2=st.columns([1.2,1])
    with col1:
        st.markdown('''**Ringkasan temuan eksperimen**
- XGBoost memiliki RMSE terendah pada skema utama untuk horizon t+1 maupun t+3.
- SVR memperoleh MAE terendah pada skema utama t+3, serta RMSE terendah pada ablasi t+3.
- Performa menurun saat seluruh fitur historis suhu dihapus (ablasi 55 fitur).
- Evaluasi satelit memperlihatkan keterbatasan kesesuaian observasional target ERA5-Land.
- 15 catatan historis dianalisis **deskriptif**, bukan sebagai label klasifikasi upwelling.
''')
    with col2:
        sub=x[(x['Skema']=="Utama · 73 fitur") & (x['Model Tampilan'].isin(['XGBoost','Random Forest','SVR']))]
        fig=px.bar(sub,x='Model Tampilan',y='Test_Anom_RMSE',color='Horizon',barmode='group',title='RMSE skema utama (lebih kecil lebih baik)',labels={'Test_Anom_RMSE':'RMSE (°C)','Model Tampilan':'Algoritma'},text_auto='.4f')
        st.plotly_chart(plot_style(fig,370),use_container_width=True)
    st.warning("Interpretasi: hasil ini menggambarkan prediksi terhadap **keluaran model reanalisis ERA5-Land**, bukan ketepatan suhu air terukur danau ataupun akurasi deteksi upwelling.")

elif page == "Komparasi Model":
    st.title("Komparasi akurasi model")
    horizon=st.radio('Horizon prediksi', ['t+1','t+3'], horizontal=True)
    data=prep_metric().query('Horizon == @horizon').copy()
    scheme=st.selectbox('Rezim fitur', ['Utama · 73 fitur','Ablasi · 55 fitur'])
    data=data[data['Skema']==scheme]
    vals={'RMSE (°C)':'Test_Anom_RMSE','MAE (°C)':'Test_Anom_MAE','R² anomali':'Test_Anom_R2','R² suhu ERA5':'Test_Temp_R2'}
    metric_choice=st.selectbox('Metrik',list(vals))
    key=vals[metric_choice]
    data=data.sort_values(key,ascending='R²' not in metric_choice)
    fig=px.bar(data,x='Model Tampilan',y=key,color='Model Tampilan',color_discrete_map=COLORS,text_auto='.4f',labels={'Model Tampilan':'Model',key:metric_choice},title=f'{metric_choice} · {scheme} · {horizon}')
    fig.update_layout(showlegend=False)
    st.plotly_chart(plot_style(fig),use_container_width=True)
    st.dataframe(data[['Horizon','Skema','Model Tampilan','Test_Anom_MAE','Test_Anom_RMSE','Test_Anom_R2','Test_Temp_R2']],hide_index=True,use_container_width=True)
    download_df(data,'komparasi_model.csv')
    st.caption('MAE dan RMSE pada skala suhu dan anomali identik karena perbedaan hanya berupa klimatologi deterministik; R² bisa berbeda.')

elif page == "Prediksi Deret Waktu":
    st.title("Deret waktu: referensi vs prediksi")
    hor=st.radio('Horizon', ['t+1','t+3'], horizontal=True)
    df=files['prediksi_t1.csv' if hor=='t+1' else 'prediksi_t3.csv'].copy()
    df['target_date']=pd.to_datetime(df['target_date'])
    regimen=st.selectbox('Rezim model', ['Utama · 73 fitur','Ablasi · 55 fitur'])
    names=['XGBoost','Random Forest','SVR']
    if regimen.startswith('Utama'):
        names += ['Persistence Anomali','Persistence Suhu']
    selected=st.multiselect('Tampilkan model',names,default=['XGBoost','SVR'])
    mapping={
        'XGBoost':('pred_anom_XGBoost','pred_anom_meteo_XGBoost'),
        'Random Forest':('pred_anom_Random Forest','pred_anom_meteo_Random Forest'),
        'SVR':('pred_anom_SVR','pred_anom_meteo_SVR'),
        'Persistence Anomali':('pred_anom_Persistence_Anom',None),
        'Persistence Suhu':('pred_anom_Persistence_RawTemp',None),
    }
    low,hi=df['target_date'].min().date(),df['target_date'].max().date()
    interval=st.date_input('Rentang tanggal target',value=(low,hi),min_value=low,max_value=hi)
    if len(interval)==2:
        df=df[df['target_date'].dt.date.between(*interval)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['target_date'],y=df['target_anom'],mode='lines',name='Target referensi ERA5-Land',line=dict(color='#111827',width=2.4)))
    for name in selected:
        col=mapping[name][0 if regimen.startswith('Utama') else 1]
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df['target_date'],y=df[col],mode='lines',name=name,line=dict(color=COLORS[name],width=1.5)))
    fig.update_layout(title=f'Anomali suhu (°C) · {hor}',xaxis_title='Tanggal target',yaxis_title='Anomali (°C)')
    st.plotly_chart(plot_style(fig,550),use_container_width=True)
    st.caption('Semua kurva berasal dari file prediksi data uji independen 2024–2025. Titik pada grafik bukan pengukuran suhu air in-situ.')
    download_df(df,'deret_waktu_'+hor.replace('+','plus')+'.csv')

elif page == "Validasi ROCV":
    st.title('Validasi kronologis 3-fold dan tuning')
    hor=st.radio('Horizon', ['t+1','t+3'], horizontal=True)
    x=tuning[tuning['Horizon']==hor].copy()
    x['Skema']=x['Rezim_Fitur'].map(scheme_label)
    sk=st.selectbox('Skema',['Utama · 73 fitur','Ablasi · 55 fitur'])
    x=x[x['Skema']==sk].copy()
    best=x.sort_values('ROCV_Mean_RMSE').groupby('Algoritma',as_index=False).first()
    st.subheader('Konfigurasi terbaik per algoritma (berdasarkan ROCV mean RMSE)')
    st.dataframe(best[['Algoritma','Konfigurasi','ROCV_Mean_RMSE','ROCV_Std_RMSE','ROCV_Mean_R2']],hide_index=True,use_container_width=True)
    fold_cols=['Fold1_2020_2021_RMSE','Fold2_2021_2022_RMSE','Fold3_2022_2023_RMSE']
    long=best.melt(id_vars=['Algoritma'],value_vars=fold_cols,var_name='Fold',value_name='RMSE')
    long['Fold']=long['Fold'].map(dict(zip(fold_cols,['Fold 1: 2020–2021','Fold 2: 2021–2022','Fold 3: 2022–2023'])))
    fig=px.bar(long,x='Fold',y='RMSE',color='Algoritma',barmode='group',title='RMSE masing-masing fold pada konfigurasi terbaik',color_discrete_map=COLORS)
    st.plotly_chart(plot_style(fig),use_container_width=True)
    with st.expander('Lihat seluruh kandidat hyperparameter'):
        st.dataframe(x,hide_index=True,use_container_width=True)
        download_df(x,'tuning_rocv.csv')
    st.caption('Pemilihan hyperparameter menggunakan data pra-uji hingga 2023; data 2024–2025 tidak digunakan untuk tuning.')

elif page == "Feature Importance":
    st.title('Analisis kepentingan fitur')
    hor=st.radio('Horizon',['t+1','t+3'],horizontal=True)
    algo=st.radio('Algoritma',['XGBoost','Random Forest'],horizontal=True)
    sk=st.radio('Skema',['Utama · 73 fitur','Ablasi · 55 fitur'],horizontal=True)
    raw='1_Utama' if sk.startswith('Utama') else '2_Ablasi_Murni_Meteo'
    x=fi[(fi['Horizon']==hor)&(fi['Model']==algo)&(fi['Rezim_Fitur']==raw)].sort_values('Importance',ascending=False)
    top=st.slider('Jumlah fitur teratas',min_value=5,max_value=min(10,len(x)),value=min(8,len(x)))
    view=x.head(top).sort_values('Importance')
    fig=px.bar(view,x='Importance',y='Fitur',orientation='h',color_discrete_sequence=['#0D9488'],text_auto='.4f',title=f'{algo} · {hor} · {sk}',labels={'Importance':'Kepentingan relatif'})
    st.plotly_chart(plot_style(fig,500),use_container_width=True)
    st.dataframe(x,hide_index=True,use_container_width=True)
    st.caption('Nilai feature importance merupakan ukuran relatif yang spesifik pada model, bukan bukti hubungan kausal variabel meteorologi terhadap peristiwa upwelling.')

elif page == "Audit Satelit":
    st.title('Audit kesesuaian ERA5-Land dan pengamatan satelit')
    st.markdown('Satelit MODIS dan Landsat digunakan sebagai **pembanding observasional**, bukan sebagai label pelatihan regresi.')
    st.subheader('Audit grid ERA5-Land')
    grid=files['audit_grid.csv']
    st.dataframe(grid,hide_index=True,use_container_width=True)
    cols=['Kategori_Sensitivitas','N_Hari','Persen_Hari_10Thn','Bias_Terra_minus_ERA5_Mix_C','MAE_vs_ERA5_Mix_C','RMSE_vs_ERA5_Mix_C','Pearson_r_vs_ERA5_Mix']
    valid=[c for c in cols if c in sat.columns]
    st.subheader('Sensitivitas penyaringan kualitas observasi')
    st.dataframe(sat[valid],hide_index=True,use_container_width=True)
    x=sat[sat['Kategori_Sensitivitas'].str.contains('Terra|Ambang|Fokus',case=False,na=False)].copy()
    fig=px.scatter(x,x='N_Hari',y='MAE_vs_ERA5_Mix_C',size='N_Hari',hover_name='Kategori_Sensitivitas',title='Trade-off jumlah hari valid vs MAE terhadap ERA5-Land',labels={'N_Hari':'Jumlah hari pengamatan lolos filter','MAE_vs_ERA5_Mix_C':'MAE (°C)'},size_max=38)
    st.plotly_chart(plot_style(fig),use_container_width=True)
    st.info('Pada filter MODIS Terra siang paling ketat, terdapat 117 dari 3.653 hari (≈3,2%), dengan MAE ≈1,10°C dan korelasi Pearson ≈0,298. Perbandingan ini mencampurkan besaran fisik dan waktu pengamatan yang berbeda.')
    download_df(sat,'audit_satelit.csv')

elif page == "Catatan Historis":
    st.title('Analisis deskriptif catatan historis 2016–2025')
    st.warning('15 kandidat catatan historis bukan label tervalidasi untuk klasifikasi/deteksi upwelling. Interpretasi dibedakan menurut presisi tanggal dan periode evaluasi.')
    a,b,c=st.columns(3)
    a.metric('Catatan historis',str(len(upw)))
    counts=upw['date_precision'].value_counts()
    b.metric('Tanggal harian spesifik',str(int(counts.get('Exact_Day',0))))
    c.metric('Episode / bulanan',str(int(len(upw)-counts.get('Exact_Day',0))))
    p=st.selectbox('Presisi tanggal',['Semua','Exact_Day','Multi_Day','Month_Only'])
    x=upw if p=='Semua' else upw[upw['date_precision']==p]
    if len(x):
        fig=px.bar(x,x='record_id',y='obs_mean_anom_c',color='date_precision',hover_data=['fish_loss_tons','eval_regime'],title='Rata-rata anomali pada jendela masing-masing catatan',labels={'record_id':'ID catatan','obs_mean_anom_c':'Anomali rerata (°C)'})
        fig.add_hline(y=0,line_dash='dash',line_color='gray')
        st.plotly_chart(plot_style(fig),use_container_width=True)
    st.caption('Jendela kelompok Exact_Day (5 hari) tidak boleh disamakan secara langsung dengan jendela Multi_Day atau Month_Only.')
    st.subheader('Evaluasi deskriptif')
    st.dataframe(x,hide_index=True,use_container_width=True)
    with st.expander('Inventaris sumber dan catatan kejadian'):
        st.dataframe(inv,hide_index=True,use_container_width=True)
    download_df(x,'catatan_historis.csv')

else:
    st.title('Tentang sumber data dan metodologi')
    st.markdown('''**Judul penelitian:** Komparasi Algoritma Random Forest, XGBoost, dan Support Vector Regression untuk Prediksi Anomali Suhu Lapisan Tercampur Reanalisis ERA5-Land di Kawasan Danau Maninjau Berbasis Google Earth Engine.

**Sumber model:** `ECMWF/ERA5_LAND/DAILY_AGGR`, variabel `lake_mix_layer_temperature`, dengan audit CHIRPS, MODIS Terra/Aqua, dan Landsat 8/9.

**Pembagian data:** pelatihan dan tuning kronologis menggunakan periode hingga 2023; pengujian independen pada 2024–2025. Dua horizon: t+1 dan t+3. Skema utama 73 fitur dan ablasi 55 fitur.

**Metode:** Random Forest, XGBoost, SVR RBF, dibandingkan dua baseline persistence. Evaluasi MAE, RMSE, R². Analisis historis upwelling hanya deskriptif-eksploratif.

**Keterbatasan penting:** target adalah **keluaran FLake/ERA5-Land pada grid campuran**; nilai performa bukan validasi lapangan dan dashboard tidak memberi prediksi real-time.

**Reproduksibilitas:** aplikasi ini membaca CSV hasil eksperimen yang disertakan. Pelatihan ulang dilakukan secara terpisah menggunakan skrip eksperimen/Google Colab yang menjadi artefak penelitian.
''')
    st.subheader('Daftar berkas data yang dibundel')
    st.dataframe(pd.DataFrame({'Nama CSV':sorted(p.name for p in DATA.glob('*.csv'))}),hide_index=True,use_container_width=True)
    st.caption('Semua hasil harus dipertahankan konsisten dengan BAB IV dan CSV penelitian final. Jangan mengganti satu CSV saja dengan versi eksperimen lain.')
