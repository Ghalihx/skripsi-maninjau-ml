# PROPOSAL SKRIPSI

**JUDUL SKRIPSI:**
**KOMPARASI ALGORITMA RANDOM FOREST, XGBOOST, DAN SUPPORT VECTOR REGRESSION UNTUK PREDIKSI ANOMALI SUHU LAPISAN TERCAMPUR REANALISIS ERA5-LAND DI KAWASAN DANAU MANINJAU BERBASIS GOOGLE EARTH ENGINE**

**Skema Penelitian:** Pengembangan Algoritma / Model / Analisis (Skema 1)
**Program Studi:** Ilmu Komputer (S1) — Fakultas Ilmu Komputer, Universitas Djuanda (2026)

---

# BAB I — PENDAHULUAN

## A. Latar Belakang Masalah
Danau Maninjau yang terletak di Kabupaten Agam, Provinsi Sumatera Barat, merupakan danau vulkanik-tektonik (kaldera) seluas sekitar 9.950 hektare dengan kedalaman rata-rata 105 meter dan kedalaman maksimum mencapai 165 meter (Putri dkk., 2024). Danau ini memiliki fungsi strategis bagi perekonomian masyarakat sekitar, khususnya sebagai sentra budidaya perikanan air tawar menggunakan sistem Keramba Jaring Apung (KJA) serta pembangkit listrik tenaga air (PLTA). Namun, pemanfaatan perairan secara intensif yang melampaui daya dukung lingkungan (*overcapacity*) telah mengakibatkan akumulasi limbah organik sisa pakan dan metabolisme ikan yang mengendap di dasar danau serta meningkatkan konsentrasi nitrogen dan fosfat di perairan (Komala dkk., 2024; Sunaryani, 2023).

Penumpukan bahan organik di lapisan dasar (*hipolimnion*) memicu kondisi perairan yang bersifat anoksik (miskin oksigen terlarut) serta menghasilkan akumulasi senyawa toksik seperti hidrogen sulfida ($\text{H}_2\text{S}$) dan amonia ($\text{NH}_3$). Dalam kondisi normal, lapisan permukaan danau (*epilimnion*) memiliki suhu yang lebih hangat dibandingkan lapisan dasar, sehingga terbentuk stratifikasi termal yang menahan zat beracun tetap berada di dasar perairan. Akan tetapi, ketika terjadi gangguan hidrometeorologi berupa curah hujan berintensitas tinggi, penurunan radiasi matahari akibat tutupan awan tebal, serta hembusan angin di kawasan kaldera, suhu lapisan permukaan dapat mengalami penurunan anomali (*negative thermal anomaly*). Pendinginan lapisan atas berpotensi memperlemah stabilitas kolom air dan mendukung proses pencampuran vertikal (*vertical mixing* / *upwelling*), yang oleh masyarakat lokal dikenal sebagai fenomena *tubo belerang* atau *umbalan* (Putri dkk., 2024).

Fenomena *upwelling* di Danau Maninjau telah menyebabkan kerugian ekonomi dan ekologis yang berulang. Berdasarkan rangkaian laporan Dinas Ketahanan Pangan dan Perikanan (DKPP) Kabupaten Agam yang diberitakan secara terpisah melalui Kantor Berita ANTARA, insiden kematian massal ikan budidaya KJA yang dikaitkan dengan cuaca ekstrem dan *upwelling* antara lain tercatat sebesar 10 ton pada April 2021 (ANTARA, 2021a), mencapai 1.455 ton sepanjang episode 3–24 Desember 2021 dengan estimasi kerugian sekitar Rp28,9 miliar (ANTARA, 2021b), bertambah menjadi 130 ton pada pertengahan Februari 2022 dengan estimasi kerugian hingga Rp2,6 miliar (ANTARA, 2022), terjadi kembali sebesar 15 ton pada Mei 2023 (ANTARA, 2023), bertambah sebesar 50 ton sehingga mencapai total akumulasi 75 ton pada Januari 2025 (ANTARA, 2025a), serta diawali kematian 8 ton pada akhir November 2025 (ANTARA, 2025b) yang kemudian berlanjut menjadi 1.428,73 ton sepanjang episode Desember 2025 (ANTARA, 2025c). Upaya pemodelan prediksi kejadian *upwelling* secara langsung sebagai masalah klasifikasi biner menghadapi kendala metodologis apabila hanya mengandalkan data sekunder, karena catatan kejadian yang memiliki presisi tanggal harian sangat terbatas dan rentan terhadap jeda waktu pelaporan (*reporting lag*). Sebaliknya, pembentukan label kelas potensi semata-mata dari klasterisasi tanpa pengawasan (*K-Means*) pada variabel cuaca—sebagaimana dilakukan oleh Syakir dkk. (2025)—berisiko menimbulkan sirkularitas apabila tidak diuji terhadap parameter termal yang diestimasi secara terpisah.

Dalam literatur pemodelan deret waktu lingkungan dan penginderaan jauh terkini, pendekatan komputasi yang dapat dipertanggungjawabkan tanpa pengukuran lapangan langsung adalah memprediksi **dinamika suhu lapisan tercampur dan anomali termalnya** pada horizon waktu ke depan ($t+1$ dan $t+3$) menggunakan informasi historis ($\le t$). Penelitian Li dkk. (2024) menunjukkan bahwa algoritma regresi *machine learning* seperti *Random Forest* dan *XGBoost* efektif dalam memodelkan dinamika suhu permukaan danau menggunakan prediktor meteorologi. Di sisi lain, pada kawasan danau kaldera tropis seperti Danau Maninjau, pengamatan suhu permukaan berbasis satelit optik/termal (*MODIS* dan *Landsat 8/9*) sering mengalami kekosongan data harian akibat tutupan awan persisten serta perbedaan karakteristik fisik terhadap estimasi model reanalisis seperti *ERA5-Land* (`lake_mix_layer_temperature`).

Berdasarkan landasan ilmiah tersebut, penelitian ini mengusulkan komparasi kinerja tiga algoritma regresi *machine learning*, yaitu *Random Forest Regressor* (RF), *Extreme Gradient Boosting Regressor* (XGBoost), dan *Support Vector Regression* (SVR)—serta membandingkannya terhadap dua model acuan dasar (*Baseline Persistence* anomali dan suhu mentah)—untuk memprediksi **anomali suhu lapisan tercampur hasil reanalisis ERA5-Land** di kawasan Danau Maninjau berbasis platform *Google Earth Engine* (GEE) periode 2016–2025. Penelitian ini juga dilengkapi dengan audit spasial sel grid *ERA5-Land* dan analisis sensitivitas kesesuaian terhadap suhu permukaan satelit (*MODIS* dan *Landsat 8/9*) sebagai kajian keterbatasan observasional, serta analisis deskriptif-eksploratif terhadap kandidat catatan kejadian *upwelling* historis periode 2016–2025.

---

## B. Permasalahan

### 1. Identifikasi Masalah
Berdasarkan pemaparan pada latar belakang masalah, dapat diidentifikasi permasalahan penelitian yang terbagi menjadi masalah konseptual dan masalah teknikal:
1. **Masalah Konseptual:**
   * Belum adanya kajian yang mengaudit representasi fisik sel grid reanalisis *ERA5-Land* ($\approx 11,1\text{ km}$) terhadap morfologi perairan Danau Maninjau serta menguji sensitivitas kesesuaian suhu hariannya terhadap pengamatan satelit termal (*MODIS Terra/Aqua* dan *Landsat 8/9*) pada berbagai kriteria kontrol kualitas piksel.
   * Belum diketahuinya perbandingan kinerja algoritma *ensemble* pohon keputusan (*Random Forest* dan *XGBoost*) serta metode berbasis fungsi kernel (*Support Vector Regression*) terhadap model acuan *Baseline Persistence* dalam memprediksi anomali suhu lapisan tercampur reanalisis *ERA5-Land* pada horizon 1 hari ($t+1$) dan 3 hari ($t+3$) ke depan, baik ketika menggunakan riwayat suhu sebelumnya (73 fitur) maupun ketika hanya mengandalkan variabel meteorologi (55 fitur).
2. **Masalah Teknikal:**
   * Pemodelan regresi deret waktu untuk anomali suhu rentan mengalami kebocoran informasi masa depan (*lookahead bias*) apabila rata-rata klimatologi harian dihitung dari seluruh rentang dataset atau apabila tanggal target ($t+h$) pada akhir data latih menyeberang ke periode validasi/pengujian.
   * Perhitungan indeks hari kalender (*day-of-year*) standar (1–366) mengalami pergeseran satu hari setelah tanggal 28 Februari pada tahun kabisat (2016, 2020, 2024), serta evaluasi terhadap catatan kejadian *upwelling* historis berisiko bias apabila mencampurkan catatan harian spesifik, episode multi-hari, dan catatan bulanan ke dalam satu jendela waktu yang sama.

### 2. Pernyataan Masalah (*Problem Statement*)
Inti permasalahan dalam penelitian ini adalah bagaimana membangun *pipeline* pemodelan regresi deret waktu dengan pengendalian kebocoran informasi masa depan (*lookahead bias*) melalui isolasi klimatologi dan pemisahan kronologis target untuk membandingkan akurasi algoritma *Random Forest*, *XGBoost*, dan *Support Vector Regression* (SVR) terhadap dua *Baseline Persistence* dalam memprediksi anomali suhu lapisan tercampur reanalisis *ERA5-Land* di kawasan Danau Maninjau berbasis *Google Earth Engine* (2016–2025), sekaligus menetapkan batas validitas observasionalnya terhadap satelit *MODIS/Landsat* dan catatan kejadian *upwelling* historis.

### 3. Rumusan Masalah (*Research Questions*)
Berdasarkan identifikasi dan pernyataan masalah di atas, maka rumusan pertanyaan penelitian (*research questions*) adalah sebagai berikut:
1. Bagaimana karakteristik ketersediaan data, representasi spasial sel grid *ERA5-Land*, dan tingkat kesesuaian antara estimasi suhu lapisan tercampur reanalisis *ERA5-Land* (`lake_mix_layer_temperature`) di kawasan Danau Maninjau terhadap pengamatan suhu permukaan satelit *MODIS* dan *Landsat 8/9* setelah melalui penyaringan kontrol kualitas piksel dan geometri pengamatan di *Google Earth Engine*?
2. Bagaimana kinerja algoritma *Random Forest*, *XGBoost*, dan *Support Vector Regression* (SVR) dibandingkan terhadap dua model *Baseline Persistence* (anomali dan suhu mentah) dalam memprediksi anomali suhu lapisan tercampur reanalisis *ERA5-Land* pada horizon $t+1$ dan $t+3$—baik pada Skema Utama (73 fitur termal-meteorologi $\le t$) maupun Skema Ablasi Murni Meteorologi (55 fitur cuaca $\le t$)—serta bagaimana perilaku deskriptif anomali termal reanalisis tersebut pada periode catatan kejadian *upwelling* historis 2016–2025?

---

## C. Maksud dan Tujuan Penelitian

### 1. Maksud Penelitian
Maksud dari penelitian ini adalah menerapkan dan mengevaluasi secara kritis metode komputasi *machine learning* berbasis regresi deret waktu pada data penginderaan jauh dan reanalisis iklim *Google Earth Engine* guna memperoleh model prediktif anomali suhu lapisan tercampur reanalisis *ERA5-Land* di kawasan Danau Maninjau melalui *pipeline* dengan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi dan pemisahan kronologis target beserta dokumentasi keterbatasan fisiknya.

### 2. Tujuan Penelitian
Adapun tujuan yang ingin dicapai dalam penelitian ini adalah:
1. Mengevaluasi representasi spasial sel grid *ERA5-Land* di kawasan Danau Maninjau serta menganalisis sensitivitas kesesuaian suhunya terhadap pengamatan satelit *MODIS* (khususnya *MODIS Terra* siang pada berbagai ambang jumlah piksel valid, homogenitas spasial, dan sudut pandang sensor) dan *Landsat 8/9* sebagai analisis keterbatasan produk.
2. Membangun, menguji, dan membandingkan akurasi prediksi algoritma *Random Forest*, *XGBoost*, dan *Support Vector Regression* (SVR) terhadap dua model *Baseline Persistence* menggunakan metrik MAE, RMSE, dan $R^2$ (dengan RMSE sebagai metrik utama seleksi dan penentu model terbaik) melalui skema *3-Fold Rolling-Origin Cross-Validation* (2016–2023) dan pengujian independen (2024–2025) dengan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi dan pemisahan kronologis target, diikuti analisis eksploratif terhadap kandidat catatan kejadian *upwelling* historis.

---

## D. Spesifikasi Produk yang Diharapkan
Sesuai dengan karakteristik **Skema 1 (Pengembangan Algoritma / Model / Analisis)**, produk yang dihasilkan dari penelitian ini bukan berupa perangkat lunak antarmuka pengguna (*end-user application*), melainkan **artefak model komputasi dan skrip analisis terstruktur** dengan spesifikasi sebagai berikut:
1. **Skrip Ekstraksi dan Audit Spasial-Piksel Google Earth Engine (`gee_extract_maninjau.js` dan `gee_audit_grid_dan_modis.js`):** Modul JavaScript GEE yang mengekstrak deret waktu harian *ERA5-Land* dan *CHIRPS* (3.653 hari), memetakan poligon sel grid *ERA5-Land* terhadap fraksi air permanen `JRC Global Surface Water` (resolusi 30 m), serta mengekstrak statistik piksel valid, simpangan baku spasial, waktu pengamatan lokal, dan sudut pandang sensor pada citra *MODIS Terra/Aqua* dan *Landsat 8/9*.
2. **Pipa Pemodelan Regresi Deret Waktu dengan Pengendalian Kebocoran Informasi Masa Depan (`audit_dan_eksperimen_ml.py` dan `analisis_audit_lanjutan_gee.py`):** Program Python 3 berbasis pustaka `scikit-learn`, `xgboost`, `pandas`, dan `numpy` yang mengimplementasikan indeks kalender konsisten 365 hari (`cal_doy_365`), isolasi perhitungan klimatologi per lipatan validasi, pemisahan kronologis tanggal target, pembentukan matriks 73 fitur (Skema Utama) dan 55 fitur (Skema Ablasi), seleksi *hyperparameter* melalui *3-Fold Rolling-Origin Cross-Validation*, serta pengujian independen terhadap dua *Baseline Persistence*.
3. **Dataset Terverifikasi dan Laporan Evaluasi Kuantitatif (CSV & Visualisasi 300 DPI):** Himpunan tabel hasil eksperimen dan grafik evaluasi resolusi tinggi yang mendokumentasikan metrik MAE, RMSE, $R^2$, *feature importance*, serta hasil evaluasi eksploratif pada tiga kategori presisi catatan kejadian *upwelling*.

---

## E. Signifikansi dan Manfaat Penelitian
Urgensi penelitian ini terletak pada perlunya standar metodologi yang jujur secara ilmiah dengan pengendalian kebocoran informasi masa depan (*lookahead bias*) ketika memanfaatkan data penginderaan jauh dan reanalisis terbuka untuk pemodelan perairan danau vulkanik tropis di Indonesia. Tanpa audit fisik terhadap sel grid reanalisis dan penyaringan ketat terhadap piksel satelit, model *machine learning* berisiko menghasilkan klaim akurasi semu yang tidak mencerminkan kondisi perairan sebenarnya. Adapun manfaat penelitian ini meliputi:
1. **Manfaat Teoritis (Kontribusi Keilmuan):**
   * Memberikan kontribusi pada bidang Ilmu Komputer dan Sains Data mengenai perancangan *pipeline* regresi deret waktu (*time-series regression*) dengan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi per lipatan validasi, pemisahan kronologis tanggal target di perbatasan *split*, dan komparasi skema fitur autoregresif (73 fitur) versus murni meteorologi (55 fitur).
   * Mendokumentasikan secara kuantitatif perbedaan karakteristik antara keluaran suhu lapisan tercampur reanalisis *ERA5-Land* dan pengamatan suhu permukaan sesaat satelit (*MODIS* & *Landsat*) di perairan danau kaldera tropis.
2. **Manfaat Praktis (Bagi Peneliti Lanjutan dan Pengelola Informasi Lingkungan):**
   * Memberikan informasi empiris mengenai perbedaan kinerja algoritma *ensemble* pohon keputusan (*Random Forest*, *XGBoost*) dan metode kernel (*SVR*) ketika memprediksi target reanalisis dengan maupun tanpa riwayat suhu sebelumnya.
   * Menyediakan evaluasi awal yang proporsional mengenai keterbatasan penggunaan data reanalisis skala grid ($\approx 11,1\text{ km}$) dan satelit termal untuk pemantauan perairan Danau Maninjau sebelum dapat diintegrasikan dengan sensor *in-situ* lapangan.

---

## F. Asumsi dan Batasan Masalah (Keterbatasan Penelitian)

### 1. Asumsi Penelitian
1. Keluaran model danau 1-dimensi *FLake* pada dataset *ERA5-Land Daily Aggregated* (`lake_mix_layer_temperature`) diasumsikan merepresentasikan respons termodinamika teoretis lapisan air tercampur terhadap variabel pemaksa atmosfer skala grid di kawasan kaldera Danau Maninjau secara konsisten sepanjang periode 2016–2025.
2. Pada kerangka evaluasi retrospektif, seluruh agregasi variabel meteorologi dan termal harian pada hari ke-$t$ diasumsikan telah lengkap pada akhir hari ke-$t$ ketika model melakukan prediksi untuk horizon $t+1$ dan $t+3$.

### 2. Batasan Masalah dan Keterbatasan Penelitian
Agar klaim ilmiah penelitian ini terukur dan tidak melampaui bukti empiris yang tersedia, ruang lingkup penelitian dibatasi pada ketentuan berikut:
1. **Lokasi Studi dan Representasi Spasial Grid:** Zona perairan tengah (pelagis) Danau Maninjau, Kabupaten Agam, Sumatera Barat (titik pusat $0^\circ 18' 54'' \text{ LS}$ / $-0,315^\circ$ dan $100^\circ 11' 13'' \text{ BT}$ / $100,187^\circ$ dengan radius *buffer* inti perairan $2,0\text{ km}$ dan masker perairan permanen `JRC Global Surface Water > 90%`). ROI ini berada di dalam satu sel grid utama *ERA5-Land* (`100,20° BT, -0,30° LS`, luas $123,64\text{ km}^2$) yang memiliki fraksi perairan permanen JRC sebesar $52,20\%$ ($64,54\text{ km}^2$) dan area daratan/lereng kaldera sebesar $47,80\%$.
2. **Pembatasan Klaim Variabel Target ($y_{t+1}, y_{t+3}$):** Variabel target yang diprediksi dan dievaluasi secara harian adalah **suhu lapisan tercampur (`era5_lake_mix_temp_c`) dan anomali termalnya yang berasal dari model reanalisis *ERA5-Land* (*FLake*)**, **bukan** suhu air aktual Danau Maninjau yang diukur secara *in-situ*. Mengingat parameter kedalaman dan fraksi danau internal di dalam konfigurasi *FLake* ECMWF merupakan penyederhanaan 1-dimensi pada grid campuran daratan–perairan serta kesesuaiannya dengan satelit masih rendah, keberhasilan metrik regresi ($R^2$, MAE, RMSE) hanya berlaku bagi prediksi keluaran reanalisis *ERA5-Land*.
3. **Kedudukan Data Satelit (*MODIS* dan *Landsat 8/9*):** Data suhu permukaan satelit digunakan sebagai **audit keterbatasan dan pembanding observasional** (dengan memperhitungkan perbedaan waktu lintas lokal `view_time_hr`, jumlah piksel valid, simpangan baku spasial, dan sudut pandang sensor), bukan sebagai target pelatihan harian karena tingginya frekuensi tutupan awan.
4. **Asumsi Waktu Ketersediaan Data (*Data Availability Latency*):** Eksperimen ini dilakukan dalam kerangka evaluasi retrospektif; penerapan sebagai sistem prakiraan operasional *real-time* memiliki keterbatasan berupa jeda waktu rilis (*latency*) produk *ERA5-Land Daily* sekitar 5 hari.
5. **Sifat Eksploratif Analisis Kejadian *Upwelling*:** Evaluasi terhadap 15 kandidat catatan historis (2016–2025) dipisahkan secara deskriptif menurut ketepatan tanggalnya (`Exact_Day` $N=10$, `Multi_Day_Episode/Observation` $N=3$, dan `Month_Only` $N=2$). Analisis ini bersifat **deskriptif-eksploratif** dan bukan pembuktian kemampuan deteksi maupun kausalitas kejadian *upwelling*.

---

## G. Definisi Istilah dan Definisi Operasional
Untuk menghindari perbedaan penafsiran, berikut adalah definisi istilah dan definisi operasional variabel utama dalam penelitian ini:
1. **Suhu Lapisan Tercampur Reanalisis (*ERA5-Land Lake Mixed-Layer Temperature*, `era5_lake_mix_temp_c`):** Estimasi suhu rata-rata harian (dalam satuan $^\circ\text{C}$, dikonversi dari Kelvin dengan mengurangi $273,15$) pada lapisan air tercampur bagian atas yang dihitung oleh modul danau 1-dimensi *FLake* di dalam sistem reanalisis `ECMWF/ERA5_LAND/DAILY_AGGR` pada sel grid utama kawasan Danau Maninjau.
2. **Klimatologi Hari Kalender Konsisten 365 Hari (`clim_doy_c`):** Rata-rata suhu lapisan tercampur historis untuk setiap indeks tanggal kalender $d \in \{1, 2, \dots, 365\}$ (di mana tanggal 29 Februari pada tahun kabisat dipetakan ke indeks 59 bersama 28 Februari) yang dihaluskan dengan rata-rata bergerak melingkar 15 hari dan dihitung secara ketat hanya dari periode data latih aktif.
3. **Anomali Suhu Lapisan Tercampur (`anom_doy_c`, $A_t$):** Selisih antara suhu lapisan tercampur harian *ERA5-Land* pada hari ke-$t$ terhadap nilai klimatologi kalender pada hari yang bersangkutan ($A_t = T_t - \bar{T}_{\text{clim}, d(t)}$), dinyatakan dalam satuan $^\circ\text{C}$.
4. **Skema Utama (Rezim 1 — 73 Fitur Input $\le t$):** Konfigurasi pemodelan yang menggunakan gabungan 18 fitur riwayat termal danau (suhu, anomali, *lag* 1, 2, 3, 7 hari, rata-rata dan simpangan baku bergulir 3, 7, 14 hari, serta selisih 1 dan 3 hari) dan 55 fitur kalender serta riwayat meteorologi hingga hari ke-$t$.
5. **Skema Ablasi Murni Meteorologi (Rezim 2 — 55 Fitur Input $\le t$):** Konfigurasi pemodelan yang menghapus seluruh 18 fitur suhu dan anomali danau masa lalu, sehingga model hanya menggunakan 2 fitur kalender siklik ($\sin/\cos$) dan 53 fitur meteorologi (11 variabel cuaca hari ke-$t$, 24 fitur *lag*, dan 18 fitur rata-rata/akumulasi bergulir 3, 7, 14 hari).

---

## H. Sistematika Penulisan
Sistematika penulisan dalam penelitian skripsi ini disusun sebagai berikut:
* **BAB I PENDAHULUAN:** Menguraikan latar belakang masalah, permasalahan (identifikasi masalah, pernyataan masalah, dan rumusan masalah), maksud dan tujuan penelitian, spesifikasi produk yang diharapkan, signifikansi dan manfaat penelitian, asumsi dan batasan masalah, definisi istilah dan definisi operasional, serta sistematika penulisan.
* **BAB II TINJAUAN PUSTAKA DAN KERANGKA TEORITIS:** Memuat tinjauan objek penelitian (kawasan Danau Maninjau), tabel ringkasan 10 penelitian terdahulu (2022–2026), landasan teori mengenai stratifikasi dan anomali termal danau, karakteristik produk *Google Earth Engine* (*ERA5-Land FLake*, *CHIRPS*, *MODIS*, *Landsat*), formulasi matematis *Baseline Persistence*, *Random Forest*, *XGBoost*, dan SVR, metrik evaluasi regresi, kerangka pemikiran, serta hipotesis penelitian.
* **BAB III METODOLOGI PENELITIAN:** Menjelaskan metode dan tahapan penelitian Skema 1, model dan arsitektur yang diusulkan, prosedur pengembangan dan implementasi (pengumpulan data GEE, audit spasial grid dan sensitivitas piksel satelit, *preprocessing* dengan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi dan pemisahan kronologis target beserta rincian 73 dan 55 fitur), desain uji coba (*3-Fold Rolling-Origin Cross-Validation* dan pengujian independen), serta jadwal penelitian.
* **BAB IV HASIL DAN PEMBAHASAN:** Menyajikan deskripsi objek dan hasil implementasi ekstraksi data beserta audit grid *ERA5-Land*, hasil pengujian kesesuaian *ERA5–MODIS/Landsat*, hasil komparasi akurasi prediksi RF, XGBoost, dan SVR terhadap dua *Baseline Persistence* pada Skema Utama (73 fitur) dan Skema Ablasi (55 fitur), analisis *feature importance*, evaluasi deskriptif pada kandidat catatan *upwelling*, serta pembahasan komprehensif.
* **BAB V KESIMPULAN DAN SARAN:** Berisi kesimpulan yang menjawab seluruh rumusan masalah dan pengujian hipotesis secara proporsional sesuai batasan validitas data, diikuti saran pengembangan teknis untuk penelitian lanjutan.

---
---

# BAB II — TINJAUAN PUSTAKA DAN KERANGKA TEORITIS

## A. Tinjauan Objek Penelitian (Kawasan Danau Maninjau)
Danau Maninjau adalah danau kaldera vulkanik-tektonik yang terletak di Kecamatan Tanjung Raya, Kabupaten Agam, Provinsi Sumatera Barat, pada koordinat geografis sekitar $0^\circ 12' - 0^\circ 25'\text{ LS}$ dan $100^\circ 07' - 100^\circ 16'\text{ BT}$ dengan elevasi permukaan air sekitar $461,5\text{ meter}$ di atas permukaan laut (Putri dkk., 2024). Danau ini memiliki luas perairan sekitar $99,5\text{ km}^2$ ($9.950\text{ hektare}$), panjang maksimum arah utara–selatan sekitar $16\text{ km}$, lebar maksimum arah barat–timur sekitar $7\text{ km}$, kedalaman rata-rata $105\text{ m}$, dan kedalaman maksimum mencapai $165\text{ m}$. Secara morfologis, perairan Danau Maninjau dikelilingi oleh dinding perbukitan kaldera yang curam dengan ketinggian berkisar antara $600\text{ m}$ hingga lebih dari $1.400\text{ m}$ mdpl, dan hanya memiliki satu saluran keluar alami (*outlet*) yaitu Batang Antokan di sisi barat laut yang dimanfaatkan untuk PLTA Maninjau (Komala dkk., 2024).

Waktu tinggal air (*water retention time*) yang sangat lama akibat morfologi cekungan semi-tertutup menyebabkan nutrien dan bahan organik dari aktivitas antropogenik—terutama ribuan petak Keramba Jaring Apung (KJA)—terakumulasi di dasar perairan. Berdasarkan kajian deret waktu status mutu air oleh Sunaryani (2023), perairan Danau Maninjau telah mengalami penurunan kualitas menjadi kategori cemar berat dengan status trofik **eutrofik hingga hipereutrofik**. Kondisi dasar perairan yang kaya sulfida dan miskin oksigen ini menjadikan stabilitas suhu lapisan atas sangat penting untuk dipantau dan dimodelkan secara kuantitatif.

---

## B. Penelitian Terdahulu (Tinjauan Pustaka)
Penelitian ini merujuk pada 10 (sepuluh) penelitian terdahulu dari jurnal ilmiah nasional dan internasional dalam rentang waktu 5 tahun terakhir (2022–2026), yang terbagi ke dalam tiga kelompok kontribusi: (1) dinamika fisik, kualitas air, dan *upwelling* Danau Maninjau, (2) pemodelan suhu permukaan dan potensi *upwelling* danau menggunakan *machine learning*, serta (3) pemrosesan data penginderaan jauh menggunakan *Google Earth Engine*. Ringkasan penelitian terdahulu disajikan pada Tabel 1.

**Tabel 1. Ringkasan Penelitian Terdahulu (2022–2026)**

| No | Peneliti dan Tahun | Judul dan Nama Jurnal | Metode dan Variabel | Hasil Penelitian | Kekurangan / Celah Riset (*Research Gap*) |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | Li dkk. (2024) | *Lake Surface Temperature Predictions under Different Climate Scenarios with Machine Learning Methods: A Case Study of Qinghai Lake and Hulun Lake, China* <br><br>**Jurnal:** *Remote Sensing* (MDPI), Vol. 16(17), 3220. | **Metode:** *Random Forest* (RF), *XGBoost*, dan *Multilayer Perceptron Neural Network* (MLPNN).<br>**Variabel:** Suhu udara, kecepatan angin, radiasi matahari, curah hujan, dan data suhu permukaan satelit MODIS. | Model *Random Forest* dan *XGBoost* efektif memprediksi suhu permukaan danau dengan tingkat kesalahan rendah ($\text{MAE} = 0,348^\circ\text{C}$, $\text{RMSE} = 0,611^\circ\text{C}$), di mana suhu udara dan radiasi menjadi prediktor yang paling berasosiasi kuat. | Studi dilakukan pada danau subtropis dan belum mengkaji prediksi anomali suhu lapisan tercampur reanalisis *ERA5-Land* beserta audit keterbatasan satelit pada danau kaldera tropis seperti Danau Maninjau. |
| 2 | Syakir dkk. (2025) | *Forecasting Upwelling in Lake Maninjau Using Vector Autoregressive, Support Vector Machine and Dashboard Visualization* <br><br>**Jurnal:** *JITK (Jurnal Ilmu Pengetahuan dan Teknologi Komputer)*, Vol. 11(2), hal. 580–590. | **Metode:** *Vector Autoregressive* (VAR), *K-Means*, dan SVM.<br>**Variabel:** Suhu permukaan, curah hujan, dan kecepatan angin di Danau Maninjau (2020–2024). | Kombinasi peramalan VAR(17) dan klasifikasi SVM menghasilkan nilai *F1-Score* di atas 0,95 dalam mengklasifikasikan klaster potensi *upwelling*. | Label target dibentuk menggunakan klasterisasi *K-Means* dari variabel cuaca tanpa pemeriksaan terhadap catatan kejadian historis maupun perbandingan regresi RF, XGBoost, dan SVR terhadap *Baseline Persistence*. |
| 3 | Putri dkk. (2024) | *The Impact of Weather Condition Changes on Vertical Distribution of Sulfides in Lake Maninjau Based on Observation Data* <br><br>**Jurnal:** *LIMNOTEK: Perairan Darat Tropis di Indonesia* (BRIN), Vol. 30(1), hal. 1–11. | **Metode:** Analisis observasi profil vertikal suhu, DO, sulfida, dan kondisi meteorologi.<br>**Variabel:** Suhu air vertikal, durasi penyinaran matahari, curah hujan, kecepatan angin (Nov 2022 & Agu 2023). | Penurunan suhu udara, berkurangnya durasi penyinaran matahari, tingginya curah hujan, dan peningkatan kecepatan angin pada akhir November 2022 melemahkan stratifikasi termal dan mendorong kenaikan sulfida ke permukaan (hingga $4,16\,\mu\text{g/L}$). | Analisis bersifat observasional pada dua periode survei pendek dan belum membangun model prediksi deret waktu berbasis *machine learning*. |
| 4 | Ulhaq dkk. (2025) | *Integration of Machine Learning and Time Series Analysis for Upwelling Prediction Dashboard in Lake Laut Tawar, Indonesia: A Study Based on Climate Forecasting* <br><br>**Jurnal:** *Theoretical and Applied Climatology* (Springer), Vol. 156(9), 463. | **Metode:** Analisis deret waktu dan pemodelan *Machine Learning*.<br>**Variabel:** Suhu permukaan, kecepatan angin, curah hujan, dan indikator atmosfer di Danau Laut Tawar. | Integrasi peramalan deret waktu iklim dan *machine learning* mampu memetakan periode kondisi meteorologi yang berkaitan dengan risiko *upwelling* pada danau tropis di Sumatera. | Studi dilakukan di Danau Laut Tawar (Aceh) dan belum menguji komparasi algoritma regresi RF, XGBoost, dan SVR untuk memprediksi anomali termal reanalisis *ERA5-Land* berbasis katalog *Google Earth Engine*. |
| 5 | Fajri dkk. (2026) | *Perbandingan Algoritma Random Forest dan XGBoost dalam Klasifikasi Kualitas Air Danau Maninjau* <br><br>**Jurnal:** *Jurnal Nasional Teknologi dan Sistem Informasi (TEKNOSI)*, Universitas Andalas, Vol. 12(2), hal. 276–283. | **Metode:** *Random Forest* dan *XGBoost* dengan skema pembagian data 8:2.<br>**Variabel:** Parameter kualitas air dan *Water Quality Index* (WQI) Danau Maninjau. | Algoritma *Random Forest* mencatat akurasi 97%, mengungguli *XGBoost* (87%) dalam memodelkan klasifikasi mutu air di Danau Maninjau. | Terbatas pada klasifikasi mutu air statis dan belum menerapkan varian regresi (RF, XGBoost, SVR) pada data deret waktu GEE dengan evaluasi terhadap *Baseline Persistence*. |
| 6 | Aldiansyah dan Saputra (2022) | *Comparison of Machine Learning Algorithms for Land Use and Land Cover Analysis Using Google Earth Engine (Case Study: Wanggu Watershed)* <br><br>**Jurnal:** *International Journal of Remote Sensing and Earth Sciences (IJReSES)* (BRIN), Vol. 19(2), hal. 197–210. | **Metode:** *Random Forest* (RF), SVM, dan CART berbasis *Google Earth Engine*.<br>**Variabel:** Komposit spektral citra Landsat-8 dan Sentinel-2. | *Random Forest* mengungguli SVM dan CART dengan akurasi 94,74% (Landsat-8) dan 96,15% (Sentinel-2) dalam pemrosesan data geospasial di GEE. | Berfokus pada klasifikasi spasial daratan, belum mengeksplorasi ekstraksi deret waktu klimatologi (*ERA5-Land* & *CHIRPS*) serta audit kualitas piksel termal perairan danau di GEE. |
| 7 | Valbuena dkk. (2022) | *3D Flow Structures During Upwelling Events in Lakes of Moderate Size* <br><br>**Jurnal:** *Water Resources Research* (Wiley/AGU), Vol. 58(3), e2021WR030666. | **Metode:** Pemodelan hidrodinamika 3D dan analisis deret waktu angin-suhu.<br>**Variabel:** Kecepatan dan arah angin, profil suhu air, anomali suhu permukaan. | Impuls tegangan angin memicu kemiringan termoklin dan berasosiasi dengan penurunan suhu permukaan di zona terdampak *upwelling*. | Membutuhkan simulasi numerik 3D yang kompleks dan data batimetri rinci, belum mengkaji pemodelan regresi deret waktu berbasis data reanalisis terbuka. |
| 8 | Hamze-Ziabari dkk. (2023) | *Chimney-Like Intense Pelagic Upwelling in the Center of Basin-Scale Cyclonic Gyres in Large Lake Geneva* <br><br>**Jurnal:** *Journal of Geophysical Research: Oceans* (Wiley), Vol. 128(7), e2022JC019592. | **Metode:** Pemodelan hidrodinamika, pengukuran lapangan, dan citra satelit termal.<br>**Variabel:** Komponen angin, suhu permukaan danau, struktur pusaran air. | Menunjukkan bahwa *upwelling* di bagian tengah danau (*pelagic zone*) berkaitan dengan anomali dingin suhu permukaan air pasca-gangguan angin regional, namun juga menekankan keterbatasan pengamatan permukaan saja. | Fokus pada analisis mekanisme fisika fluida dan belum menerapkan algoritma regresi *machine learning* untuk memprediksi anomali suhu beberapa hari ke depan. |
| 9 | Naumenko dkk. (2025) | *Spatiotemporal Variability of the Thermal Structure of Upwelling in the Southwestern Part of Lake Baikal* <br><br>**Jurnal:** *Izvestiya, Atmospheric and Oceanic Physics* (Springer), Vol. 61(11), hal. 1446–1454. | **Metode:** Analisis spasial-temporal pengamatan suhu pada stasiun dan analisis arus permukaan.<br>**Variabel:** Suhu permukaan dan kolom air, struktur termal *coastal upwelling*, serta dinamika arus permukaan. | Mengungkapkan variabilitas spasial-temporal struktur termal selama berlangsungnya kejadian *coastal upwelling* serta keterkaitannya dengan penurunan suhu lokal dan dinamika arus permukaan di bagian barat daya Danau Baikal. | Analisis berfokus pada struktur termal observasional pada kejadian *upwelling* di danau lintang tinggi dan belum membangun model prediksi regresi harian berbasis algoritma RF, XGBoost, maupun SVR. |
| 10 | Sunaryani (2023) | *Penentuan Status Mutu Air dan Status Trofik di Perairan Danau Maninjau* <br><br>**Jurnal:** *Jurnal Teknologi Lingkungan* (BRIN), Vol. 24(1), hal. 21–27. | **Metode:** Metode STORET dan klasifikasi status trofik deret waktu 10 tahun (2011–2020).<br>**Variabel:** Parameter fisik, kimia, dan biologis mutu air di Danau Maninjau. | Perairan Danau Maninjau berstatus cemar berat serta eutrofik–hipereutrofik akibat beban limbah organik KJA, menjelaskan tingginya kerentanan ekologis saat terjadi pencampuran kolom air. | Mengkaji status mutu air secara periodik, namun belum menyediakan model prediksi kondisi termal berbasis faktor cuaca harian. |

*Sumber: Diolah dari Pustaka Primer (2026)*

---

## C. Landasan Teori

### 1. Stratifikasi Termal, Anomali Suhu Lapisan Tercampur, dan Keterbatasannya sebagai Indikator *Upwelling*
Di danau tropis dalam seperti Danau Maninjau, perbedaan suhu antara lapisan permukaan (*epilimnion*, berkisar $27^\circ\text{C} - 30^\circ\text{C}$) dan lapisan dasar (*hipolimnion*, berkisar $25^\circ\text{C} - 26^\circ\text{C}$) relatif kecil ($\Delta T \approx 2^\circ\text{C} - 4^\circ\text{C}$), namun cukup untuk mempertahankan stratifikasi termal pada kondisi cuaca normal (Putri dkk., 2024).

Ketika terjadi kombinasi gangguan meteorologi berupa berkurangnya radiasi matahari neto akibat awan tebal, curah hujan tinggi, dan peningkatan kecepatan angin permukaan selama beberapa hari, suhu lapisan atas perairan ($T_t$) dapat mengalami penurunan relatif terhadap kondisi termal latar belakangnya—sebagaimana karakteristik perubahan struktur termal permukaan yang dikaji pada fenomena *upwelling* danau dalam (Valbuena dkk., 2022; Hamze-Ziabari dkk., 2023; Naumenko dkk., 2025). Agar deviasi suhu jangka pendek dapat dipisahkan secara konsisten dari siklus musiman tahunan pada pemodelan deret waktu harian, penelitian ini menetapkan definisi operasional **Anomali Suhu Harian ($A_t$)** terhadap rata-rata klimatologis hari kalender ($\bar{T}_{\text{clim}, d(t)}$) yang dihitung dari periode data latih historis:
$$A_t = T_t - \bar{T}_{\text{clim}, d(t)}$$
di mana $d(t) \in \{1, 2, \dots, 365\}$ adalah indeks hari kalender yang konsisten pada tahun kabisat maupun non-kabisat. Secara ilmiah, perlu ditegaskan bahwa anomali suhu negatif ($A_t < 0$), terlebih lagi apabila berasal dari estimasi model reanalisis skala grid, **tidak secara otomatis membuktikan terjadinya keruntuhan stratifikasi maupun kejadian *upwelling* aktual**, karena konfirmasi *upwelling* memerlukan pengukuran profil vertikal suhu dan oksigen terlarut di lapangan. Oleh sebab itu, keterkaitan antara anomali suhu reanalisis dengan catatan kejadian *upwelling* hanya dapat dianalisis secara eksploratif.

---

### 2. Platform *Google Earth Engine* (GEE) dan Karakteristik Fisik Dataset
Penelitian ini memanfaatkan tiga kelompok dataset pada *Google Earth Engine* (GEE) dengan memperhatikan perbedaan mendasar dalam resolusi spasial, waktu pengamatan, dan besaran fisik yang diestimasi:
1. **ECMWF ERA5-Land Daily Aggregated (`ECMWF/ERA5_LAND/DAILY_AGGR`):** Memiliki resolusi spasial asli $0,1^\circ \times 0,1^\circ$ ($\approx 11.132\text{ meter}$, luas sel grid $\approx 123,64\text{ km}^2$ di ekuator). Menyediakan variabel meteorologi harian (suhu udara 2m, komponen angin $u_{10}$ dan $v_{10}$, radiasi matahari neto, radiasi termal neto, tekanan permukaan, curah hujan) serta keluaran model danau 1-dimensi *FLake* berupa **suhu lapisan air tercampur (`lake_mix_layer_temperature`)**, suhu dasar (`lake_bottom_temperature`), dan kedalaman lapisan tercampur (`lake_mix_layer_depth`). Karena ukuran grid $\approx 11,1\text{ km}$ mencakup perairan danau sekaligus daratan perbukitan kaldera di sekitarnya, serta menggunakan parameterisasi danau 1D, variabel `era5_lake_mix_temp_c` merupakan **keluaran model reanalisis regional**, bukan pengukuran suhu air *in-situ*.
2. **CHIRPS Daily (`UCSB-CHG/CHIRPS/DAILY`):** Menyediakan estimasi curah hujan harian dari gabungan citra inframerah satelit dan stasiun cuaca dengan resolusi spasial asli $0,05^\circ$ ($\approx 5.566\text{ meter}$).
3. **Observasi Satelit Termal MODIS (`MOD11A1` & `MYD11A1`) dan Landsat 8/9 (`ST_B10`):**
   * *MODIS Terra* (lintas pagi/siang $\approx 10:15\text{ WIB}$, malam $\approx 22:15\text{ WIB}$) dan *MODIS Aqua* (lintas siang $\approx 13:45\text{ WIB}$, dini hari $\approx 01:45\text{ WIB}$) memiliki resolusi spasial $\approx 1\text{ km}$ ($\approx 926,6\text{ m}$ pada proyeksi sinusoidal asli, mencakup maksimal 22 piksel air di dalam ROI radius $2\text{ km}$), dilengkapi band kontrol kualitas `QC_Day`, `QC_Night`, waktu pengamatan lokal (`Day_view_time`), dan sudut pandang sensor (`Day_view_angle`).
   * *Landsat 8/9 Level-2* (lintas pagi $\approx 10:00\text{ WIB}$) memiliki resolusi spasial $30\text{ meter}$ dengan siklus ulang 16 hari, dilengkapi band `QA_PIXEL` dan ketidakpastian termal `ST_QA`.
   Berbeda dengan `era5_lake_mix_temp_c` yang merepresentasikan rata-rata 24 jam suhu lapisan tercampur model, sensor satelit mengukur **suhu kulit permukaan sesaat (*instantaneous thermal skin temperature*)** pada lapisan tipis $\sim 10-20\,\mu\text{m}$ yang sangat dipengaruhi oleh waktu lintas satelit, sudut pandang sensor di cekungan kaldera, serta kontaminasi awan tipis (*sub-pixel cirrus*).

---

### 3. Model Pembanding Dasar (*Baseline Persistence*) dan Algoritma Regresi *Machine Learning*

#### a. Dua Model *Baseline Persistence* (Anomali dan Suhu Mentah)
Dalam peramalan deret waktu (*time-series forecasting*), algoritma *machine learning* wajib dibandingkan terhadap dua varian model acuan *Persistence* tanpa pelatihan parameter:
1. ***Baseline Persistence* Anomali ($\hat{A}_{t+h}^{\text{pers-A}} = A_t$):** Mengasumsikan bahwa deviasi anomali termal pada horizon $t+h$ bertahan sama dengan anomali pada hari ke-$t$, sehingga prediksi suhu aktualnya mengikuti pergeseran klimatologi kalender ($\hat{T}_{t+h}^{\text{pers-A}} = A_t + \bar{T}_{\text{clim}, d(t+h)}$).
2. ***Baseline Persistence* Suhu Mentah ($\hat{T}_{t+h}^{\text{pers-T}} = T_t$):** Mengasumsikan bahwa suhu absolut pada horizon $t+h$ sama persis dengan suhu pada hari ke-$t$, sehingga prediksi anomalinya adalah $\hat{A}_{t+h}^{\text{pers-T}} = T_t - \bar{T}_{\text{clim}, d(t+h)}$.

Suatu algoritma *machine learning* dinyatakan memberikan nilai tambah prediktif pada target reanalisis apabila mampu menghasilkan nilai MAE dan RMSE yang lebih rendah serta $R^2$ yang lebih tinggi dibandingkan kedua model *Persistence* tersebut.

#### b. Algoritma *Random Forest Regressor* (RF)
*Random Forest Regressor* membangun sejumlah $K$ pohon regresi (*regression trees*) secara paralel dari sampel *bootstrap* data pelatihan (Li dkk., 2024). Pada setiap pemisahan simpul (*node*), algoritma memilih subset fitur secara acak dan mencari titik potong yang meminimalkan *Sum of Squared Errors* (SSE):
$$\text{SSE} = \sum_{i \in R_1} (y_i - \bar{y}_{R_1})^2 + \sum_{i \in R_2} (y_i - \bar{y}_{R_2})^2$$
Prediksi akhir diperoleh dari rata-rata seluruh $K$ pohon:
$$\hat{y} = \frac{1}{K} \sum_{k=1}^{K} h_k(\mathbf{x})$$

#### c. Algoritma *Extreme Gradient Boosting Regressor* (XGBoost)
*XGBoost Regressor* membangun pohon regresi secara sekuensial, di mana pohon ke-$m$ dilatih untuk memprediksi residual kesalahan dari prediksi iterasi sebelumnya dengan laju pembelajaran $\eta$:
$$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta f_m(\mathbf{x}_i)$$
Fungsi objektif yang diminimalkan mencakup fungsi kerugian kuadrat dan penalti regularisasi kompleksitas pohon ($\gamma, \lambda$):
$$\mathcal{L}^{(m)} = \sum_{i=1}^{n} \frac{1}{2}\left(y_i - \hat{y}_i^{(m)}\right)^2 + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

#### d. Algoritma *Support Vector Regression* (SVR)
*Support Vector Regression* (SVR) mencari fungsi aproksimasi $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$ yang menjaga kesalahan prediksi berada di dalam tabung toleransi $\varepsilon$ (*epsilon-insensitive tube*):
$$\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)$$
dengan menggunakan fungsi kernel *Radial Basis Function* (RBF), $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$, untuk memetakan hubungan nonlinier antara prediktor dan respons termal reanalisis.

---

### 4. Metrik Evaluasi Model Regresi
Akurasi prediksi dievaluasi menggunakan tiga metrik utama pada data uji independen ($n$ sampel), baik pada skala anomali ($A_{t+h}$) maupun skala suhu reanalisis ($T_{t+h}$):
1. **Mean Absolute Error (MAE):** $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
2. **Root Mean Squared Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
3. **Coefficient of Determination ($R^2$):** $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$

Dalam penelitian ini, **RMSE dan $R^2$ ditetapkan sebagai metrik utama penentu peringkat model terbaik** karena fungsi objektif pelatihan pohon regresi dan kriteria seleksi *hyperparameter* pada *Rolling-Origin Cross-Validation* meminimalkan kesalahan kuadrat (*squared error*) yang memberikan bobot penalti lebih tegas terhadap simpangan prediksi besar, sedangkan **MAE** dilaporkan sebagai metrik pendamping untuk mengukur rata-rata kesalahan absolut harian.

*(Catatan matematis: Karena untuk setiap tanggal target $t+h$ nilai klimatologi $\bar{T}_{\text{clim}, d(t+h)}$ bersifat deterministik, maka selisih kesalahan $\hat{T}_{t+h} - T_{t+h} \equiv \hat{A}_{t+h} - A_{t+h}$, sehingga nilai MAE dan RMSE pada skala suhu maupun skala anomali bernilai identik, sementara nilai $R^2$ berbeda karena penyebut varians pada skala suhu mencakup siklus musiman tahunan).*

---

## D. Kerangka Pemikiran (Kerangka Berpikir)
Kerangka pemikiran penelitian ini diilustrasikan pada Gambar 1 berikut:

```mermaid
flowchart TD
    A["<b>MASALAH PENELITIAN</b><br/>1. Diperlukan pemodelan prediktif anomali termal perairan Danau Maninjau berbasis GEE dengan pengendalian lookahead bias.<br/>2. Perlu diaudit representasi fisik grid ERA5-Land serta tingkat kesesuaiannya terhadap pengamatan satelit MODIS & Landsat."] --> B

    subgraph GEE ["EKSTRAKSI & AUDIT SPASIAL-OBSERVASIONAL GEE (2016–2025)"]
        B["<b>Ekstraksi Multi-Sumber pada ROI Inti Air (r=2 km, JRC > 90%)</b><br/>• Prediktor & Target Reanalisis: ERA5-Land (11,1 km) & CHIRPS (5,5 km)<br/>• Pembanding Satelit: MODIS Terra/Aqua (piksel valid, std, view angle, view time) & Landsat 8/9"]
        B --> B2["<b>Audit Grid ERA5-Land & Evaluasi Kesesuaian Satelit</b><br/>• Fraksi air JRC pada grid utama (52,20%) & stratifikasi FLake<br/>• Sensitivitas ERA5 vs MODIS Terra siang (≥15/22 piksel, homogenitas, near-nadir)<br/>• Penetapan era5_lake_mix_temp_c sebagai target reanalisis dengan batasan eksplisit"]
    end

    B2 --> D

    D["<b>PREPROCESSING PENGENDALIAN KEBOCORAN INFORMASI & ROLLING-ORIGIN CV</b><br/>1. Indeks Kalender Konsisten 365 Hari (cal_doy_365) Tanpa Pergeseran Tahun Kabisat<br/>2. Isolasi Klimatologi Hanya dari Data Latih Masing-Masing Fold & Pemisahan Kronologis Target<br/>3. Dua Rezim Fitur (≤ t): Skema Utama (73 Fitur) & Skema Ablasi Murni Meteo (55 Fitur)<br/>4. Tuning Terpisah via 3-Fold Rolling-Origin CV (2016–2023) & Uji Independen Sekali (2024–2025)"] --> E

    subgraph MODEL ["KOMPARASI MODEL REGRESI & DUA BASELINE PERSISTENCE"]
        E0["<b>Dua Baseline Persistence</b><br/>(Anomali Â_t+h = A_t & Suhu T̂_t+h = T_t)"]
        E1["<b>Random Forest Regressor</b>"]
        E2["<b>XGBoost Regressor</b>"]
        E3["<b>Support Vector Regression (SVR)</b>"]
    end

    D --> E0
    D --> E1
    D --> E2
    D --> E3

    E0 --> F
    E1 --> F
    E2 --> F
    E3 --> F

    C["<b>15 Kandidat Catatan Historis (2016–2025)</b><br/>Dipisahkan menurut Presisi Tanggal: Exact_Day (5 hari), Multi_Day, & Month_Only"] --> F

    F["<b>EVALUASI & ANALISIS EKSPLORATIF</b><br/>1. Komparasi MAE, RMSE (Metrik Utama), dan R² pada Data Uji Independen (2024–2025)<br/>2. Analisis Ketergantungan Autoregresif vs Kemampuan Murni Meteorologi (Feature Importance)<br/>3. Evaluasi Deskriptif Anomali Termal Reanalisis pada Periode Catatan Upwelling"] --> G

    G["<b>HASIL AKHIR (OUTPUT RISET)</b><br/>Komparasi Kinerja Algoritma pada Target Reanalisis ERA5-Land & Dokumentasi Ilmiah Keterbatasan Validasi Satelit/Upwelling"]
```

**Gambar 1. Bagan kerangka pemikiran penelitian komparasi algoritma regresi dan audit spasial-observasional di Danau Maninjau**
*Sumber: Hasil Perancangan Penelitian (2026)*

---

## E. Hipotesis Penelitian
Berdasarkan landasan teori dan kerangka pemikiran di atas, dirumuskan hipotesis penelitian sebagai berikut:
1. **Hipotesis Pertama ($H_1$):** Penerapan penyaringan kontrol kualitas bertingkat pada citra satelit *MODIS Terra* siang (ambang jumlah piksel air valid $\ge 15/22$, simpangan baku spasial $\le 0,5^\circ\text{C}$, dan sudut pandang sensor dekat nadir $\le 30^\circ$) dapat menurunkan selisih kesalahan absolut (*Mean Absolute Error*) terhadap suhu lapisan tercampur *ERA5-Land*, namun ketersediaan hari pengamatan yang tersisa sangat terbatas sehingga belum memadai sebagai target pelatihan harian kontinu.
2. **Hipotesis Kedua ($H_2$):** Algoritma *Random Forest*, *XGBoost*, dan *Support Vector Regression* (SVR) yang dilatih menggunakan *pipeline* dengan pengendalian kebocoran informasi masa depan melalui isolasi klimatologi dan pemisahan kronologis target pada Skema Utama (73 fitur termal-meteorologi $\le t$) mampu menghasilkan nilai RMSE yang lebih rendah dan $R^2$ yang lebih tinggi dibandingkan kedua model acuan *Baseline Persistence* ($\hat{A}_{t+h} = A_t$ dan $\hat{T}_{t+h} = T_t$) pada prediksi anomali suhu lapisan tercampur *ERA5-Land* horizon $t+1$ dan $t+3$ di periode uji independen (2024–2025).
3. **Hipotesis Ketiga ($H_3$):** Penghapusan fitur riwayat suhu danau pada Skema Ablasi Murni Meteorologi (55 fitur cuaca $\le t$) menyebabkan penurunan nilai $R^2$ secara substansial dibandingkan Skema Utama (73 fitur), yang menunjukkan dominannya informasi riwayat suhu dalam kemampuan prediksi jangka pendek.

---
---

# BAB III — METODOLOGI PENELITIAN

## A. Metode dan Tahapan Penelitian (Skema 1: Pengembangan Algoritma / Model)
Penelitian ini menggunakan metode penelitian kuantitatif eksperimental yang mengacu pada kerangka **Skema 1 (Pengembangan Algoritma / Model / Analisis)** pada buku panduan skripsi Fakultas Ilmu Komputer Universitas Djuanda. Tahapan metodologi penelitian disusun secara sistematis ke dalam 7 (tujuh) tahapan utama sebagaimana disajikan pada Tabel 2.

**Tabel 2. Tahapan Metodologi Penelitian Pengembangan Model (Skema 1)**

| Tahap | Kegiatan | Output |
| :--- | :--- | :--- |
| **1. Pengumpulan Data** | Ekstraksi data meteorologi dan reanalisis ERA5-Land, CHIRPS, MODIS, dan Landsat di GEE (2016–2025) beserta atribut kualitas spasial piksel dan penyusunan inventaris 15 kandidat catatan historis. | File CSV dataset harian, Landsat, audit grid ERA5, audit piksel MODIS, dan inventaris kejadian |
| **2. Analisis & Audit Data** | Audit fraksi air dan parameter *FLake* pada sel grid ERA5-Land serta analisis sensitivitas kesesuaian suhu ERA5 terhadap MODIS Terra/Aqua (ambang $\ge 15/22$ piksel, homogenitas, sudut pandang) dan Landsat. | Tabel audit grid ERA5 dan tabel evaluasi sensitivitas ERA5 vs MODIS/Landsat |
| **3. Preprocessing** | Pemetaan indeks kalender konsisten 365 hari (`cal_doy_365`), isolasi perhitungan klimatologi pada batas data latih, pembentukan 73 fitur (Skema Utama) dan 55 fitur (Skema Ablasi Meteorologi), serta pemisahan kronologis tanggal target di perbatasan *split*. | Matriks fitur dengan isolasi klimatologi dan pemisahan kronologis target untuk masing-masing horizon ($t+1$ dan $t+3$) |
| **4. Perancangan Model** | Perancangan *pipeline* regresi untuk dua *Baseline Persistence* ($\hat{A}_{t+h}=A_t$ dan $\hat{T}_{t+h}=T_t$), *Random Forest*, *XGBoost*, dan SVR beserta skema *3-Fold Rolling-Origin Cross-Validation* (ROCV). | Arsitektur pemodelan dan skema seleksi *hyperparameter* terpisah untuk kedua rezim fitur |
| **5. Implementasi** | Penulisan skrip ekstraksi dan audit GEE (`gee_extract_maninjau.js`, `gee_audit_grid_dan_modis.js`) serta implementasi komputasi Python (`audit_dan_eksperimen_ml.py`, `analisis_audit_lanjutan_gee.py`). | Kode program yang portabel dan dapat direproduksi |
| **6. Pengujian** | Eksekusi seleksi *hyperparameter* melalui 3-Fold ROCV (2016–2023) secara terpisah untuk Skema Utama (73 fitur) dan Skema Ablasi (55 fitur), dilanjutkan pengujian satu kali pada *Independent Test Set* (2024–2025). | Tabel hasil *tuning* ROCV dan tabel hasil evaluasi uji independen 2024–2025 |
| **7. Evaluasi** | Analisis perbandingan MAE, RMSE (metrik utama), dan $R^2$ terhadap kedua *Baseline Persistence*, analisis *feature importance*, pembahasan keterbatasan kesesuaian satelit, serta analisis deskriptif pada 3 kategori catatan *upwelling*. | Kesimpulan komparasi algoritma dan rekomendasi metodologis |

*Sumber: Hasil Perancangan Penelitian (2026)*

---

## B. Model dan Arsitektur yang Diusulkan
Model yang diusulkan dalam penelitian ini adalah arsitektur komparasi regresi deret waktu dua rezim (*dual-regime time-series regression pipeline*) yang memprediksi anomali suhu lapisan tercampur $\hat{A}_{t+h}$ untuk horizon $h \in \{1, 3\}$ hari ke depan, kemudian merekonstruksi estimasi suhu absolut reanalisis melalui relasi deterministik $\hat{T}_{t+h} = \hat{A}_{t+h} + \bar{T}_{\text{clim}, d(t+h)}$. Arsitektur ini menguji tiga keluarga algoritma regresi yang memiliki karakteristik induktif berbeda:
1. ***Random Forest Regressor* (RF):** Mewakili pendekatan *bagging* pohon keputusan paralel yang mereduksi varians melalui perataan prediksi antar-pohon independen.
2. ***Extreme Gradient Boosting Regressor* (XGBoost):** Mewakili pendekatan *boosting* pohon keputusan sekuensial dengan regularisasi $L_1/L_2$ yang secara bertahap mengoreksi kesalahan residual dari iterasi sebelumnya.
3. ***Support Vector Regression* (SVR) dengan Kernel RBF:** Mewakili pendekatan optimisasi margin berbasis fungsi kernel nonlinier pada ruang fitur berdimensi tinggi yang didahului oleh standardisasi fitur (`StandardScaler`) di dalam objek `Pipeline` untuk mencegah kebocoran statistik skala fitur dari data uji.

---

## C. Prosedur Pengembangan dan Implementasi

### 1. Pengumpulan Data dan Audit Spasial-Piksel di Google Earth Engine
Pengumpulan data dan audit kualitas dilakukan melalui dua skrip Google Earth Engine (`gee_extract_maninjau.js` dan `gee_audit_grid_dan_modis.js`):
1. **Penentuan Area Inti Perairan (*Core Water ROI*):**
   Titik pusat perairan Danau Maninjau ($[100,187^\circ\text{ BT}, -0,315^\circ\text{ LS}]$) diberi radius *buffer* $2.000\text{ meter}$ dan difilter dengan masker `JRC/GSW1_4/GlobalSurfaceWater` (`occurrence > 90%`).
2. **Audit Spasial Sel Grid ERA5-Land (`maninjau_audit_grid_era5.csv`):**
   Memetakan poligon sel grid asli `ECMWF/ERA5_LAND/DAILY_AGGR` ($\approx 0,1^\circ \times 0,1^\circ$) di sekitar kaldera Maninjau dan menghitung luas irisan terhadap ROI Inti, persentase fraksi air permanen JRC resolusi $30\text{ m}$ di dalam sel grid, elevasi rata-rata SRTM, serta nilai rata-rata multi-tahun parameter *FLake* (`lake_mix_layer_temperature`, `lake_bottom_temperature`, `lake_mix_layer_depth`, `skin_temperature`, dan `temperature_2m`).
3. **Ekstraksi dan Audit Kualitas Piksel Satelit (`maninjau_audit_modis_pixels_2016_2025.csv` & `maninjau_landsat_lst_2016_2025.csv`):**
   * **MODIS Terra (`MOD11A1`) & Aqua (`MYD11A1`):** Menyaring band `QC_Day` dan `QC_Night` (*Mandatory QA* $\le 1$ dan *LST error* $\le 2\text{ K}$) pada piksel air JRC di dalam ROI (maksimum 22 piksel beresolusi $\approx 926,6\text{ m}$), serta mengekstrak rata-rata suhu (`mean_c`), simpangan baku spasial antar-piksel (`std_c`), nilai minimum/maksimum (`min_c`, `max_c`), **jumlah piksel air valid (`valid_pixels`)**, **waktu pengamatan matahari lokal (`view_time_hr`)**, dan **sudut pandang zenit sensor (`view_angle_deg`)**.
   * **Landsat 8/9 Level-2 (`scale: 30` m):** Menyaring awan/bayangan awan melalui `QA_PIXEL`, ketidakpastian `ST_QA < 3,0 K`, dan jumlah piksel air valid $\ge 100$ piksel.

### 2. Evaluasi Kesesuaian ERA5–MODIS dan Penetapan Target Reanalisis
Sebelum pemodelan dilakukan, dijalankan evaluasi khusus kesesuaian suhu harian `era5_lake_mix_temp_c` terhadap pengamatan satelit—khususnya **MODIS Terra Siang** yang memiliki ketersediaan tertinggi—dengan menerapkan ambang jumlah piksel valid ($\ge 1, \ge 5, \ge 10, \ge 15, \ge 18, \ge 20$, hingga $22/22$ piksel), keseragaman spasial (`std_c <= 0,5°C`), dan sudut pandang dekat nadir (`|view_angle_deg| <= 30°`). Ambang $\ge 15$ dari 22 piksel digunakan sebagai analisis sensitivitas kualitas cakupan. Mengingat pengamatan satelit yang memenuhi kriteria kualitas ketat bersifat tidak kontinu (hanya mencakup sebagian kecil hari dalam 10 tahun) serta mengukur besaran fisik yang berbeda (suhu kulit permukaan sesaat pada jam lintas satelit vs suhu lapisan tercampur rata-rata 24 jam), maka **target pemodelan harian ditetapkan pada keluaran reanalisis `era5_lake_mix_temp_c`**, sedangkan hasil audit satelit didokumentasikan sebagai batasan validitas observasional.

### 3. Preprocessing dengan Pengendalian Kebocoran Informasi Masa Depan dan Rincian Konstruksi Fitur (73 vs 55 Fitur)
Tahap *preprocessing* dirancang di dalam `audit_dan_eksperimen_ml.py` melalui *pipeline* dengan pengendalian kebocoran informasi masa depan (*lookahead bias*) melalui isolasi klimatologi dan pemisahan kronologis target:
1. **Indeks Kalender Konsisten 365 Hari (`cal_doy_365`):**
   Setiap tanggal dipetakan berdasarkan pasangan `(bulan, hari)` ke kalender acuan non-kabisat 365 hari (di mana tanggal 29 Februari pada tahun kabisat 2016, 2020, dan 2024 dipetakan ke indeks 59 bersama 28 Februari). Dengan demikian, tanggal 1 Maret selalu berada pada indeks 60 dan 31 Desember selalu pada indeks 365 di seluruh tahun tanpa pergeseran pasca-Februari.
2. **Isolasi Klimatologi Berdasarkan Batas Akhir Pelatihan (`clim_end_date`):**
   Nilai klimatologi harian ($\bar{T}_{\text{clim}, d}$, dihaluskan dengan jendela bergerak melingkar 15 hari pada indeks $1..365$) dihitung **hanya dari baris data di dalam periode pelatihan aktif** (`date <= clim_end_date`). Pada saat validasi silang, klimatologi tidak pernah menyertakan data dari periode validasi maupun data uji; pada saat pengujian final (2024–2025), klimatologi dihitung hanya dari periode pra-uji (2016–2023).
3. **Pemotongan Tanggal Target di Perbatasan *Split* (*Boundary Target Isolation*):**
   Untuk horizon prediksi $h \in \{1, 3\}$, setiap baris pengamatan pada tanggal fitur $t$ memiliki tanggal target $t_{\text{target}} = t + h$. Agar target pada akhir periode pelatihan tidak menyeberang ke periode validasi atau pengujian, sampel pelatihan diwajibkan memenuhi syarat ganda: $t \le t_{\text{akhir\_latih}}$ **dan** $t + h \le t_{\text{akhir\_latih}}$.
4. **Konstruksi Dua Rezim Fitur Input ($\mathbf{X}_{\le t}$: 73 Fitur vs 55 Fitur):**
   Sesuai implementasi fungsi `build_leak_free_dataset()` pada `audit_dan_eksperimen_ml.py`, rincian komposisi fitur disajikan secara eksak pada Tabel 3.
   * **Rezim 1 — Skema Utama (Termal Historis + Meteorologi $\le t$, total 73 fitur):** Menggabungkan **18 fitur autoregresif termal danau (`ar_cols`)** dan **55 fitur kalender serta meteorologi (`feature_cols_meteo`)**.
   * **Rezim 2 — Skema Ablasi Murni Meteorologi ($\le t$, total 55 fitur):** Menghapus seluruh 18 fitur suhu dan anomali danau masa lalu (`ar_cols`), sehingga model hanya menggunakan **55 fitur kalender dan riwayat meteorologi (`feature_cols_meteo`)** untuk menguji seberapa besar kemampuan prediksi murni dari variabel pemaksa atmosfer.

**Tabel 3. Rincian Eksak Konstruksi Fitur Input pada Skema Utama (73 Fitur) dan Skema Ablasi Murni Meteorologi (55 Fitur)**

| Kelompok Fitur | Nama Kolom Fitur dalam Kode (`audit_dan_eksperimen_ml.py`) | Jumlah Fitur | Skema Utama (Rezim 1) | Skema Ablasi Meteo (Rezim 2) |
| :--- | :--- | :-: | :-: | :-: |
| **1. Termal Hari-$t$** | `era5_lake_mix_temp_c`, `anom_doy_c` | 2 | $\checkmark$ | — |
| **2. *Lag* Termal Danau** | `lake_temp_lag{1,2,3,7}`, `anom_doy_lag{1,2,3,7}` | 8 | $\checkmark$ | — |
| **3. *Rolling* Termal Danau** | `lake_temp_mean{3,7,14}d`, `lake_temp_std{3,7,14}d` | 6 | $\checkmark$ | — |
| **4. Laju Perubahan Termal** | `lake_temp_diff1d`, `lake_temp_diff3d` | 2 | $\checkmark$ | — |
| *Subtotal Fitur Termal (`ar_cols`)* | *Seluruh fitur riwayat suhu & anomali danau ($\le t$)* | **18** | **18 Fitur** | **0 Fitur** |
| **5. Kalender Siklik** | `sin_doy`, `cos_doy` | 2 | $\checkmark$ | $\checkmark$ |
| **6. Meteorologi Hari-$t$** | `air_temp_mean_c`, `air_temp_min_c`, `air_temp_max_c`, `u_wind_10m_ms`, `v_wind_10m_ms`, `wind_speed_10m`, `solar_net_rad_mjm2`, `thermal_net_rad_mjm2`, `surface_pressure_hpa`, `precip_era5_mm`, `precip_chirps_mm` | 11 | $\checkmark$ | $\checkmark$ |
| **7. *Lag* Meteorologi** | *Lag* $\{1, 2, 3, 7\}$ hari untuk 6 variabel utama (`air_temp_mean_c`, `wind_speed_10m`, `solar_net_rad_mjm2`, `thermal_net_rad_mjm2`, `precip_era5_mm`, `precip_chirps_mm`) ($6 \times 4$) | 24 | $\checkmark$ | $\checkmark$ |
| **8. *Rolling* Meteorologi** | Rata-rata bergulir (`mean{3,7,14}d`) untuk 4 variabel cuaca kontinu dan akumulasi bergulir (`sum{3,7,14}d`) untuk 2 variabel curah hujan ($6 \times 3$) | 18 | $\checkmark$ | $\checkmark$ |
| *Subtotal Fitur Meteo (`feature_cols_meteo`)* | *Seluruh fitur kalender & riwayat meteorologi ($\le t$)* | **55** | **55 Fitur** | **55 Fitur** |
| **TOTAL FITUR INPUT** | **`len(feature_cols_full)` vs `len(feature_cols_meteo)`** | — | **73 Fitur** | **55 Fitur** |

*Sumber: Hasil Implementasi Pipa Fitur `audit_dan_eksperimen_ml.py` (2026)*

---

## D. Desain Uji Coba dan Evaluasi Model

### 1. Desain Uji Coba: *3-Fold Rolling-Origin Cross-Validation* (ROCV) dan Uji Independen
Pemilihan *hyperparameter* dilakukan menggunakan skema **3-Fold Expanding-Window / Rolling-Origin Cross-Validation (ROCV)** pada periode pra-uji (2016–2023):
* **Fold 1:** Latih `2016–2019` (Klimatologi $\le$ `2019-12-31`, $t+h \le$ `2019-12-31`) $\rightarrow$ Validasi `2020–2021` ($t+h \le$ `2021-12-31`).
* **Fold 2:** Latih `2016–2020` (Klimatologi $\le$ `2020-12-31`, $t+h \le$ `2020-12-31`) $\rightarrow$ Validasi `2021–2022` ($t+h \le$ `2022-12-31`).
* **Fold 3:** Latih `2016–2021` (Klimatologi $\le$ `2021-12-31`, $t+h \le$ `2021-12-31`) $\rightarrow$ Validasi `2022–2023` ($t+h \le$ `2023-12-31`).

Perlu dicatat secara metodologis bahwa ketiga *fold* validasi di atas menggunakan jendela evaluasi 2 tahun yang bergeser per 1 tahun (sehingga tahun 2021 dan 2022 beririsan antar-*fold* yang berurutan); oleh karena itu, metrik antar-*fold* tidak sepenuhnya saling bebas secara statistik, melainkan berfungsi untuk memilih konfigurasi *hyperparameter* dengan rata-rata `ROCV_Mean_RMSE` paling stabil sebelum diuji satu kali pada **Data Uji Independen (`2024–2025`)**. Proses pencarian *hyperparameter* ROCV ini dijalankan **secara terpisah** untuk Skema Utama (73 fitur) dan Skema Ablasi Murni Meteorologi (55 fitur). Selain itu, hasil prediksi pada Fold 3 (`2022–2023`) disimpan terpisah sebagai *Validation Set* dan tidak digabungkan sebagai hasil uji independen.

### 2. Subjek/Objek Uji Coba dan Pembagian Data
Objek uji coba dalam penelitian ini adalah deret waktu harian multi-sumber di kawasan perairan Danau Maninjau sepanjang 10 tahun (1 Januari 2016 – 31 Desember 2025, 3.653 hari). Setelah pemotongan 14 hari pertama pada awal tahun 2016 untuk pembentukan jendela bergulir 14 hari (`rolling 14d`) serta pemotongan tanggal target di perbatasan *split*, jumlah sampel pada pengujian final adalah:
* **Horizon $t+1$:** Data Latih Pra-Uji (`2016–2023`) sebanyak $2.907$ hari dan Data Uji Independen (`2024–2025`) sebanyak $730$ hari.
* **Horizon $t+3$:** Data Latih Pra-Uji (`2016–2023`) sebanyak $2.905$ hari dan Data Uji Independen (`2024–2025`) sebanyak $728$ hari.

### 3. Jenis Data dan Instrumen Pengumpulan Data
Penelitian ini menggunakan data kuantitatif sekunder yang terdiri atas: (1) data deret waktu reanalisis dan penginderaan jauh yang diekstrak melalui instrumen *Application Programming Interface* (API) *Google Earth Engine* menggunakan bahasa JavaScript, serta (2) data inventaris 15 kandidat catatan historis kejadian *upwelling* / kematian ikan massal di Danau Maninjau (2016–2025) yang dikompilasi dari publikasi ilmiah BRIN (Putri dkk., 2024) dan laporan resmi DKPP Kabupaten Agam pada Kantor Berita ANTARA (ANTARA, 2021a, 2021b, 2022, 2023, 2025a, 2025b, 2025c) beserta tautan sumber per kejadian pada berkas `inventaris_upwelling_maninjau_2016_2025.csv`.

### 4. Teknik Analisis Data dan Evaluasi Eksploratif Catatan Historis
Analisis data dilakukan melalui tiga tahapan kuantitatif:
1. **Analisis Statistik Kesesuaian Satelit:** Menghitung bias rata-rata, MAE, RMSE, korelasi linier Pearson ($r$), dan korelasi peringkat Spearman ($\rho$) antara `era5_lake_mix_temp_c` dan pengamatan satelit pada setiap tingkatan filter kualitas piksel.
2. **Analisis Komparasi Regresi:** Membandingkan nilai MAE, RMSE, dan $R^2$ (skala anomali dan skala suhu) antara *Random Forest*, *XGBoost*, dan SVR terhadap kedua *Baseline Persistence* pada data uji independen 2024–2025, dengan menetapkan **RMSE terendah dan $R^2$ tertinggi sebagai kriteria utama penentu model terbaik**.
3. **Evaluasi Deskriptif-Eksploratif terhadap Catatan Kejadian Historis:** Analisis terhadap 15 kandidat catatan di `inventaris_upwelling_maninjau_2016_2025.csv` dilakukan secara deskriptif dengan menyeragamkan acuan klimatologi antara seri observasi ERA5 dan seri prediksi pada masing-masing rezim waktu, serta memisahkan tabel analisis ke dalam tiga kelompok durasi yang tidak dicampur:
   * **Kelompok 1 (`Exact_Day`, $N=10$ catatan):** Catatan yang memiliki tanggal laporan harian spesifik, dianalisis pada tanggal laporan ($t_0$) dan jendela seragam 5 hari ($[t_0 - 3, t_0 + 1]$) menggunakan nilai rata-rata jendela (`mean`) dan nilai minimum jendela (`min`).
   * **Kelompok 2 (`Multi_Day_Episode` & `Multi_Day_Observation`, $N=3$ catatan):** Episode yang berlangsung selama beberapa hari (4 hingga 24 hari), dianalisis sepanjang rentang tanggal episode tanpa diberlakukan jendela $H-3$.
   * **Kelompok 3 (`Month_Only`, $N=2$ catatan):** Catatan sekunder yang hanya menyebutkan bulan kejadian (Maret 2017 dan Februari 2018), disajikan terpisah sebagai statistik bulanan dan dikecualikan dari analisis kejadian harian.

---

## E. Jadwal Penelitian (Penjadwalan)
Pelaksanaan penelitian dirancang selama 1 (satu) semester (4 bulan efektif) sebagaimana disajikan pada Tabel 4.

**Tabel 4. Jadwal Pelaksanaan Penelitian Skripsi**

| No | Kegiatan Penelitian | Bulan 1 | Bulan 2 | Bulan 3 | Bulan 4 |
| :-: | :--- | :-: | :-: | :-: | :-: |
| 1 | Studi literatur, perumusan masalah, dan penyusunan Proposal Skripsi | $\blacksquare\blacksquare\blacksquare\blacksquare$ | | | |
| 2 | Ekstraksi dataset harian GEE serta audit spasial grid ERA5-Land dan piksel MODIS/Landsat | | $\blacksquare\blacksquare$ | | |
| 3 | Perancangan *pipeline* dengan pengendalian *lookahead bias* (73 & 55 fitur) dan *tuning* 3-Fold ROCV | | $\blacksquare\blacksquare$ | $\blacksquare$ | |
| 4 | Pengujian pada *Independent Test Set* (2024–2025), analisis *feature importance*, dan evaluasi eksploratif | | | $\blacksquare\blacksquare\blacksquare$ | |
| 5 | Penyusunan laporan akhir (Bab IV–V), pembuatan grafik, dan sidang Skripsi | | | | $\blacksquare\blacksquare\blacksquare\blacksquare$ |

*Sumber: Rencana Kerja Penelitian (2026)*

---
---

# DAFTAR RUJUKAN

**A. Pustaka Jurnal Ilmiah Primer (11 Artikel)**

Aldiansyah, S. dan Saputra, R. A. (2022). Comparison of Machine Learning Algorithms for Land Use and Land Cover Analysis Using Google Earth Engine (Case Study: Wanggu Watershed). *International Journal of Remote Sensing and Earth Sciences (IJReSES)*, Vol. 19(2), hal. 197–210. DOI: `10.30536/ijreses.v19i2.13716`.

Fajri, M., Kencana, M. R., Putra, W. M., dan Thoriq, M. (2026). Perbandingan Algoritma Random Forest dan XGBoost dalam Klasifikasi Kualitas Air Danau Maninjau. *Jurnal Nasional Teknologi dan Sistem Informasi (TEKNOSI)*, Vol. 12(2), hal. 276–283. DOI: `10.25077/TEKNOSI.v12i2.2026.276-283`.

Hamze-Ziabari, S. M., Lemmin, U., Foroughan, M., Reiss, R. S., dan Barry, D. A. (2023). Chimney-Like Intense Pelagic Upwelling in the Center of Basin-Scale Cyclonic Gyres in Large Lake Geneva. *Journal of Geophysical Research: Oceans*, Vol. 128(7), e2022JC019592. DOI: `10.1029/2022JC019592`.

Komala, P. S., Zulkarnaini, Z., Kurniati, R. I., dan Fauzi, M. (2024). The Effect of Anthropogenic Activities on the Spatial Distribution of Total Nitrogen and Total Phosphate in Lake Maninjau. *Journal of Water and Land Development*, No. 60, hal. 71–78. DOI: `10.24425/jwld.2024.149109`.

Li, Z., Zhang, Z., Xiong, S., Zhang, W., dan Li, R. (2024). Lake Surface Temperature Predictions under Different Climate Scenarios with Machine Learning Methods: A Case Study of Qinghai Lake and Hulun Lake, China. *Remote Sensing*, Vol. 16(17), 3220, hal. 1–22. DOI: `10.3390/rs16173220`.

Naumenko, M. A., Guzivaty, V. V., Budnev, N. M., dan Troitskaya, E. S. (2025). Spatiotemporal Variability of the Thermal Structure of Upwelling in the Southwestern Part of Lake Baikal. *Izvestiya, Atmospheric and Oceanic Physics*, Vol. 61(11), hal. 1446–1454. DOI: `10.1134/S0001433826700210`.

Putri, M. R., Jasalesmana, T., Abdurrachman, M., Henny, C., Nomosatryo, S., dan Albani, A. S. (2024). The Impact of Weather Condition Changes on Vertical Distribution of Sulfides in Lake Maninjau Based on Observation Data. *LIMNOTEK: Perairan Darat Tropis di Indonesia*, Vol. 30(1), hal. 1–11. DOI: `10.55981/limnotek.2024.2203`.

Sunaryani, A. (2023). Penentuan Status Mutu Air dan Status Trofik di Perairan Danau Maninjau. *Jurnal Teknologi Lingkungan*, Vol. 24(1), hal. 21–27. DOI: `10.55981/jtl.2023.246`.

Syakir, F., Irhamsyah, M., Melinda, Yunidar, Zulhelmi, dan Miftahujjannah, R. (2025). Forecasting Upwelling in Lake Maninjau Using Vector Autoregressive, Support Vector Machine and Dashboard Visualization. *JITK (Jurnal Ilmu Pengetahuan dan Teknologi Komputer)*, Vol. 11(2), hal. 580–590. DOI: `10.33480/jitk.v11i2.6665`.

Ulhaq, M. Z., Farid, M., Aziza, Z. I., Nuzullah, T. M. F., Syakir, F., dan Sasmita, N. R. (2025). Integration of Machine Learning and Time Series Analysis for Upwelling Prediction Dashboard in Lake Laut Tawar, Indonesia: A Study Based on Climate Forecasting. *Theoretical and Applied Climatology*, Vol. 156(9), 463. DOI: `10.1007/s00704-025-05717-3`.

Valbuena, S. A., Bombardelli, F. A., Cortés, A., Largier, J. L., Roberts, D. C., Forrest, A. L., dan Schladow, S. G. (2022). 3D Flow Structures During Upwelling Events in Lakes of Moderate Size. *Water Resources Research*, Vol. 58(3), e2021WR030666. DOI: `10.1029/2021WR030666`.

**B. Sumber Laporan Kejadian Sekunder (Berita Resmi LKBN ANTARA / DKPP Kabupaten Agam)**

ANTARA. (2021a, 27 April). *10 ton ikan di Danau Maninjau mati massal*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/2124642/10-ton-ikan-di-danau-maninjau-mati-massal`.

ANTARA. (2021b, 27 Desember). *Kurang oksigen, jumlah ikan mati di Danau Maninjau menjadi 1.455 ton*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/2609237/kurang-oksigen-jumlah-ikan-mati-di-danau-maninjau-menjadi-1455-ton`.

ANTARA. (2022, 15 Februari). *Ikan mati di Danau Maninjau bertambah, petani rugi hingga Rp2,6 miliar*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/2703957/ikan-mati-di-danau-maninjau-bertambah-petani-rugi-hingga-rp26-miliar`.

ANTARA. (2023, 16 Mei). *Pemkab Agam: 15 ton ikan di Danau Maninjau mati akibat angin kencang*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/3542046/pemkab-agam-15-ton-ikan-di-danau-maninjau-mati-akibat-angin-kencang`.

ANTARA. (2025a, 19 Januari). *Kematian ikan di Danau Maninjau bertambah 50 ton*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/4594238/kematian-ikan-di-danau-maninjau-bertambah-50-ton`.

ANTARA. (2025b, 27 November). *Ikan Danau Maninjau mati capai delapan ton dampak hujan-angin kencang*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/5270801/ikan-danau-maninjau-mati-capai-delapan-ton-dampak-hujan-angin-kencang`.

ANTARA. (2025c, 20 Desember). *Sekitar 1.428,73 ton ikan mati dampak upwelling air Danau Maninjau*. Lembaga Kantor Berita Nasional (LKBN) ANTARA (Laporan DKPP Kabupaten Agam). Tersedia pada: `https://www.antaranews.com/berita/5314777/sekitar-142873-ton-ikan-mati-dampak-upwelling-air-danau-maninjau`.
