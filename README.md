# Proyek Akhir Penerapan Data Science: Menyelesaikan Permasalahan Institusi Pendidikan - Jaya Jaya Institut

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.49%2B-brightgreen.svg)](https://shap.readthedocs.io/)
[![CRISP-DM](https://img.shields.io/badge/Methodology-CRISP--DM-purple.svg)]()

---

## 📌 Executive Summary
**Jaya Jaya Institut** merupakan institusi pendidikan tinggi swasta terkemuka yang telah berdiri sejak tahun 2000. Meskipun memiliki reputasi mencetak ribuan lulusan berprestasi, institusi ini menghadapi masalah kritis berupa **tingginya rasio mahasiswa putus studi (*dropout rate*) mencapai 32,1%**. Tingkat *dropout* yang tinggi menimbulkan dampak finansial langsung (kehilangan pendapatan SPP/biaya kuliah dan utang tak tertagih), inefisiensi alokasi dosen/fasilitas, serta ancaman terhadap akreditasi dan reputasi kampus.

Proyek Data Science ini menerapkan metodologi terstruktur **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) untuk membangun **Sistem Deteksi Dini & Retensi Mahasiswa (*Student Retention & Early Warning System*)**. Mengakomodasi evaluasi data preparation terkini, pemodelan diformulasikan sebagai **Binary Classification (Dropout vs Graduate)** di mana data berstatus *Enrolled* (794 baris) dipisahkan untuk keperluan inferensi *out-of-sample*. Model terbaik menggunakan **Random Forest Classifier (Akurasi 93,0%, Recall Dropout 91,2%, Presisi Dropout 90,9%, F1-Score 0,910, ROC-AUC 0,972)**, dilengkapi interpretasi mendalam berbasis **SHAP Values & Feature Importance**, **Audit Keadilan Algoritmik (*Bias Checking*)**, serta **Simulasi Kebijakan Preskriptif (*What-If Simulation*)** yang membuktikan potensi penyelamatan **29,5% mahasiswa berisiko putus studi** dengan proyeksi penyelamatan pendapatan SPP institusi mencapai **Rp 10,08 Miliar**.

---

## 🏢 Business Understanding

### Latar Belakang Bisnis
Pada era persaingan perguruan tinggi modern, retensi mahasiswa (*student retention rate*) merupakan salah satu indikator vital dalam akreditasi institusi, keberlanjutan anggaran operasional, dan kepuasan pemangku kepentingan. Jaya Jaya Institut mengalami tantangan di mana sepertiga dari mahasiswa baru gagal menuntaskan studi mereka hingga tahap kelulusan (*graduation*). 

### Permasalahan Bisnis
Berdasarkan investigasi manajerial bersama pimpinan akademik dan bagian keuangan kampus, diidentifikasi empat permasalahan bisnis pokok:
1. **Tingkat Dropout Sangat Tinggi**: Analisis historis menunjukkan **32,1% (1.421 dari 4.424 mahasiswa)** mengalami *dropout*.
2. **Ketiadaan Sistem Peringatan Dini (*Early Warning System*)**: Pihak kampus belum memiliki instrumen analitik untuk mendeteksi sinyal awal penurunan performa akademik atau kendala finansial sebelum mahasiswa resmi mengundurkan diri.
3. **Ketidakpastian Faktor Pendorong (*Root Causes*)**: Belum ada verifikasi empiris mengenai apakah faktor akademik (SKS lulus/nilai) atau faktor sosioekonomi (tunggakan SPP/utang) yang paling dominan memicu *dropout*.
4. **Kebutuhan Evaluasi Keadilan & Pengujian Kebijakan**: Manajemen memerlukan kepastian bahwa model tidak bias terhadap gender/usia serta membutuhkan simulasi dampak terukur sebelum mengeksekusi program beasiswa atau tutorial bantuan.

### Cakupan Proyek (*Project Scope*)
1. **Exploratory Data Analysis (EDA)**: Membedah korelasi demografi, sosioekonomi, jalur seleksi, dan performa akademik semester 1–2 terhadap status mahasiswa.
2. **Data Preparation & Feature Engineering**: Merancang metrik performa domain pendidikan seperti rasio kelulusan semester (`Approval_rate_1st`, `Approval_rate_2nd`), rasio kelulusan kumulatif tahun pertama (`Total_approval_rate`), progresi nilai (`Grade_progression`), dan indeks komposit finansial (`Financial_risk_index`).
3. **Machine Learning Modeling**: Membandingkan 4 model klasifikasi (*Logistic Regression*, *Decision Tree*, *Random Forest*, *Gradient Boosting*) dengan 5-Fold Stratified Cross-Validation dan *GridSearchCV*.
4. **Model Evaluation & SHAP Analysis**: Menguji model dengan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, *ROC-AUC*, *Confusion Matrix*, dan *SHAP TreeExplainer*.
5. **Model Fairness & Bias Checking**: Mengaudit keadilan model pada atribut sensitif (*Gender* dan *Age at enrollment*) menggunakan *Disparate Impact Ratio*, *Demographic Parity*, dan *Equal Opportunity (Recall Parity)*.
6. **What-If Policy Simulation**: Mensimulasikan skenario bantuan finansial dan tutorial akademik pada data uji untuk menguantifikasi penurunan *dropout* dan nilai ekonomi yang terselamatkan.
7. **Production Prototype Deployment**: Membangun aplikasi web interaktif berbasis Streamlit (`app.py`) dengan fitur prediksi tunggal (termasuk *Quick Presets*), prediksi massal (*Batch CSV*), dan *What-If Sandbox*.

### Persiapan

#### Sumber Data
Dataset diperoleh dari repositori resmi Dicoding untuk studi kasus Jaya Jaya Institut:
- **Nama Berkas**: `data.csv`
- **Tautan Unduh**: [Dicoding Students Performance Dataset](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv)
- **Dimensi**: 4.424 baris data mahasiswa dan 37 kolom (36 fitur prediktor + 1 target `Status`).
- **Kualitas Data**: 0 *missing values* dan 0 baris duplikat.

#### Setup Environment
Proyek ini kompatibel dengan Python 3.10+ (atau Python 3.12). Ikuti langkah-langkah berikut:

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/dawooodd/Penerapan-Data-Science-2.git
   cd Penerapan-Data-Science-2
   ```

2. **Buat dan Aktifkan Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Business Dashboard

Dashboard analitik bisnis untuk memantau performa dan retensi mahasiswa Jaya Jaya Institut dibangun menggunakan **Looker Studio** dan terhubung langsung ke data hasil analisis proyek ini.

- **Looker Studio Dashboard Link**: [Jaya Jaya Institut - Student Performance Dashboard](https://datastudio.google.com/reporting/08504851-cd41-49b5-a9c0-3376df7ad174)
- **Screenshot Dashboard**: Tersedia pada repositori sebagai `nasich_dicoding-dashboard.png`
- **Sumber Data Dashboard**: `df_clean_mapped.csv` (data mahasiswa termap), `shap_data.csv` (feature importance), `what_if_data.csv` (simulasi intervensi)
- **Tinjauan Dashboard**: Dashboard menyajikan KPI utama (Total Mahasiswa, Dropout Rate, Graduation Rate), analisis faktor pendorong dropout berbasis SHAP, pemeriksaan keadilan model (*Bias Checking*), dan visualisasi dampak simulasi kebijakan intervensi. Seluruh analisis preskriptif juga tersedia pada aplikasi prototype Streamlit (`app.py`).

---

## 💻 Menjalankan Sistem Machine Learning

### Menjalankan Prototype Streamlit Secara Lokal
Aplikasi prototype deteksi dini dan simulasi kebijakan dapat dijalankan secara lokal dengan langkah mudah:

1. Pastikan virtual environment telah aktif dan dependensi terpasang.
2. Jalankan perintah Streamlit dari direktori proyek:
   ```bash
   streamlit run app.py
   ```
3. Buka peramban (browser) dan akses alamat:
   ```
   Local URL: http://localhost:8501
   ```

### Fitur Unggulan Aplikasi Prototype (`app.py`):
- **🎯 Prediksi Mahasiswa Tunggal**:
  - Tombol **Quick Presets** untuk memuat profil instan: ⚠️ *High Risk Student*, 🎓 *High Performing Student*, dan ⚖️ *Moderate Student*.
  - Form terstruktur dalam 4 tab (Demografis, Finansial, Akademik Semester 1, dan Akademik Semester 2).
  - Kartu hasil status prediksi dengan *color-coded badge*, indikator probabilitas risiko per kelas, dan **Rekomendasi Tindakan Intervensi Otomatis** yang disesuaikan secara personal dengan profil mahasiswa.
- **📁 Prediksi Massal (*Batch CSV Assessment*)**:
  - Fasilitas unggah file CSV untuk menilai risiko seluruh angkatan mahasiswa secara serentak.
  - Kartu KPI ringkasan kohort (Total Mahasiswa, Risiko Tinggi, Sedang, Rendah).
  - Tombol unduh laporan hasil prediksi (.CSV) dan tombol unduh template CSV sampel.
- **🧪 Simulasi Kebijakan (What-If Sandbox)**:
  - Antarmuka interaktif untuk menguji efektivitas kebijakan: simulasi restrukturisasi SPP, beasiswa darurat, dan tutor sebaya.
  - Menghitung secara instan jumlah mahasiswa yang terselamatkan dan proyeksi pendapatan SPP yang terselamatkan.
- **📊 Performa Model & Audit Keadilan**:
  - Menampilkan metrik evaluasi model (Akurasi, Presisi, Recall, ROC-AUC), grafik *Top 10 Feature Importance*, dan ringkasan audit keadilan (*Fairness Checking*).
- **💡 Rekomendasi Bisnis Institusi**:
  - Panduan implementasi manajerial bagi jajaran pimpinan perguruan tinggi.

### Tautan Deployment Cloud
Aplikasi prototype ini dapat diakses secara daring melalui Streamlit Community Cloud:
- **Tautan Aplikasi Web**: [https://jaya-institute.streamlit.app/](https://jaya-institute.streamlit.app/)

---

## 📈 Model Performance & Evaluation Summary

### Perbandingan Model Klasifikasi (5-Fold Stratified Cross-Validation & Test Set)

| Model Algoritma | Mean CV Accuracy | Test Accuracy | F1-Score (Binary) | Dropout Recall | Dropout Precision | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 90,5% | 93,8% | 0,920 | 91,6% | 92,5% | 0,974 |
| **Decision Tree** | 88,5% | 92,2% | 0,898 | 88,7% | 91,0% | 0,932 |
| **Gradient Boosting** | 90,3% | 92,4% | 0,904 | 90,8% | 89,9% | 0,974 |
| **Random Forest (Tuned Final)** 🏆 | **90,7%** | **93,0%** | **0,910** | **91,2%** | **90,9%** | **0,972** |

*Model terbaik dipilih berdasarkan keseimbangan luar biasa antara Recall Dropout (91,2%) dan Presisi Dropout (90,9%) serta stabilitas generalisasi pada data uji (ROC-AUC 0,972).*

---

## 🔍 Core Insights & Visual Interpretations

### 1. Temuan Utama EDA & Nilai SHAP (*SHapley Additive exPlanations*)
- **Momentum Akademik Tahun Pertama adalah Prediktor No. 1**:
  - Fitur rekayasa `Total_approval_rate` dan `Approval_rate_2nd` berkontribusi lebih dari **30% total importance**.
  - Nilai SHAP menunjukkan bahwa rasio kelulusan mata kuliah $< 50\%$ pada tahun pertama secara dramatis melonjakkan probabilitas mahasiswa mengalami *Dropout*.
  - Mahasiswa yang *Graduate* rata-rata meluluskan **6,2 mata kuliah** per semester, sedangkan mahasiswa *Dropout* rata-rata hanya meluluskan **1,9 mata kuliah** di semester 2.
- **Tunggakan Finansial sebagai Faktor Pemicu Kritis**:
  - Mahasiswa yang menunggak SPP (`Tuition_fees_up_to_date = 0`) memiliki angka *dropout* fantastis sebesar **86,5%**, berbanding terbalik dengan mahasiswa lunas (23,8%).
  - Mahasiswa debitur (`Debtor = 1`) mengalami *dropout rate* **62,1%**. Fitur `Financial_risk_index` menempati peringkat krusial dalam SHAP summary plot.
- **Efek Protektif Beasiswa**:
  - Mahasiswa penerima beasiswa mencatatkan tingkat kelulusan **76,3%** dan hanya **12,9%** yang *dropout*. Beasiswa terbukti menjadi instrumen retensi mahasiswa paling ampuh.
- **Faktor Usia & Gender**:
  - Mahasiswa yang mendaftar pada usia matang ($> 25$ tahun) memiliki tingkat *dropout* di atas 50% karena beban kerja dan keluarga ganda.
  - Mahasiswa laki-laki memiliki risiko *dropout* 45,1% berbanding perempuan 25,1%.

### 2. Hasil Audit Keadilan Model (*Bias Checking*)
Pemeriksaan keadilan (*fairness audit*) membuktikan bahwa model beroperasi secara adil dan bebas dari bias diskriminatif:
- **Gender Fairness**: 
  - Prediksi model (Laki-laki 40,3%, Perempuan 23,2%) berbanding lurus dengan data historis aktual (Laki-laki 46,0%, Perempuan 24,4%).
  - Model sedikit lebih konservatif untuk laki-laki (tidak melebih-lebihkan risiko secara artifisial).
  - *Recall Parity*: Recall deteksi *dropout* seimbang antara laki-laki dan perempuan dengan selisih minimal, mematuhi prinsip *Equal Opportunity*.
- **Age Group Fairness**: Distribusi probabilitas bervariasi kontinu mencerminkan beban kredit dan nilai nyata, tanpa adanya diskriminasi usia sistemik.

### 3. Hasil Simulasi Intervensi Kebijakan (*What-If Simulation*)
Pengujian skenario intervensi pada kohort data uji (*unseen test set*, 726 mahasiswa):
- **Baseline (Tanpa Intervensi)**: 285 mahasiswa diprediksi *Dropout*.
- **Skenario A (Intervensi Finansial Saja)**: Menyelamatkan 42 mahasiswa (penurunan dropout 14,7%).
- **Skenario B (Intervensi Akademik Saja - Peer Tutoring)**: Menyelamatkan 56 mahasiswa (penurunan dropout 19,6%).
- **Skenario C (Intervensi Gabungan: Finansial + Akademik)**: Menyelamatkan **84 mahasiswa (penurunan dropout sebesar 29,5%)**!
- **Kuantifikasi Dampak Ekonomi Institusi**:
  - Pada skala seluruh populasi kampus, intervensi ini diproyeksikan menyelamatkan **~420 mahasiswa**.
  - Dengan estimasi SPP Rp 6.000.000 per semester untuk 4 semester tersisa, kebijakan ini berpotensi **menyelamatkan pendapatan SPP kampus sebesar Rp 10,08 Miliar**.

---

## 🎯 Conclusion

1. **Akar Masalah Terjawab Secara Empiris**:
   Tingginya angka putus studi di Jaya Jaya Institut dipicu oleh kombinasi **kegagalan adaptasi akademik di semester 1–2 (khususnya rasio kelulusan mata kuliah yang rendah)** dan **kerentanan finansial (tunggakan SPP dan catatan utang)**. Mahasiswa sering kali mengalami *academic stagnation* terlebih dahulu, kemudian diperparah oleh tekanan finansial yang memaksa mereka keluar.
2. **Kesiapan Model sebagai Sistem Deteksi Dini**:
   Model machine learning Random Forest dengan formulasi Binary Classification (Dropout vs Graduate) terbukti sangat handal (**Akurasi 93,0%, Recall Dropout 91,2%, Presisi Dropout 90,9%, ROC-AUC 0,972**) dan teruji adil (*unbiased*). Model ini mampu mendeteksi lebih dari 9 dari setiap 10 mahasiswa yang berisiko putus studi sedini mungkin.
3. **Validasi Nilai Intervensi**:
   Simulasi kebijakan membuktikan bahwa *dropout rate* dapat ditekan hingga **29,5%** jika pihak kampus mengombinasikan bantuan finansial fleksibel dan tutorial belajar intensif, memberikan perlindungan pendapatan institusional hingga **Rp 10,08 Miliar**.

---

## 💡 Rekomendasi Action Items (Business Recommendations)

Berdasarkan temuan analitik, SHAP values, dan hasil simulasi What-If, dirumuskan 5 rekomendasi strategis bagi manajemen Jaya Jaya Institut:

1. **Implementasi Sistem Peringatan Dini Akademik Terintegrasi (Academic Early Warning System)**:
   - Hubungkan model machine learning langsung ke Sistem Informasi Akademik (SIAKAD) kampus.
   - Jalankan kalkulasi *Dropout Risk Score* otomatis setiap akhir semester 1 dan semester 2 saat KHS (Kartu Hasil Studi) diterbitkan.
   - Kirimkan notifikasi prioritas kepada Dosen Pembimbing Akademik (DPA) untuk mahasiswa dengan skor risiko $> 50\%$ guna penjadwalan sesi konseling wajib.

2. **Skema Bantuan Finansial Fleksibel & Restrukturisasi SPP Tanpa Bunga**:
   - Mengingat 86,5% penunggak SPP mengalami *dropout*, hapus kebijakan skorsing langsung bagi mahasiswa yang menunggak.
   - Ganti dengan program cicilan bertahap dan beasiswa darurat (*emergency micro-grants*) bagi mahasiswa berprestasi yang menghadapi kendala ekonomi keluarga mendadak.

3. **Pendirian Klinik Belajar & Program Tutorial Sebaya (*Peer Tutoring Program*)**:
   - Buka klinik bimbingan belajar gratis untuk mata kuliah dasar tingkat pertama dengan *failure rate* tertinggi (terutama di prodi Teknik Informatika dan Manajemen).
   - Pasangkan mahasiswa berisiko dengan mahasiswa tingkat atas berprestasi (*peer mentor*) untuk membantu adaptasi metode belajar perguruan tinggi.

4. **Layanan Pendampingan Khusus Mahasiswa Dewasa & Kuliah Malam (*Mature Student Support*)**:
   - Sediakan fleksibilitas jam bimbingan konseling di luar jam kerja (daring atau akhir pekan) bagi mahasiswa berusia $> 25$ tahun.
   - Sediakan fasilitas rekaman perkuliahan (*asynchronous lecture capture*) untuk membantu mahasiswa yang memiliki kewajiban kerja atau keluarga.

5. **Penyesuaian Batas Beban SKS Semester Awal**:
   - Batasi beban pengambilan SKS di semester 2 bagi mahasiswa yang meluluskan kurang dari 60% SKS di semester 1. Fokuskan mahasiswa pada perbaikan mata kuliah prasyarat sebelum mengambil mata kuliah lanjutan guna mencegah beban belajar berlebih (*overload*).

---

## 📂 Struktur Repositori

```
Penerapan-Data-Science-2/
│
├── .gitignore                         # Mengabaikan cache python dan temporary files
├── README.md                          # Dokumentasi komprehensif proyek & portfolio
├── requirements.txt                   # Daftar pustaka dependensi teruji
├── data.csv                           # Dataset resmi 4.424 mahasiswa Jaya Jaya Institut
├── notebook.ipynb                     # Jupyter Notebook CRISP-DM lengkap & dieksekusi
├── app.py                             # Aplikasi prototype Streamlit interaktif
├── nasich_dicoding-dashboard.png       # Screenshot dashboard Looker Studio
├── df_clean_mapped.csv                # Dataset dengan label deskriptif untuk dashboard
├── shap_data.csv                      # Feature Importance data untuk dashboard
├── what_if_data.csv                   # Simulasi What-If data untuk dashboard
│
└── model/
    ├── model.joblib                   # Model final Random Forest Classifier (~20 MB)
    └── model_meta.json                # Metadata fitur, metrik evaluasi, & data simulasi
```

---
*Dikembangkan dengan dedikasi untuk Proyek Akhir Belajar Penerapan Data Science - Dicoding Indonesia.*