# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Business Understanding

Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan telah mencetak banyak lulusan berprestasi. Namun, institusi ini menghadapi tantangan serius berupa **tingginya angka mahasiswa putus studi (*dropout*) yang mencapai 32,1%** dari total 4.424 mahasiswa.

Tingkat *dropout* yang tinggi ini berdampak langsung terhadap:
- **Kerugian finansial**: kehilangan pendapatan SPP dan biaya kuliah yang tidak tertagih.
- **Inefisiensi operasional**: alokasi dosen, ruang kelas, dan fasilitas kampus menjadi tidak optimal.
- **Penurunan reputasi**: menurunnya peringkat akreditasi dan daya saing institusi di mata calon mahasiswa baru.

### Permasalahan Bisnis

1. Pihak kampus belum memiliki sistem deteksi dini (*Early Warning System*) untuk mengidentifikasi mahasiswa berisiko putus studi sebelum terlambat.
2. Belum ada verifikasi empiris apakah faktor akademik (jumlah SKS lulus, nilai semester) atau faktor sosioekonomi (tunggakan SPP, status debitur) yang lebih dominan memicu *dropout*.

### Cakupan Proyek

Proyek ini bertujuan untuk:
1. Mengidentifikasi faktor-faktor utama yang mempengaruhi *dropout* mahasiswa melalui analisis data eksploratif dan interpretasi model machine learning (SHAP Values).
2. Membangun model prediktif klasifikasi untuk mendeteksi mahasiswa berisiko putus studi sedini mungkin.
3. Memberikan rekomendasi *action items* berbasis data kepada manajemen institusi.

---

## Proses Data Science (Alur Pengerjaan)

Proyek ini mengikuti metodologi **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) dengan tahapan:

1. **Data Wrangling**: Pemuatan dataset (4.424 baris × 37 kolom), pemeriksaan *missing values*, duplikasi, dan validasi tipe data.
2. **Exploratory Data Analysis (EDA)**: Analisis distribusi target (`Status`), pengaruh status finansial (tunggakan SPP, utang, beasiswa), kinerja akademik semester 1 & 2, usia pendaftaran, gender, dan matriks korelasi.
3. **Feature Engineering**: Pembuatan fitur analitis baru — rasio kelulusan semester (`Approval_rate_1st`, `Approval_rate_2nd`), rasio kelulusan kumulatif (`Total_approval_rate`), progresi nilai (`Grade_progression`), dan indeks risiko finansial (`Financial_risk_index`).
4. **Modeling**: Komparasi 4 algoritma klasifikasi (*Logistic Regression*, *Decision Tree*, *Random Forest*, *Gradient Boosting*) dengan 5-Fold Stratified Cross-Validation. Hyperparameter tuning menggunakan `GridSearchCV` pada model terbaik (Random Forest).
5. **Evaluation**: Evaluasi komprehensif menggunakan *Accuracy*, *Precision*, *Recall*, *F1-Score*, *ROC-AUC*, *Confusion Matrix*, analisis *Feature Importance*, dan *SHAP TreeExplainer*.

### Hasil Model

| Metrik | Nilai |
| :--- | :--- |
| Akurasi | 77,1% |
| Recall Dropout | 75,7% |
| Presisi Dropout | 83,0% |
| F1-Score (Macro) | 0,715 |
| ROC-AUC (OvR) | 0,892 |

---

## Links

### Dashboard Looker Studio

Tautan dashboard analitik bisnis untuk monitoring performa dan retensi mahasiswa:

[https://datastudio.google.com/reporting/08504851-cd41-49b5-a9c0-3376df7ad174](https://datastudio.google.com/reporting/08504851-cd41-49b5-a9c0-3376df7ad174)

Screenshot dashboard tersedia pada repositori sebagai `nasich_dicoding-dashboard.png`.

### Streamlit App (Prototype Machine Learning)

Tautan aplikasi web prototype prediksi risiko *dropout* mahasiswa:

[https://jaya-institute.streamlit.app/](https://jaya-institute.streamlit.app/)

---

## Rekomendasi Action Items

Berdasarkan temuan analitik, SHAP Values, dan hasil simulasi What-If, berikut **3 rekomendasi strategis** bagi manajemen Jaya Jaya Institut:

### 1. Program Bantuan Finansial Terarah untuk Prodi Berisiko Tinggi

**Temuan**: Mahasiswa yang menunggak SPP (`Tuition_fees_up_to_date = 0`) memiliki *dropout rate* sebesar **86,5%**, dan mahasiswa berstatus debitur (`Debtor = 1`) memiliki *dropout rate* **62,0%**. Program studi dengan *dropout* tertinggi adalah Teknik Informatika (54,1%) dan Manajemen Malam (50,7%).

**Aksi**: Implementasikan program cicilan SPP tanpa bunga dan beasiswa darurat (*emergency micro-grants*) yang diprioritaskan untuk mahasiswa di prodi Teknik Informatika dan Manajemen yang terdeteksi memiliki skor risiko finansial tinggi (`Financial_risk_index ≥ 3`). Alokasikan anggaran dari dana cadangan retensi mahasiswa.

### 2. Klinik Bimbingan Akademik Intensif di Semester 1–2

**Temuan**: Fitur `Total_approval_rate` (rasio kelulusan SKS kumulatif) dan `Approval_rate_2nd` (rasio kelulusan semester 2) adalah **2 prediktor terkuat** dalam model. Mahasiswa *Dropout* rata-rata hanya meluluskan 1,9 SKS per semester, sedangkan mahasiswa *Graduate* meluluskan 6,2 SKS.

**Aksi**: Dirikan program *Peer Tutoring* dan klinik belajar gratis khusus mata kuliah dasar tingkat pertama. Pasangkan mahasiswa berisiko dengan mahasiswa tingkat atas berprestasi sebagai mentor. Jalankan program ini wajib bagi mahasiswa dengan rasio kelulusan semester 1 di bawah 50%.

### 3. Sistem Peringatan Dini Terintegrasi dengan SIAKAD

**Temuan**: Model machine learning Random Forest berhasil mendeteksi **3 dari 4 mahasiswa berisiko putus studi** (Recall 75,7%) dengan tingkat presisi tinggi (83,0%). Simulasi What-If membuktikan bahwa kombinasi intervensi finansial dan akademik dapat menurunkan angka *dropout* sebesar **37,1%**.

**Aksi**: Integrasikan model prediktif ke dalam Sistem Informasi Akademik (SIAKAD) kampus. Jalankan kalkulasi *Dropout Risk Score* otomatis setiap akhir semester 1 dan 2 saat KHS diterbitkan. Kirimkan notifikasi prioritas kepada Dosen Pembimbing Akademik (DPA) untuk mahasiswa dengan skor risiko di atas 50%, dan jadwalkan sesi konseling wajib dalam 2 minggu pertama semester berikutnya.

---

## Cara Menjalankan Aplikasi Streamlit (Lokal)

### Setup Environment

```bash
# 1. Clone repositori
git clone https://github.com/dawooodd/Penerapan-Data-Science-2.git
cd Penerapan-Data-Science-2

# 2. Buat virtual environment (opsional, direkomendasikan)
python -m venv venv

# 3. Aktifkan virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependensi
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

Akses aplikasi pada browser di alamat: `http://localhost:8501`