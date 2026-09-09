# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan - Jaya Jaya Institut

## Business Understanding
**Jaya Jaya Institut** merupakan salah satu institusi perguruan tinggi swasta terkemuka yang telah beroperasi sejak tahun 2000. Selama lebih dari dua dekade, institusi ini telah berhasil meluluskan ribuan alumni yang sukses di berbagai bidang industri. Namun, terlepas dari rekam jejak akademiknya yang gemilang, Jaya Jaya Institut menghadapi tantangan struktural yang signifikan, yaitu **tingginya proporsi mahasiswa yang mengalami putus studi (*dropout*)**.

Tingginya angka *dropout* bukan sekadar masalah administratif internal, melainkan ancaman langsung terhadap keberlangsungan dan reputasi perguruan tinggi:
1. **Dampak Finansial**: Kehilangan pendapatan berkelanjutan dari uang kuliah (SPP/Tuition Fees) serta akumulasi piutang mahasiswa yang tidak tertagih.
2. **Efisiensi Sumber Daya**: Pemborosan kapasitas ruang kelas, alokasi rasio beban dosen, dan fasilitas laboratorium yang telah dianggarkan untuk kapasitas penuh.
3. **Reputasi dan Akreditasi**: Rasio retensi dan kelulusan mahasiswa merupakan metrik vital dalam akreditasi perguruan tinggi di tingkat nasional maupun internasional.
4. **Dampak Sosial bagi Mahasiswa**: Mahasiswa yang putus kuliah menghadapi hambatan prospek karier dan sering kali menanggung beban utang pendidikan tanpa memiliki ijazah formal.

### Permasalahan Bisnis
Berdasarkan investigasi dan diskusi manajerial, permasalahan pokok yang dihadapi oleh Jaya Jaya Institut meliputi:
1. **Tingkat Dropout yang Sangat Tinggi**: Analisis data historis menunjukkan sebanyak **32,1% (1.421 dari 4.424 mahasiswa)** mengalami *dropout*, suatu rasio yang berada jauh di atas ambang batas toleransi institusi pendidikan tinggi yang sehat.
2. **Ketiadaan Sistem Deteksi Dini (*Early Warning System*)**: Manajemen kampus belum memiliki instrumen berbasis data yang mampu mendeteksi indikasi kerentanan mahasiswa di semester-semester awal sebelum mereka mengambil keputusan resmi untuk mengundurkan diri.
3. **Ketidakjelasan Faktor Pemicu (*Root Causes*)**: Belum tersedianya pemahaman mendalam mengenai faktor utama apa saja (apakah kendala finansial, latar belakang demografis, atau penurunan performa akademik) yang paling berpengaruh terhadap keputusan putus studi.
4. **Intervensi yang Terlambat**: Upaya konseling atau bantuan yang dilakukan pihak kampus umumnya terjadi saat mahasiswa sudah menumpuk tunggakan biaya atau sudah berhenti menghadiri perkuliahan.

### Cakupan Proyek
Untuk mengatasi permasalahan bisnis tersebut, proyek Data Science ini mencakup tahapan *end-to-end* yang sistematis:
- **Analisis Data Eksploratif (EDA)**: Membedah profil mahasiswa dari aspek demografis, sosioekonomi, riwayat seleksi masuk, hingga kinerja akademik semester 1 dan 2.
- **Pembersihan & Rekayasa Fitur (*Feature Engineering*)**: Merancang metrik-metrik analitis baru (seperti *Total Approval Rate*, *Grade Progression*, dan *Financial Risk Index*) untuk meningkatkan daya beda model.
- **Pemodelan Machine Learning**: Membangun, mengomparasikan, dan melakukan *hyperparameter tuning* pada beberapa algoritma klasifikasi (*Logistic Regression*, *Decision Tree*, *Random Forest*, dan *Gradient Boosting*).
- **Evaluasi Model Komprehensif**: Menguji model menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score (Macro & Weighted)*, *Confusion Matrix*, dan *Multi-class ROC-AUC*.
- **Analisis Kepentingan Fitur (*Feature Importance*)**: Mengidentifikasi variabel-variabel kunci pemicu *dropout* guna menghasilkan dasar pengambilan keputusan manajerial.
- **Pengembangan Prototype Web (Streamlit)**: Membangun aplikasi web interaktif (`app.py`) yang siap digunakan oleh staf akademik dan konselor untuk melakukan prediksi risiko baik secara individu maupun massal (*batch CSV*).

### Persiapan

#### Sumber Data
Dataset yang digunakan merupakan data resmi mahasiswa **Jaya Jaya Institut** yang mencakup **4.424 data mahasiswa** dengan **36 fitur prediktor** dan **1 kolom target** (`Status`: *Dropout*, *Enrolled*, *Graduate*).
- Sumber Dataset: `data.csv` (repositori resmi Dicoding: `https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv`)

#### Setup Environment
Proyek ini dikembangkan menggunakan Python 3.10+ (atau Python 3.12). Ikuti langkah-langkah berikut untuk menyiapkan environment:

1. **Clone Repositori**:
   ```bash
   git clone https://github.com/dawooodd/Penerapan-Data-Science-2.git
   cd Penerapan-Data-Science-2
   ```

2. **Buat dan Aktifkan Virtual Environment**:
   - **Windows**:
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

## Business Dashboard
Sesuai dengan ketentuan dan batasan proyek, konfigurasi dan implementasi Business Intelligence Dashboard eksternal (menggunakan tools seperti Metabase, Tableau, Looker, atau Power BI) dikelola dan diatur secara mandiri oleh tim internal Jaya Jaya Institut.

- **Tautan Dashboard**: *[Placeholder link dashboard BI - akan ditambahkan oleh pengguna]*
- **Tinjauan Dashboard**: Visualisasi analitik komparatif performa akademik, retensi fakultas, dan ringkasan metrik kohort mahasiswa dapat diintegrasikan langsung pada platform BI terkait.

---

## Menjalankan Sistem Machine Learning

### Cara Menjalankan Prototype Secara Lokal
Aplikasi prototype deteksi dini risiko mahasiswa dibangun menggunakan framework **Streamlit**. Ikuti langkah-langkah berikut untuk menjalankannya:

1. Pastikan virtual environment telah aktif dan dependensi terpasang.
2. Jalankan perintah Streamlit dari direktori utama proyek:
   ```bash
   streamlit run app.py
   ```
3. Buka peramban (browser) dan akses alamat lokal:
   ```
   Local URL: http://localhost:8501
   ```

### Fitur Utama Aplikasi Prototype:
- **🎯 Prediksi Mahasiswa Tunggal**: Formulir interaktif lengkap dengan fitur **Preset Quick Profiles** (Profil Risiko Tinggi, Profil Sedang, Profil Mahasiswa Berprestasi) untuk evaluasi instan. Dilengkapi *gauge progress bar* probabilitas *dropout* dan kotak rekomendasi intervensi personal otomatis.
- **📁 Prediksi Massal (*Batch Assessment*)**: Unggah berkas CSV kohort mahasiswa untuk memetakan distribusi risiko secara serentak, dilengkapi ringkasan KPI kohort dan tombol unduh laporan hasil prediksi (.CSV). Tersedia pula tombol untuk mengunduh template CSV sampel.
- **📊 Performa Model & Fitur**: Transparansi metrik evaluasi model (Akurasi, Presisi, Recall, ROC-AUC) dan visualisasi interaktif *Top 10 Feature Importance*.
- **💡 Rekomendasi Bisnis Institusi**: Panduan taktis dan strategis bagi pimpinan perguruan tinggi dalam mereduksi tingkat putus studi.

### Tautan Deployment Cloud
Aplikasi prototype ini dapat diakses secara daring melalui Streamlit Community Cloud:
- **Tautan Aplikasi Web**: *[https://jaya-jaya-institut-retention.streamlit.app/ - Placeholder]*

---

## Conclusion

Berdasarkan rangkaian analisis data eksploratif (EDA), rekayasa fitur, dan evaluasi pemodelan *machine learning*, diperoleh kesimpulan strategis sebagai berikut:

1. **Akar Permasalahan Utama (*Core Problem Drivers*)**:
   - **Momentum Akademik Tahun Pertama adalah Penentu Utama**: Fitur rasio kelulusan mata kuliah (`Total_approval_rate`, `Approval_rate_2nd`, dan `Approval_rate_1st`) serta nilai semester 2 (`Curricular_units_2nd_sem_grade`) menyumbang lebih dari **30% pengaruh total** dalam memprediksi *dropout*. Mahasiswa yang gagal meluluskan lebih dari 40% mata kuliah pada semester 1 atau 2 memiliki kecenderungan putus studi yang eksponensial.
   - **Tunggakan Finansial sebagai Titik Kritis**: Mahasiswa yang menunggak pembayaran SPP (`Tuition_fees_up_to_date = 0`) memiliki angka *dropout* mencapai **86,5%**, berbanding terbalik dengan mahasiswa yang SPP-nya lancar (23,8%). Hal ini menunjukkan bahwa banyak mahasiswa terpaksa *dropout* bukan murni karena ketidakmampuan akademis, melainkan keterbatasan finansial yang tak teratasi.
   - **Efek Protektif Beasiswa**: Penerima beasiswa mencatatkan tingkat kelulusan **76,3%** dan tingkat *dropout* hanya **12,9%**. Beasiswa terbukti menjadi instrumen retensi yang sangat kuat.
   - **Faktor Usia dan Gender**: Mahasiswa yang mendaftar pada usia matang (> 25 tahun) memiliki tingkat *dropout* di atas 50%, dan mahasiswa laki-laki memiliki risiko *dropout* lebih tinggi (45,1%) dibanding perempuan (25,1%).

2. **Kinerja Model Machine Learning**:
   - Model **Random Forest Classifier** yang telah dituning dengan pembobotan kelas (*balanced subsample*) berhasil mencapai performa yang sangat solid:
     - **Akurasi Keseluruhan**: **77,1%**
     - **Recall Kelas Dropout**: **75,7%** (mampu mendeteksi lebih dari 3 dari setiap 4 mahasiswa yang berisiko putus studi).
     - **Presisi Kelas Dropout**: **83,0%** (meminimalkan alarm palsu sehingga alokasi bantuan tepat sasaran).
     - **ROC-AUC (One-vs-Rest)**: **0,892** (daya diskriminasi probabilitas yang sangat andal).

3. **Jawaban terhadap Kebutuhan Bisnis**:
   Dengan mengintegrasikan model ini ke dalam alur operasional akademik melalui prototype Streamlit, Jaya Jaya Institut kini memiliki instrumen objektif berbasis data untuk melakukan intervensi proaktif pada semester 1 dan 2, jauh sebelum mahasiswa memutuskan untuk putus kuliah.

---

### Rekomendasi Action Items
Guna menekan angka *dropout* secara signifikan dan mencapai target retensi mahasiswa jangka panjang, Jaya Jaya Institut disarankan untuk mengimplementasikan rekomendasi aksi berikut:

1. **Implementasi Sistem Peringatan Dini Akademik (*Automated Early Warning System*)**:
   - Menghubungkan model machine learning langsung ke Sistem Informasi Akademik (SIAKAD).
   - Menghitung *Dropout Risk Score* secara otomatis segera setelah nilai Ujian Tengah Semester (UTS) dan Ujian Akhir Semester (UAS) semester 1 keluar.
   - Sistem secara otomatis mengirimkan notifikasi kepada Dosen Pembimbing Akademik (DPA) untuk mahasiswa yang memiliki skor risiko > 50%.

2. **Skema Bantuan Finansial & Restrukturisasi SPP Fleksibel**:
   - Mengingat korelasi fatal antara tunggakan SPP dan *dropout* (86,5%), perguruan tinggi perlu menerapkan kebijakan **skema cicilan biaya kuliah tanpa denda** dan penundaan pembayaran bersyarat bagi mahasiswa yang mengalami kesulitan ekonomi darurat.
   - Mengalokasikan pos dana bantuan darurat (*emergency micro-grants*) atau beasiswa parsial bagi mahasiswa semester 1-2 yang berprestasi namun terancam kendala pembayaran SPP.

3. **Klinik Bimbingan Akademik & Program Tutorial Sebaya (*Peer Tutoring*)**:
   - Membuka klinik belajar gratis khusus mata kuliah dasar yang memiliki tingkat ketidaklulusan (*failure rate*) tertinggi di setiap program studi (terutama di Fakultas Teknik dan Manajemen).
   - Memasangkan mahasiswa berisiko dengan mentor mahasiswa tingkat atas berprestasi (*peer mentor*) untuk membantu adaptasi metode belajar perguruan tinggi.

4. **Layanan Pendampingan Khusus Mahasiswa Usia Dewasa (*Mature Students Support*)**:
   - Menyediakan fleksibilitas waktu bimbingan akademik dan sesi konseling di luar jam kerja (sore/malam atau daring) bagi mahasiswa berusia di atas 25 tahun yang umumnya memiliki beban kerja atau keluarga.
   - Mengadakan lokakarya manajemen waktu (*time-management workshop*) dan literasi adaptasi studi pada masa orientasi mahasiswa baru.

5. **Pemberian Insentif Beasiswa Berbasis Retensi**:
   - Memperluas kuota beasiswa berbasis kebutuhan finansial (*need-based scholarship*) yang dikaitkan dengan kehadiran dan kelulusan mata kuliah semester 1, bukan semata-mata nilai ujian masuk SMA.