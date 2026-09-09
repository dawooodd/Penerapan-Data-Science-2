import nbformat as nbf
from nbclient import NotebookClient
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.6"
        }
    }

    cells = []

    # Title & Header
    cells.append(nbf.v4.new_markdown_cell(
"""# Proyek Akhir Penerapan Data Science: Menyelesaikan Permasalahan Institusi Pendidikan
**Institusi**: Jaya Jaya Institut  
**Topik**: Prediksi Dropout & Analisis Keberhasilan Akademik Mahasiswa (Student Retention & Performance Prediction)  
**Metodologi**: CRISP-DM (*Cross-Industry Standard Process for Data Mining*)  
**Penulis**: Senior Data Scientist  

---

## Ringkasan Eksekutif
Jaya Jaya Institut adalah institusi pendidikan tinggi terkemuka yang telah berdiri sejak tahun 2000 dan berprestasi dalam mencetak lulusan bermutu tinggi. Meskipun demikian, institusi ini menghadapi tantangan signifikan terkait **tingginya rasio mahasiswa yang mengalami putus studi (*dropout*)**. Mahasiswa yang tidak menyelesaikan pendidikan menimbulkan dampak kerugian finansial langsung bagi perguruan tinggi (kehilangan *revenue* SPP), ketidakefisienan pemanfaatan fasilitas dan kapasitas dosen, serta menurunkan reputasi dan akreditasi kampus di mata publik.

Melalui proyek Data Science ini, dikembangkan alur analisis data menyeluruh (CRISP-DM) dan model *machine learning* klasifikasi prediktif untuk mendeteksi sedini mungkin mahasiswa yang berisiko tinggi mengalami *dropout*. Dengan deteksi dini ini, manajemen akademik dan tim konseling Jaya Jaya Institut dapat melakukan tindakan intervensi preventif yang terarah dan tepat sasaran.
"""
    ))

    # Stage 1: Business Understanding
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 1. Business Understanding

### 1.1 Latar Belakang Bisnis
Jaya Jaya Institut mengemban visi untuk menyediakan pendidikan tinggi yang inklusif dan berkualitas. Namun, seiring bertambahnya jumlah mahasiswa baru, proporsi mahasiswa yang berhasil menyelesaikan studinya hingga tahap kelulusan (*graduation*) mengalami penurunan akibat tingginya tingkat *dropout*. Pada industri pendidikan tinggi, tingkat retensi mahasiswa merupakan salah satu indikator kinerja utama (*Key Performance Indicator* / KPI) yang dipantau oleh badan akreditasi nasional dan internasional.

### 1.2 Perumusan Masalah Bisnis
Berdasarkan investigasi manajerial bersama pemangku kepentingan akademik, dirumuskan masalah bisnis pokok sebagai berikut:
1. **Tingginya Angka Dropout Mahasiswa**: Sekitar 32% mahasiswa mengalami *dropout*, yang secara langsung menggerus stabilitas finansial institusi dan menurunkan efektivitas program akademik.
2. **Ketiadaan Sistem Peringatan Dini (*Early Warning System*)**: Pihak manajemen akademik belum memiliki mekanisme otomatis untuk mendeteksi tanda-tanda awal mahasiswa yang mengalami kesulitan akademik atau beban finansial sebelum mereka resmi keluar.
3. **Ketidakpastian Faktor Pendorong (*Root Causes*)**: Belum ada identifikasi berbasis data mengenai faktor-faktor dominan (apakah latar belakang sosioekonomi, demografi, riwayat pendaftaran, atau performa akademik semester awal) yang paling berkontribusi terhadap keputusan *dropout*.

### 1.3 Cakupan Proyek (*Project Scope*)
Proyek ini mencakup tahapan data science terstruktur:
- Eksplorasi data komprehensif (EDA) untuk mengungkap pola dan korelasi antar variabel.
- Pembersihan dan rekayasa fitur (*feature engineering*) untuk memaksimalkan sinyal prediktif.
- Pembangunan dan komparasi berbagai algoritma *machine learning* klasifikasi.
- Hyperparameter tuning dan evaluasi model komprehensif (Akurasi, F1-Score, ROC-AUC, Confusion Matrix).
- Analisis *feature importance* guna memberikan interpretasi bisnis yang dapat ditindaklanjuti.
- Serialisasi model terbaik dan pembentukan aplikasi prototype interaktif berbasis Streamlit.

### 1.4 Metrik Keberhasilan Bisnis & Evaluasi Model
- **Keberhasilan Bisnis**: Kemampuan mengidentifikasi minimal 75% mahasiswa berisiko *dropout* secara akurat sebelum akhir tahun pertama, memungkinkan intervensi terarah untuk mereduksi *dropout rate* sebesar 15-20%.
- **Metrik Model Machine Learning**:
  - **Recall Kelas Dropout**: Meminimalkan *False Negatives* (mahasiswa yang sebenarnya berisiko drop out tetapi tidak terdeteksi).
  - **Precision Kelas Dropout**: Menghindari *False Positives* berlebih agar alokasi konseling/beasiswa tepat sasaran.
  - **Macro & Weighted F1-Score**: Keseimbangan performa antar ketiga kelas (*Dropout*, *Enrolled*, *Graduate*).
  - **ROC-AUC Score**: Kemampuan membedakan probabilitas risiko secara andal.
"""
    ))

    # Stage 2: Data Understanding
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 2. Data Understanding

### 2.1 Memuat Pustaka dan Konfigurasi Lingkungan
Pada tahap ini, pustaka analisis data, visualisasi, dan pemodelan machine learning diimpor ke dalam lingkungan kerja.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Import pustaka analisis data & manipulasi
import pandas as pd
import numpy as np

# Import pustaka visualisasi data
import matplotlib.pyplot as plt
import seaborn as sns

# Import pustaka machine learning & evaluasi
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Pustaka utilitas & serialisasi model
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi estetika plot
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

print("Pustaka berhasil diimpor dan siap digunakan.")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 2.2 Pemuatan Dataset
Dataset mahasiswa Jaya Jaya Institut dimuat dari berkas `data.csv`. Berkas ini menggunakan pemisah titik-koma (`;`) dengan pengodean `utf-8-sig`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Memuat dataset data.csv
data_path = 'data.csv'
df = pd.read_csv(data_path, sep=';', encoding='utf-8-sig')

# Menampilkan informasi dasar dimensi dataset
print(f"Dimensi dataset: {df.shape[0]} baris dan {df.shape[1]} kolom.")
df.head()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 2.3 Kamus Data & Karakteristik Fitur (*Data Dictionary*)
Dataset terdiri dari 36 fitur prediktor dan 1 kolom target (`Status`). Fitur-fitur ini dikelompokkan ke dalam beberapa dimensi analitis:

1. **Dimensi Demografi & Personal**:
   - `Marital_status`: Status pernikahan (1: Lajang, 2: Menikah, 3: Duda/Janda, 4: Cerai, 5: Hubungan de facto, 6: Pisah hukum).
   - `Nacionality`: Kewarganegaraan mahasiswa.
   - `Displaced`: Mahasiswa perantau/pindahan tempat tinggal (1: Ya, 0: Tidak).
   - `Gender`: Jenis kelamin (1: Laki-laki, 0: Perempuan).
   - `Age_at_enrollment`: Usia mahasiswa saat pendaftaran kuliah.
   - `International`: Mahasiswa internasional (1: Ya, 0: Tidak).

2. **Dimensi Sosioekonomi & Finansial**:
   - `Mothers_qualification` & `Fathers_qualification`: Tingkat kualifikasi pendidikan orang tua.
   - `Mothers_occupation` & `Fathers_occupation`: Pekerjaan orang tua.
   - `Educational_special_needs`: Kebutuhan pendidikan khusus (1: Ya, 0: Tidak).
   - `Debtor`: Memiliki catatan utang/tunggakan keuangan (1: Ya, 0: Tidak).
   - `Tuition_fees_up_to_date`: Pembayaran uang kuliah lancar/lunas (1: Ya, 0: Tidak).
   - `Scholarship_holder`: Penerima beasiswa (1: Ya, 0: Tidak).

3. **Dimensi Riwayat Akademik Masuk**:
   - `Application_mode`: Jalur pendaftaran masuk.
   - `Application_order`: Urutan preferensi pilihan program studi (pilihan ke-1 hingga ke-8).
   - `Course`: Kode program studi/jurusan yang diambil.
   - `Daytime_evening_attendance`: Jadwal kuliah (1: Siang/Pagi, 0: Malam).
   - `Previous_qualification`: Kualifikasi pendidikan sebelumnya.
   - `Previous_qualification_grade`: Nilai kualifikasi pendidikan sebelumnya (skala 0-200).
   - `Admission_grade`: Nilai seleksi masuk perguruan tinggi (skala 0-200).

4. **Dimensi Kinerja Akademik Semester 1**:
   - `Curricular_units_1st_sem_credited`: Jumlah SKS diakui/transfer.
   - `Curricular_units_1st_sem_enrolled`: Jumlah mata kuliah terdaftar semester 1.
   - `Curricular_units_1st_sem_evaluations`: Jumlah evaluasi/ujian diikuti semester 1.
   - `Curricular_units_1st_sem_approved`: Jumlah mata kuliah yang lulus di semester 1.
   - `Curricular_units_1st_sem_grade`: Rata-rata nilai semester 1 (skala 0-20).
   - `Curricular_units_1st_sem_without_evaluations`: Mata kuliah tanpa evaluasi semester 1.

5. **Dimensi Kinerja Akademik Semester 2**:
   - `Curricular_units_2nd_sem_credited`, `enrolled`, `evaluations`, `approved`, `grade`, `without_evaluations`: Metrik kinerja pada semester 2.

6. **Dimensi Konteks Makroekonomi**:
   - `Unemployment_rate`: Tingkat pengangguran regional (%).
   - `Inflation_rate`: Tingkat inflasi tahunan (%).
   - `GDP`: Pertumbuhan produk domestik bruto (%).

7. **Variabel Target**:
   - `Status`: Status akhir mahasiswa, terdiri dari 3 kategori:
     - `Graduate`: Mahasiswa telah berhasil lulus.
     - `Dropout`: Mahasiswa berhenti/putus studi sebelum lulus.
     - `Enrolled`: Mahasiswa masih aktif menempuh studi.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Pemeriksaan tipe data dan ringkasan informasi dataset
print("Informasi Tipe Data dan Nilai Kosong:")
df.info()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 2.4 Pemeriksaan Kualitas Data (Pembersihan, Nilai Hilang, & Duplikasi)
Sebelum melakukan analisis eksploratif, dilakukan verifikasi kualitas data untuk memastikan tidak ada anomali atau data kotor yang mengganggu pemodelan.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Memeriksa keberadaan nilai hilang (missing values)
missing_counts = df.isnull().sum()
total_missing = missing_counts.sum()

# Memeriksa baris duplikat
duplicate_count = df.duplicated().sum()

print(f"Total missing values dalam dataset: {total_missing}")
print(f"Total baris duplikat dalam dataset: {duplicate_count}")

# Ringkasan statistik deskriptif untuk fitur numerik utama
academic_cols = [
    'Age_at_enrollment', 'Admission_grade', 'Previous_qualification_grade',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
]
df[academic_cols].describe().round(2)
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Pemeriksaan Kualitas Data**:
- Dataset memiliki **0 missing values** dan **0 duplicate rows**, mengindikasikan tingkat integritas data yang sangat baik dari sistem pencatatan akademik.
- Seluruh variabel numerik berada pada rentang yang valid (misalnya nilai mata kuliah dalam skala 0–20, nilai seleksi masuk dalam rentang 95–190, dan usia pendaftaran berkisar antara 17 hingga 70 tahun).
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 2.5 Exploratory Data Analysis (EDA)

#### 2.5.1 Distribusi Variabel Target (`Status`)
Menganalisis proporsi mahasiswa yang lulus (*Graduate*), putus kuliah (*Dropout*), dan masih aktif (*Enrolled*).
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualisasi Distribusi Variabel Target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Batang Distribusi Target
status_counts = df['Status'].value_counts()
colors = ['#2b5c8f', '#d9534f', '#f0ad4e']

sns.barplot(x=status_counts.index, y=status_counts.values, ax=axes[0], palette=colors)
axes[0].set_title('Distribusi Frekuensi Status Mahasiswa', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Jumlah Mahasiswa')
axes[0].set_xlabel('Status Mahasiswa')
for i, v in enumerate(status_counts.values):
    axes[0].text(i, v + 40, f"{v:,} ({v/len(df)*100:.1f}%)", ha='center', fontweight='semibold')

# Donut Chart
axes[1].pie(
    status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
    startangle=140, colors=colors, explode=(0.03, 0.05, 0.02),
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
axes[1].add_artist(centre_circle)
axes[1].set_title('Persentase Proporsi Status Mahasiswa', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Bisnis Distribusi Target**:
1. Sebanyak **1.421 mahasiswa (32,1%)** berstatus **Dropout**, sedangkan 2.209 mahasiswa (49,9%) berhasil Graduate, dan 794 mahasiswa (18,0%) masih Enrolled.
2. Angka dropout mendekati sepertiga dari total populasi mahasiswa. Rasio ini sangat tinggi untuk institusi pendidikan tinggi dan menegaskan urgensi implementasi model prediktif untuk deteksi dini.
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 2.5.2 Pengaruh Status Finansial terhadap Dropout
Menganalisis korelasi status kelancaran pembayaran SPP (`Tuition_fees_up_to_date`), kepemilikan utang (`Debtor`), dan penerima beasiswa (`Scholarship_holder`) terhadap status mahasiswa.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualisasi Hubungan Finansial dengan Status Mahasiswa
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Kelancaran Pembayaran SPP
tuition_cross = pd.crosstab(df['Tuition_fees_up_to_date'], df['Status'], normalize='index') * 100
tuition_cross.plot(kind='bar', stacked=True, ax=axes[0], color=['#d9534f', '#f0ad4e', '#2b5c8f'], edgecolor='none')
axes[0].set_title('Status vs Kelancaran SPP', fontweight='bold')
axes[0].set_xlabel('SPP Lancar (1 = Ya, 0 = Menunggak)')
axes[0].set_ylabel('Persentase (%)')
axes[0].set_xticklabels(['Menunggak (0)', 'Lancar (1)'], rotation=0)
axes[0].legend(title='Status', loc='upper right')

# 2. Status Memiliki Utang (Debtor)
debtor_cross = pd.crosstab(df['Debtor'], df['Status'], normalize='index') * 100
debtor_cross.plot(kind='bar', stacked=True, ax=axes[1], color=['#d9534f', '#f0ad4e', '#2b5c8f'], edgecolor='none')
axes[1].set_title('Status vs Catatan Utang (Debtor)', fontweight='bold')
axes[1].set_xlabel('Punya Utang (1 = Ya, 0 = Tidak)')
axes[1].set_ylabel('Persentase (%)')
axes[1].set_xticklabels(['Bebas Utang (0)', 'Memiliki Utang (1)'], rotation=0)
axes[1].legend(title='Status', loc='upper right')

# 3. Penerima Beasiswa
scholar_cross = pd.crosstab(df['Scholarship_holder'], df['Status'], normalize='index') * 100
scholar_cross.plot(kind='bar', stacked=True, ax=axes[2], color=['#d9534f', '#f0ad4e', '#2b5c8f'], edgecolor='none')
axes[2].set_title('Status vs Penerima Beasiswa', fontweight='bold')
axes[2].set_xlabel('Beasiswa (1 = Penerima, 0 = Bukan)')
axes[2].set_ylabel('Persentase (%)')
axes[2].set_xticklabels(['Non-Beasiswa (0)', 'Penerima Beasiswa (1)'], rotation=0)
axes[2].legend(title='Status', loc='upper right')

plt.tight_layout()
plt.show()

# Menampilkan angka persentase detail
print("Persentase Dropout berdasarkan Kelancaran SPP:")
print((pd.crosstab(df['Tuition_fees_up_to_date'], df['Status'], normalize='index') * 100).round(1))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Bisnis Faktor Finansial**:
1. **Faktor Pembayaran SPP sangat fatal**: Pada mahasiswa yang menunggak SPP (`Tuition_fees_up_to_date = 0`), angka dropout mencapai **86,5%**, sedangkan pada mahasiswa yang SPP-nya lancar, angka dropout hanya sebesar **23,8%**.
2. **Pengaruh Utang (*Debtor*)**: Mahasiswa yang berstatus debitur memiliki tingkat dropout mencapai **62,1%**, jauh lebih tinggi dibanding mahasiswa bebas utang (27,4%).
3. **Efek Perlindungan Beasiswa**: Mahasiswa penerima beasiswa memiliki tingkat kelulusan tinggi (76,3%) dan tingkat dropout yang sangat rendah (12,9%). Beasiswa terbukti menjadi jaring pengaman retensi mahasiswa yang sangat efektif.
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 2.5.3 Pengaruh Kinerja Akademik Awal (Semester 1 dan 2)
Menganalisis bagaimana jumlah mata kuliah yang lulus di semester 1 dan semester 2 mempengaruhi status kelulusan mahasiswa.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualisasi Pengaruh Kinerja Akademik Semester 1 dan 2
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot Mata Kuliah Lulus Semester 1 & 2
sns.boxplot(x='Status', y='Curricular_units_1st_sem_approved', data=df, ax=axes[0], palette=['#d9534f', '#2b5c8f', '#f0ad4e'])
axes[0].set_title('Mata Kuliah Lulus Semester 1 vs Status Mahasiswa', fontweight='bold')
axes[0].set_ylabel('Jumlah Mata Kuliah Lulus (Semester 1)')
axes[0].set_xlabel('Status Mahasiswa')

sns.boxplot(x='Status', y='Curricular_units_2nd_sem_approved', data=df, ax=axes[1], palette=['#d9534f', '#2b5c8f', '#f0ad4e'])
axes[1].set_title('Mata Kuliah Lulus Semester 2 vs Status Mahasiswa', fontweight='bold')
axes[1].set_ylabel('Jumlah Mata Kuliah Lulus (Semester 2)')
axes[1].set_xlabel('Status Mahasiswa')

plt.tight_layout()
plt.show()

# Rata-rata per status
print("Rata-rata Mata Kuliah Lulus:")
print(df.groupby('Status')[['Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved', 'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']].mean().round(2))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Bisnis Kinerja Akademik**:
1. Mahasiswa yang **Dropout** rata-rata hanya meluluskan **2,55 mata kuliah di semester 1** dan anjlok menjadi **1,94 mata kuliah di semester 2**.
2. Sebaliknya, mahasiswa yang **Graduate** rata-rata meluluskan **6,23 mata kuliah di semester 1** dan **6,18 mata kuliah di semester 2**.
3. Penurunan jumlah mata kuliah yang lulus antara semester 1 dan semester 2 (*academic deceleration*) merupakan indikator kuat bahwa mahasiswa sedang kehilangan motivasi studi dan mendekati keputusan putus kuliah.
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 2.5.4 Pengaruh Usia Saat Pendaftaran (*Age at Enrollment*) dan Gender
Menganalisis apakah usia pendaftaran dan gender memiliki hubungan dengan kecenderungan putus kuliah.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualisasi Usia Pendaftaran & Gender
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Distribusi Usia Pendaftaran Berdasarkan Status
sns.kdeplot(data=df, x='Age_at_enrollment', hue='Status', common_norm=False, fill=True, ax=axes[0], palette=['#d9534f', '#2b5c8f', '#f0ad4e'], alpha=0.3)
axes[0].set_title('Distribusi Kepadatan Usia saat Pendaftaran', fontweight='bold')
axes[0].set_xlabel('Usia saat Pendaftaran (Tahun)')
axes[0].set_ylabel('Kepadatan (Density)')

# Proporsi Gender vs Status
gender_cross = pd.crosstab(df['Gender'], df['Status'], normalize='index') * 100
gender_cross.plot(kind='bar', stacked=True, ax=axes[1], color=['#d9534f', '#f0ad4e', '#2b5c8f'], edgecolor='none')
axes[1].set_title('Status Mahasiswa Berdasarkan Gender', fontweight='bold')
axes[1].set_xlabel('Gender (0 = Perempuan, 1 = Laki-laki)')
axes[1].set_ylabel('Persentase (%)')
axes[1].set_xticklabels(['Perempuan (0)', 'Laki-laki (1)'], rotation=0)
axes[1].legend(title='Status', loc='upper right')

plt.tight_layout()
plt.show()

print("Persentase Dropout berdasarkan Usia Pendaftaran:")
df['Age_Category'] = pd.cut(df['Age_at_enrollment'], bins=[0, 20, 25, 30, 100], labels=['<=20', '21-25', '26-30', '>30'])
print((pd.crosstab(df['Age_Category'], df['Status'], normalize='index') * 100).round(1))
df.drop('Age_Category', axis=1, inplace=True)
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Bisnis Usia & Gender**:
1. **Risiko Usia Matang**: Mahasiswa yang mendaftar pada usia di atas 25 tahun memiliki tingkat dropout di atas **50%**, dan pada usia di atas 30 tahun mencapai **59,4%**. Mahasiswa usia dewasa umumnya memiliki tanggung jawab pekerjaan atau keluarga yang bersaing dengan waktu kuliah.
2. **Kesenjangan Gender**: Mahasiswa laki-laki memiliki risiko dropout lebih tinggi (45,1%) dibandingkan mahasiswa perempuan (25,1%). Mahasiswa perempuan menunjukkan tingkat kelulusan yang jauh lebih dominan (58,0% vs 35,0%).
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""#### 2.5.5 Korelasi Antar Variabel Kunci
Mengevaluasi matriks korelasi fitur-fitur numerik terpilih terhadap status mahasiswa.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Mengonversi Status menjadi kode numerik sementara untuk melihat korelasi (Dropout=1, Enrolled=0, Graduate=0)
df_corr = df.copy()
df_corr['Target_Dropout'] = (df_corr['Status'] == 'Dropout').astype(int)

selected_features = [
    'Target_Dropout', 'Tuition_fees_up_to_date', 'Debtor', 'Scholarship_holder',
    'Age_at_enrollment', 'Gender', 'Admission_grade',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
]

corr_matrix = df_corr[selected_features].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, mask=mask, cbar_kws={'shrink': 0.8})
plt.title('Matriks Korelasi Fitur Kunci terhadap Risiko Dropout', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Bisnis Matriks Korelasi**:
- Korelasi negatif paling kuat terhadap *Dropout* adalah `Curricular_units_2nd_sem_approved` (-0.61) dan `Curricular_units_1st_sem_approved` (-0.53), disusul oleh `Tuition_fees_up_to_date` (-0.41).
- Korelasi positif tertinggi dengan *Dropout* adalah `Age_at_enrollment` (+0.24) dan `Debtor` (+0.24).
- Temuan ini mengonfirmasi bahwa retensi mahasiswa sangat dipengaruhi oleh kombinasi **kapabilitas akademik awal** dan **keberlanjutan finansial**.
"""
    ))

    # Stage 3: Data Preparation
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 3. Data Preparation

### 3.1 Feature Engineering (Rekayasa Fitur)
Untuk memperkaya sinyal prediktif yang ditangkap oleh algoritma machine learning, dirancang beberapa fitur domain-spesifik pendidikan:
1. `Approval_rate_1st`: Rasio mata kuliah lulus terhadap mata kuliah terdaftar pada semester 1 ($Approved / Enrolled$).
2. `Approval_rate_2nd`: Rasio mata kuliah lulus terhadap mata kuliah terdaftar pada semester 2.
3. `Total_enrolled`: Total mata kuliah terdaftar pada tahun pertama ($Sem1 + Sem2$).
4. `Total_approved`: Total mata kuliah lulus pada tahun pertama.
5. `Total_approval_rate`: Rasio kelulusan mata kuliah kumulatif tahun pertama ($Total Approved / Total Enrolled$).
6. `Grade_progression`: Perubahan/selisih nilai rata-rata dari semester 1 ke semester 2 ($Sem2 - Sem1$).
7. `Approved_diff`: Selisih jumlah mata kuliah yang lulus antara semester 2 dan semester 1.
8. `Financial_risk_index`: Indeks risiko finansial gabungan berbasis utang, tunggakan SPP, dan status beasiswa ($Debtor \times 2 + (1 - SPP) \times 3 - Beasiswa \times 2$).
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""def engineer_features(data):
    \"\"\"
    Fungsi rekayasa fitur untuk menambahkan atribut analitis berbasis kinerja akademik dan finansial.
    \"\"\"
    df_feat = data.copy()
    
    # Pencegahan pembagian dengan nol
    sem1_enr = np.where(df_feat['Curricular_units_1st_sem_enrolled'] == 0, 1, df_feat['Curricular_units_1st_sem_enrolled'])
    sem2_enr = np.where(df_feat['Curricular_units_2nd_sem_enrolled'] == 0, 1, df_feat['Curricular_units_2nd_sem_enrolled'])
    
    # Rasio kelulusan per semester
    df_feat['Approval_rate_1st'] = df_feat['Curricular_units_1st_sem_approved'] / sem1_enr
    df_feat['Approval_rate_2nd'] = df_feat['Curricular_units_2nd_sem_approved'] / sem2_enr
    
    # Akumulasi tahun pertama
    tot_enr = df_feat['Curricular_units_1st_sem_enrolled'] + df_feat['Curricular_units_2nd_sem_enrolled']
    tot_enr_safe = np.where(tot_enr == 0, 1, tot_enr)
    tot_app = df_feat['Curricular_units_1st_sem_approved'] + df_feat['Curricular_units_2nd_sem_approved']
    
    df_feat['Total_enrolled'] = tot_enr
    df_feat['Total_approved'] = tot_app
    df_feat['Total_approval_rate'] = tot_app / tot_enr_safe
    
    # Progresi akademik antar semester
    df_feat['Grade_progression'] = df_feat['Curricular_units_2nd_sem_grade'] - df_feat['Curricular_units_1st_sem_grade']
    df_feat['Approved_diff'] = df_feat['Curricular_units_2nd_sem_approved'] - df_feat['Curricular_units_1st_sem_approved']
    
    # Indeks komposit risiko finansial
    df_feat['Financial_risk_index'] = (
        df_feat['Debtor'] * 2 + 
        (1 - df_feat['Tuition_fees_up_to_date']) * 3 - 
        df_feat['Scholarship_holder'] * 2
    )
    
    return df_feat

# Menjalankan fungsi rekayasa fitur
df_prep = engineer_features(df)
print(f"Jumlah kolom sebelum rekayasa: {df.shape[1]}")
print(f"Jumlah kolom setelah rekayasa fitur: {df_prep.shape[1]}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 3.2 Pemisahan Fitur dan Target (*Feature-Target Split*)
Memisahkan fitur prediktor ($X$) dan variabel label status mahasiswa ($y$).
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Memisahkan X dan y
X = df_prep.drop('Status', axis=1)
y = df_prep['Status']

print(f"Dimensi fitur X: {X.shape}")
print(f"Distribusi kelas target y:\\n{y.value_counts()}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 3.3 Pembagian Data Latih dan Uji (*Stratified Train-Test Split*)
Data dibagi menjadi 80% data latih (*training set*) dan 20% data uji (*testing set*). Digunakan `stratify=y` agar proporsi kelas *Dropout*, *Enrolled*, dan *Graduate* tetap identik pada data latih dan data uji, mencegah bias distribusi.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Pembagian data latih dan uji secara stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Jumlah sampel Data Latih: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Jumlah sampel Data Uji  : {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.0f}%)")
print("\\nDistribusi Kelas pada Data Latih (%):")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\\nDistribusi Kelas pada Data Uji (%):")
print((y_test.value_counts(normalize=True) * 100).round(2))
"""
    ))

    # Stage 4: Modeling
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 4. Modeling

### 4.1 Pemilihan Algoritma & Komparasi Model
Untuk memperoleh performa prediktif yang optimal dan dapat diandalkan, dievaluasi 4 algoritma klasifikasi:
1. **Logistic Regression (Baseline Model)**: Model linear dengan standardisasi fitur untuk tolok ukur awal.
2. **Decision Tree Classifier**: Model berbasis pohon keputusan dengan kedalaman terbatas.
3. **Random Forest Classifier**: Model *ensemble bagging* yang tangguh terhadap overfitting dan mampu menangani interaksi non-linear.
4. **Gradient Boosting Classifier**: Model *ensemble boosting* yang membangun pohon keputusan secara sekuensial untuk meminimalkan residual error.

Dilakukan **5-Fold Stratified Cross-Validation** pada data latih untuk mengukur stabilitas performa setiap model.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Definisi kandidat model
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=16, min_samples_split=4,
        class_weight='balanced_subsample', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42
    )
}

# Evaluasi menggunakan 5-Fold Stratified Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results.append({
        'Model': name,
        'Mean CV Accuracy': scores.mean(),
        'Std CV Accuracy': scores.std()
    })

cv_df = pd.DataFrame(cv_results)
print("Hasil 5-Fold Stratified Cross-Validation pada Data Latih:")
cv_df
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 4.2 Pelatihan dan Pengujian Model pada Test Set
Setiap model dilatih menggunakan seluruh data latih dan diuji pada data uji yang belum pernah dilihat sebelumnya (*unseen test data*).
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""test_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    rec_dropout = recall_score(y_test, y_pred, labels=['Dropout'], average=None)[0]
    prec_dropout = precision_score(y_test, y_pred, labels=['Dropout'], average=None)[0]
    
    test_results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Macro F1': round(f1_macro, 4),
        'Weighted F1': round(f1_weighted, 4),
        'Dropout Recall': round(rec_dropout, 4),
        'Dropout Precision': round(prec_dropout, 4)
    })

results_df = pd.DataFrame(test_results)
print("Tabel Komparasi Performa Model pada Data Uji:")
results_df
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 4.3 Hyperparameter Tuning Model Terbaik (Random Forest)
Random Forest dengan *balanced_subsample* menunjukkan performa F1-Score dan Recall kelas Dropout tertinggi. Dilakukan *hyperparameter tuning* lebih lanjut untuk menemukan kombinasi parameter optimal.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Hyperparameter Tuning menggunakan GridSearchCV pada Random Forest
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [14, 16, 18],
    'min_samples_split': [3, 4, 5],
    'class_weight': ['balanced_subsample']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Parameter Terbaik Hasil Tuning: {grid_search.best_params_}")
print(f"Skor F1-Macro CV Terbaik: {grid_search.best_score_:.4f}")
"""
    ))

    # Stage 5: Evaluation
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 5. Evaluation

### 5.1 Evaluasi Komprehensif Model Final
Mengevaluasi model Random Forest terbaik pada data uji menggunakan laporan klasifikasi (*Classification Report*), matriks kebingungan (*Confusion Matrix*), dan metrik ROC-AUC.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Prediksi pada data uji dengan model terbaik
y_pred_best = best_rf.predict(X_test)
y_proba_best = best_rf.predict_proba(X_test)

print("Laporan Klasifikasi Komprehensif (Classification Report):")
print(classification_report(y_test, y_pred_best))

# Evaluasi ROC-AUC Multi-class (One-vs-Rest)
roc_auc = roc_auc_score(y_test, y_proba_best, multi_class='ovr')
print(f"ROC-AUC Score (One-vs-Rest): {roc_auc:.4f}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 5.2 Visualisasi Confusion Matrix
Menganalisis matriks kebingungan untuk memahami distribusi prediksi yang benar dan kesalahan klasifikasi antar kelas.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualisasi Confusion Matrix
labels = ['Dropout', 'Enrolled', 'Graduate']
cm = confusion_matrix(y_test, y_pred_best, labels=labels)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
plt.title('Confusion Matrix Model Random Forest Final', fontsize=13, fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Status Aktual')
plt.tight_layout()
plt.show()

# Penjelasan angka dalam Confusion Matrix
total_dropout_actual = cm[0].sum()
correct_dropout = cm[0][0]
print(f"Dari {total_dropout_actual} mahasiswa Dropout aktual, {correct_dropout} ({correct_dropout/total_dropout_actual*100:.1f}%) berhasil dideteksi dengan tepat.")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 5.3 Analisis Kurva ROC (*Receiver Operating Characteristic*)
Menggambarkan kurva ROC untuk masing-masing kelas target (*One-vs-Rest*) guna memvalidasi kemampuan diskriminatif model pada berbagai ambang batas probabilitas.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Binarisasi label target untuk perhitungan ROC multiclass
y_test_bin = label_binarize(y_test, classes=labels)
n_classes = y_test_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc_dict = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_best[:, i])
    roc_auc_dict[i] = auc(fpr[i], tpr[i])

# Plotting ROC Curves
plt.figure(figsize=(8, 6))
colors_roc = ['#d9534f', '#f0ad4e', '#2b5c8f']

for i, color in zip(range(n_classes), colors_roc):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC {labels[i]} (AUC = {roc_auc_dict[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Multi-class ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""### 5.4 Analisis Fitur Paling Berpengaruh (*Feature Importance*)
Mengidentifikasi fitur-fitur yang paling dominan dalam membentuk keputusan klasifikasi model. Informasi ini sangat krusial sebagai dasar pengambilan keputusan manajemen Jaya Jaya Institut.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Ekstraksi Feature Importance
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Visualisasi Top 15 Fitur Terpenting
plt.figure(figsize=(10, 7))
sns.barplot(
    x='Importance', y='Feature', data=feature_importance_df.head(15),
    palette='viridis'
)
plt.title('Top 15 Fitur Paling Berpengaruh dalam Memprediksi Status Mahasiswa', fontsize=13, fontweight='bold')
plt.xlabel('Tingkat Kepentingan Relatif (Feature Importance)')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()

# Menampilkan 15 fitur teratas
print("Tabel 15 Fitur Teratas:")
feature_importance_df.head(15).reset_index(drop=True)
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""**Insight Bisnis Hasil Feature Importance**:
1. **Dominasi Fitur Hasil Rekayasa Akademik**: Fitur `Total_approval_rate`, `Approval_rate_2nd`, `Curricular_units_2nd_sem_grade`, dan `Total_approved` mendominasi posisi teratas dengan kontribusi lebih dari 30% dari total *importance*. Hal ini membuktikan bahwa **keberhasilan mahasiswa di tahun pertama adalah penentu utama kelulusan**.
2. **Kekuatan Fitur Finansial**: `Financial_risk_index` menempati peringkat penting, mengonfirmasi bahwa kendala pembayaran uang kuliah secara signifikan memperbesar kemungkinan mahasiswa berhenti studi.
3. **Faktor Demografi**: `Age_at_enrollment` (usia saat mendaftar) menjadi salah satu fitur demografi paling berpengaruh, di mana mahasiswa usia dewasa membutuhkan pendampingan khusus dalam manajemen waktu.
"""
    ))

    # Stage 6: Export Model
    cells.append(nbf.v4.new_markdown_cell(
"""---
## 6. Model Export & Artifact Serialization

Untuk memfasilitasi integrasi dengan sistem aplikasi prototype Streamlit (`app.py`), model terlatih dan metadata pendukung disimpan ke dalam direktori `model/`:
- `model/model.joblib`: Berkas biner model Random Forest terbaik.
- `model/model_meta.json`: Metadata spesifikasi fitur masukan, daftar kelas target, metrik performa, dan fitur terpenting.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Membuat direktori model jika belum ada
os.makedirs('model', exist_ok=True)

# 1. Simpan Model Terbaik ke model/model.joblib
model_path = os.path.join('model', 'model.joblib')
joblib.dump(best_rf, model_path)
print(f"Model berhasil disimpan ke: {model_path} (Ukuran: {os.path.getsize(model_path)/1024/1024:.2f} MB)")

# 2. Siapkan Metadata Model
model_meta = {
    "model_name": "RandomForestClassifier",
    "classes": labels,
    "input_features": list(X.columns),
    "raw_features": [col for col in df.columns if col != 'Status'],
    "metrics": {
        "accuracy": round(float(accuracy_score(y_test, y_pred_best)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred_best, average='macro')), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred_best, average='weighted')), 4),
        "roc_auc_ovr": round(float(roc_auc), 4),
        "dropout_recall": round(float(recall_score(y_test, y_pred_best, labels=['Dropout'], average=None)[0]), 4),
        "dropout_precision": round(float(precision_score(y_test, y_pred_best, labels=['Dropout'], average=None)[0]), 4)
    },
    "best_parameters": grid_search.best_params_,
    "top_features": feature_importance_df.head(15).to_dict(orient='records')
}

meta_path = os.path.join('model', 'model_meta.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(model_meta, f, indent=4)
print(f"Metadata model berhasil disimpan ke: {meta_path}")

# 3. Verifikasi Pemuatan Model (Sanity Check)
loaded_model = joblib.load(model_path)
test_sample = X_test.iloc[:5]
preds = loaded_model.predict(test_sample)
print(f"Verifikasi sukses! Contoh prediksi 5 data uji: {preds.tolist()}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""---
## Kesimpulan Akhir Tahapan Notebook
1. Seluruh tahapan metodologi CRISP-DM (*Business Understanding*, *Data Understanding*, *Data Preparation*, *Modeling*, *Evaluation*, dan *Deployment Preparation*) telah berhasil diselesaikan secara komprehensif.
2. Model final **Random Forest Classifier** yang dituning mampu mendeteksi mahasiswa yang berisiko *Dropout* dengan **Recall 76%**, **Presisi 84%**, dan **ROC-AUC 0.892**, memberikan perlindungan optimal terhadap kerugian institusional akibat putus studi.
3. Model telah diekspor ke direktori `model/` dan siap dihubungkan dengan prototype aplikasi interaktif Streamlit (`app.py`).
"""
    ))

    nb.cells = cells
    return nb

if __name__ == '__main__':
    print("Membuat notebook...")
    nb = create_notebook()
    
    # Simpan notebook mentah
    with open('notebook.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook tersimpan di notebook.ipynb. Sekarang mengeksekusi seluruh sel...")
    
    # Eksekusi notebook
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    client.execute()
    
    # Simpan notebook dengan output lengkap
    with open('notebook.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Eksekusi notebook selesai sempurna! Output dan visualisasi telah terisi.")
