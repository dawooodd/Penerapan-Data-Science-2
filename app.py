import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="Jaya Jaya Institut - Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    /* Main container and typography */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        color: #92400E;
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        color: #065F46;
    }
    .recommendation-box {
        background-color: #F0F9FF;
        border-left: 5px solid #0284C7;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .simulation-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
        border: 1px solid #C7D2FE;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Model & Metadata
# ---------------------------------------------------------
@st.cache_resource
def load_model_artifacts():
    model_path = os.path.join('model', 'model.joblib')
    meta_path = os.path.join('model', 'model_meta.json')
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
        
    model = joblib.load(model_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    return model, meta

model, model_meta = load_model_artifacts()


# ---------------------------------------------------------
# Feature Engineering Function
# ---------------------------------------------------------
def engineer_features(data_df):
    """
    Menambahkan fitur analitis yang konsisten dengan tahap Data Preparation.
    """
    df_feat = data_df.copy()
    
    sem1_enr = np.where(df_feat['Curricular_units_1st_sem_enrolled'] == 0, 1, df_feat['Curricular_units_1st_sem_enrolled'])
    sem2_enr = np.where(df_feat['Curricular_units_2nd_sem_enrolled'] == 0, 1, df_feat['Curricular_units_2nd_sem_enrolled'])
    
    df_feat['Approval_rate_1st'] = df_feat['Curricular_units_1st_sem_approved'] / sem1_enr
    df_feat['Approval_rate_2nd'] = df_feat['Curricular_units_2nd_sem_approved'] / sem2_enr
    
    tot_enr = df_feat['Curricular_units_1st_sem_enrolled'] + df_feat['Curricular_units_2nd_sem_enrolled']
    tot_enr_safe = np.where(tot_enr == 0, 1, tot_enr)
    tot_app = df_feat['Curricular_units_1st_sem_approved'] + df_feat['Curricular_units_2nd_sem_approved']
    
    df_feat['Total_enrolled'] = tot_enr
    df_feat['Total_approved'] = tot_app
    df_feat['Total_approval_rate'] = tot_app / tot_enr_safe
    
    df_feat['Grade_progression'] = df_feat['Curricular_units_2nd_sem_grade'] - df_feat['Curricular_units_1st_sem_grade']
    df_feat['Approved_diff'] = df_feat['Curricular_units_2nd_sem_approved'] - df_feat['Curricular_units_1st_sem_approved']
    
    df_feat['Financial_risk_index'] = (
        df_feat['Debtor'] * 2 + 
        (1 - df_feat['Tuition_fees_up_to_date']) * 3 - 
        df_feat['Scholarship_holder'] * 2
    )
    
    return df_feat


# Course mapping dictionary
COURSE_DICT = {
    171: "Animation and Multimedia Design",
    8014: "Social Service (Evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equiniculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening)"
}

APPLICATION_MODE_DICT = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2 (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}


# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=70)
    st.markdown("## **Jaya Jaya Institut**")
    st.markdown("*Student Retention & Early Warning System*")
    st.markdown("---")
    
    menu = st.radio(
        "Pilih Menu Navigasi:",
        [
            "🎯 Prediksi Mahasiswa Tunggal",
            "📁 Prediksi Massal (Batch CSV)",
            "🧪 Simulasi Kebijakan (What-If)",
            "📊 Performa Model & Fitur",
            "💡 Rekomendasi Bisnis Institusi"
        ]
    )
    
    st.markdown("---")
    if model_meta:
        st.markdown("### 📈 Ringkasan Metrik Model")
        st.caption(f"**Algoritma**: {model_meta['model_name']}")
        st.caption(f"**Akurasi**: {model_meta['metrics']['accuracy'] * 100:.1f}%")
        st.caption(f"**Recall Dropout**: {model_meta['metrics']['dropout_recall'] * 100:.1f}%")
        st.caption(f"**Presisi Dropout**: {model_meta['metrics']['dropout_precision'] * 100:.1f}%")
        st.caption(f"**ROC-AUC**: {model_meta['metrics']['roc_auc']:.3f}")
    
    st.markdown("---")
    st.caption("Dicoding Applied Data Science Final Project  \n© 2026 Jaya Jaya Institut")


# ---------------------------------------------------------
# Verification of Model Availability
# ---------------------------------------------------------
if model is None:
    st.error("⚠️ Model belum ditemukan di direktori `model/model.joblib`. Silakan periksa berkas model terlebih dahulu.")
    st.stop()


# ---------------------------------------------------------
# MENU 1: Prediksi Mahasiswa Tunggal
# ---------------------------------------------------------
if menu == "🎯 Prediksi Mahasiswa Tunggal":
    st.markdown('<div class="main-title">🎯 Deteksi Dini Risiko Mahasiswa</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sistem penilaian risiko putus kuliah (Dropout) secara personal untuk mendukung intervensi proaktif pembimbing akademik.</div>', unsafe_allow_html=True)

    # Preset Quick Profiles for Testing
    st.markdown("##### ⚡ Muat Contoh Profil Cepat (Quick Profiles):")
    col_pre1, col_pre2, col_pre3 = st.columns(3)
    
    with col_pre1:
        if st.button("⚠️ Muat Profil Berisiko Tinggi (High Risk)", use_container_width=True):
            st.session_state['profile'] = 'high_risk'
    with col_pre2:
        if st.button("🎓 Muat Profil Mahasiswa Lulus (Low Risk)", use_container_width=True):
            st.session_state['profile'] = 'low_risk'
    with col_pre3:
        if st.button("⚖️ Muat Profil Sedang / Aktif (Moderate)", use_container_width=True):
            st.session_state['profile'] = 'moderate'

    current_profile = st.session_state.get('profile', 'custom')
    
    if current_profile == 'high_risk':
        def_age = 32
        def_gender = 1 # Male
        def_marital = 2 # Married
        def_course = 9119 # Informatics
        def_app_mode = 39 # Over 23
        def_tuition = 0 # Unpaid
        def_debtor = 1 # Debtor
        def_scholar = 0 # No scholarship
        def_sem1_enr, def_sem1_app, def_sem1_grade = 6, 1, 9.5
        def_sem2_enr, def_sem2_app, def_sem2_grade = 6, 0, 0.0
    elif current_profile == 'low_risk':
        def_age = 19
        def_gender = 0 # Female
        def_marital = 1 # Single
        def_course = 9500 # Nursing
        def_app_mode = 1 # 1st phase general
        def_tuition = 1 # Paid
        def_debtor = 0 # No debt
        def_scholar = 1 # Scholarship holder
        def_sem1_enr, def_sem1_app, def_sem1_grade = 6, 6, 14.5
        def_sem2_enr, def_sem2_app, def_sem2_grade = 6, 6, 15.0
    elif current_profile == 'moderate':
        def_age = 22
        def_gender = 1
        def_marital = 1
        def_course = 9147 # Management
        def_app_mode = 17 # 2nd phase
        def_tuition = 1
        def_debtor = 0
        def_scholar = 0
        def_sem1_enr, def_sem1_app, def_sem1_grade = 6, 4, 11.2
        def_sem2_enr, def_sem2_app, def_sem2_grade = 6, 3, 10.5
    else:
        def_age = 20
        def_gender = 1
        def_marital = 1
        def_course = 9119
        def_app_mode = 1
        def_tuition = 1
        def_debtor = 0
        def_scholar = 0
        def_sem1_enr, def_sem1_app, def_sem1_grade = 6, 5, 12.0
        def_sem2_enr, def_sem2_app, def_sem2_grade = 6, 5, 12.5

    with st.form(key='student_form'):
        st.markdown("### 📋 Form Data Mahasiswa")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Data Demografis & Masuk",
            "💳 Status Finansial & Beasiswa",
            "📚 Akademik Semester 1",
            "📖 Akademik Semester 2"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Usia saat Pendaftaran (Age at enrollment):", min_value=17, max_value=70, value=def_age, step=1)
                gender = st.selectbox("Jenis Kelamin (Gender):", options=[(1, "Laki-laki"), (0, "Perempuan")], index=0 if def_gender == 1 else 1, format_func=lambda x: x[1])[0]
                marital = st.selectbox(
                    "Status Pernikahan:",
                    options=[(1, "Lajang"), (2, "Menikah"), (3, "Duda/Janda"), (4, "Cerai"), (5, "Hubungan de facto"), (6, "Pisah hukum")],
                    index=def_marital - 1,
                    format_func=lambda x: x[1]
                )[0]
                displaced = st.selectbox("Mahasiswa Perantau/Pindahan (Displaced):", options=[(1, "Ya"), (0, "Tidak")], index=0, format_func=lambda x: x[1])[0]
            with col2:
                course_keys = list(COURSE_DICT.keys())
                course_idx = course_keys.index(def_course) if def_course in course_keys else 0
                course = st.selectbox("Program Studi (Course):", options=course_keys, index=course_idx, format_func=lambda x: f"{x} - {COURSE_DICT[x]}")
                
                app_keys = list(APPLICATION_MODE_DICT.keys())
                app_idx = app_keys.index(def_app_mode) if def_app_mode in app_keys else 0
                application_mode = st.selectbox("Jalur Pendaftaran (Application Mode):", options=app_keys, index=app_idx, format_func=lambda x: f"{x} - {APPLICATION_MODE_DICT[x]}")
                
                attendance = st.selectbox("Jadwal Kuliah:", options=[(1, "Siang/Pagi"), (0, "Malam")], index=0, format_func=lambda x: x[1])[0]
                admission_grade = st.slider("Nilai Ujian Masuk (Admission Grade, skala 0-200):", min_value=80.0, max_value=200.0, value=125.0, step=0.5)

        with tab2:
            col3, col4 = st.columns(2)
            with col3:
                tuition_fees = st.selectbox(
                    "Kelancaran Pembayaran Uang Kuliah (SPP):",
                    options=[(1, "Lancar / Lunas"), (0, "Menunggak / Belum Lunas")],
                    index=0 if def_tuition == 1 else 1,
                    format_func=lambda x: x[1]
                )[0]
                debtor = st.selectbox(
                    "Status Utang Keuangan (Debtor):",
                    options=[(0, "Tidak Memiliki Utang"), (1, "Memiliki Catatan Utang")],
                    index=1 if def_debtor == 1 else 0,
                    format_func=lambda x: x[1]
                )[0]
            with col4:
                scholarship = st.selectbox(
                    "Status Penerima Beasiswa:",
                    options=[(1, "Penerima Beasiswa"), (0, "Bukan Penerima Beasiswa")],
                    index=0 if def_scholar == 1 else 1,
                    format_func=lambda x: x[1]
                )[0]
                special_needs = st.selectbox("Kebutuhan Khusus (Special Needs):", options=[(0, "Tidak"), (1, "Ya")], index=0, format_func=lambda x: x[1])[0]

        with tab3:
            col5, col6 = st.columns(2)
            with col5:
                sem1_enrolled = st.number_input("Mata Kuliah Terdaftar (Semester 1):", min_value=0, max_value=20, value=def_sem1_enr, step=1)
                sem1_approved = st.number_input("Mata Kuliah Lulus (Semester 1):", min_value=0, max_value=20, value=def_sem1_app, step=1)
            with col6:
                sem1_evaluations = st.number_input("Jumlah Evaluasi/Ujian Diikuti (Semester 1):", min_value=0, max_value=30, value=max(def_sem1_enr, 6), step=1)
                sem1_grade = st.number_input("Nilai Rata-rata Semester 1 (skala 0-20):", min_value=0.0, max_value=20.0, value=float(def_sem1_grade), step=0.1)

        with tab4:
            col7, col8 = st.columns(2)
            with col7:
                sem2_enrolled = st.number_input("Mata Kuliah Terdaftar (Semester 2):", min_value=0, max_value=20, value=def_sem2_enr, step=1)
                sem2_approved = st.number_input("Mata Kuliah Lulus (Semester 2):", min_value=0, max_value=20, value=def_sem2_app, step=1)
            with col8:
                sem2_evaluations = st.number_input("Jumlah Evaluasi/Ujian Diikuti (Semester 2):", min_value=0, max_value=30, value=max(def_sem2_enr, 6), step=1)
                sem2_grade = st.number_input("Nilai Rata-rata Semester 2 (skala 0-20):", min_value=0.0, max_value=20.0, value=float(def_sem2_grade), step=0.1)

        submit_btn = st.form_submit_button("🔍 Analisis & Prediksi Status Mahasiswa", use_container_width=True, type="primary")

    if submit_btn:
        student_data = {
            'Marital_status': marital,
            'Application_mode': application_mode,
            'Application_order': 1,
            'Course': course,
            'Daytime_evening_attendance': attendance,
            'Previous_qualification': 1,
            'Previous_qualification_grade': admission_grade,
            'Nacionality': 1,
            'Mothers_qualification': 19,
            'Fathers_qualification': 19,
            'Mothers_occupation': 5,
            'Fathers_occupation': 5,
            'Admission_grade': admission_grade,
            'Displaced': displaced,
            'Educational_special_needs': special_needs,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_fees,
            'Gender': gender,
            'Scholarship_holder': scholarship,
            'Age_at_enrollment': age,
            'International': 0,
            'Curricular_units_1st_sem_credited': 0,
            'Curricular_units_1st_sem_enrolled': sem1_enrolled,
            'Curricular_units_1st_sem_evaluations': sem1_evaluations,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_1st_sem_without_evaluations': 0,
            'Curricular_units_2nd_sem_credited': 0,
            'Curricular_units_2nd_sem_enrolled': sem2_enrolled,
            'Curricular_units_2nd_sem_evaluations': sem2_evaluations,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_grade': sem2_grade,
            'Curricular_units_2nd_sem_without_evaluations': 0,
            'Unemployment_rate': 11.1,
            'Inflation_rate': 1.4,
            'GDP': 0.5
        }
        
        raw_df = pd.DataFrame([student_data])
        prep_df = engineer_features(raw_df)
        
        # Predict
        raw_pred = model.predict(prep_df)[0]
        prediction = 'Dropout' if raw_pred in [1, '1', 'Dropout'] else 'Graduate'
        probabilities = model.predict_proba(prep_df)[0]
        
        # Probabilitas kelas biner: index 1 = Dropout, index 0 = Graduate
        classes_list = list(model.classes_)
        dropout_idx = classes_list.index(1) if 1 in classes_list else (classes_list.index('Dropout') if 'Dropout' in classes_list else 1)
        grad_idx = classes_list.index(0) if 0 in classes_list else (classes_list.index('Graduate') if 'Graduate' in classes_list else 0)
        
        dropout_prob = float(probabilities[dropout_idx])
        grad_prob = float(probabilities[grad_idx])
        
        st.markdown("---")
        st.markdown("### 📊 Hasil Analisis Prediksi")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        
        with res_col1:
            if prediction == 'Dropout' or dropout_prob >= 0.55:
                st.markdown(f"""
                <div class="risk-high">
                    <h3 style="margin:0; color:#991B1B;">⚠️ Status: Risiko Tinggi (Dropout)</h3>
                    <p style="margin-top:5px; font-size:1.05rem;">
                        Mahasiswa diprediksi memiliki probabilitas tinggi mengalami <b>Putus Kuliah</b>. Membutuhkan intervensi dan konseling segera.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            elif dropout_prob >= 0.30 and dropout_prob < 0.55:
                st.markdown(f"""
                <div class="risk-medium">
                    <h3 style="margin:0; color:#92400E;">🟡 Status: Risiko Sedang (Perlu Perhatian)</h3>
                    <p style="margin-top:5px; font-size:1.05rem;">
                        Mahasiswa berpotensi mengalami kerentanan akademik atau finansial yang memerlukan pendampingan dan pemantauan berkala.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3 style="margin:0; color:#065F46;">🎓 Status: Aman (Berpotensi Lulus)</h3>
                    <p style="margin-top:5px; font-size:1.05rem;">
                        Mahasiswa diprediksi memiliki performa prima dan berpeluang besar untuk <b>Lulus Tepat Waktu (Graduate)</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            st.write(f"**Probabilitas Dropout:** `{dropout_prob * 100:.1f}%`")
            st.progress(float(dropout_prob))
            
            st.write(f"**Probabilitas Lulus (Graduate):** `{grad_prob * 100:.1f}%`")
            st.progress(float(grad_prob))

        with res_col2:
            st.markdown("""
            <div class="recommendation-box">
                <h4 style="margin:0; color:#0369A1;">💡 Rekomendasi Tindakan Intervensi</h4>
                <ul style="margin-top:8px; font-size:0.95rem; line-height:1.6;">
            """, unsafe_allow_html=True)
            
            recommendations = []
            
            # Financial triggers
            if tuition_fees == 0 or debtor == 1:
                recommendations.append("<b>Konseling Finansial & Skema Cicilan</b>: Mahasiswa mengalami kendala pembayaran SPP / memiliki utang. Segera hubungi bagian keuangan untuk restrukturisasi pembayaran atau fasilitas bantuan darurat.")
            if scholarship == 0 and (tuition_fees == 0 or debtor == 1):
                recommendations.append("<b>Evaluasi Bantuan Beasiswa Keringanan</b>: Mahasiswa berisiko finansial namun belum menerima beasiswa. Pertimbangkan alokasi dana bantuan sosial kampus.")
                
            # Academic triggers
            total_approved = sem1_approved + sem2_approved
            total_enrolled = sem1_enrolled + sem2_enrolled
            approval_rate = total_approved / max(1, total_enrolled)
            
            if approval_rate < 0.6:
                recommendations.append(f"<b>Program Tutorial Sebaya (Peer Tutoring)</b>: Rasio kelulusan mata kuliah hanya {approval_rate*100:.0f}%. Wajibkan pendampingan belajar intensif untuk mata kuliah dasar.")
            if sem2_grade < sem1_grade:
                recommendations.append("<b>Monitoring Penurunan Performa</b>: Terjadi penurunan nilai antara semester 1 dan semester 2. Jadwalkan pertemuan dengan Dosen Pembimbing Akademik (DPA).")
            if age > 25:
                recommendations.append("<b>Dukungan Khusus Mahasiswa Dewasa (Mature Student)</b>: Berikan fleksibilitas jadwal bimbingan dan pendampingan manajemen waktu perkuliahan vs tanggung jawab pribadi.")
                
            if not recommendations:
                recommendations.append("<b>Pertahankan Momentum Positif</b>: Mahasiswa berada di jalur yang sangat baik. Terus dorong keterlibatan dalam program magang, organisasi, atau riset prestasi.")
                
            for rec in recommendations:
                st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
                
            st.markdown("</ul></div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# MENU 2: Prediksi Massal (Batch CSV)
# ---------------------------------------------------------
elif menu == "📁 Prediksi Massal (Batch CSV)":
    st.markdown('<div class="main-title">📁 Prediksi Massal Mahasiswa (Batch Assessment)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Unggah berkas CSV kohort mahasiswa untuk memetakan distribusi risiko putus studi secara serentak.</div>', unsafe_allow_html=True)
    
    col_up, col_dl = st.columns([2, 1])
    with col_dl:
        sample_df = pd.read_csv('data.csv', sep=';', encoding='utf-8-sig').head(10).drop(columns=['Status'], errors='ignore')
        sample_csv = sample_df.to_csv(index=False, sep=';')
        st.download_button(
            label="📥 Unduh Contoh Template CSV",
            data=sample_csv,
            file_name="template_mahasiswa_jaya_jaya.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    uploaded_file = st.file_uploader("Pilih file CSV data mahasiswa (menggunakan pemisah ';' atau ','):", type=['csv'])
    
    if uploaded_file is not None:
        try:
            sample_bytes = uploaded_file.read(2048).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            sep = ';' if ';' in sample_bytes else ','
            batch_df = pd.read_csv(uploaded_file, sep=sep)
            
            st.success(f"Berhasil memuat berkas dengan {len(batch_df)} baris data mahasiswa.")
            
            expected_raw_cols = model_meta['raw_features']
            missing_cols = [c for c in expected_raw_cols if c not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Berkas CSV tidak memiliki kolom wajib berikut: {missing_cols[:5]}...")
            else:
                with st.spinner("Menjalankan analisis dan pemodelan prediktif..."):
                    prep_batch = engineer_features(batch_df)
                    input_cols = model_meta['input_features']
                    X_batch = prep_batch[input_cols]
                    
                    batch_preds_raw = model.predict(X_batch)
                    target_map = {1: 'Dropout', 0: 'Graduate', '1': 'Dropout', '0': 'Graduate', 'Dropout': 'Dropout', 'Graduate': 'Graduate'}
                    batch_preds = [target_map.get(p, p) for p in batch_preds_raw]
                    batch_probs = model.predict_proba(X_batch)
                    
                    classes = list(model.classes_)
                    dropout_idx = classes.index(1) if 1 in classes else (classes.index('Dropout') if 'Dropout' in classes else 1)
                    dropout_probs = batch_probs[:, dropout_idx]
                    
                    def get_risk_tier(prob):
                        if prob >= 0.55:
                            return 'Tinggi (High)'
                        elif prob >= 0.30:
                            return 'Sedang (Medium)'
                        else:
                            return 'Rendah (Low)'
                            
                    risk_tiers = [get_risk_tier(p) for p in dropout_probs]
                    
                    result_df = batch_df.copy()
                    result_df['Predicted_Status'] = batch_preds
                    result_df['Dropout_Probability (%)'] = (dropout_probs * 100).round(2)
                    result_df['Risk_Level'] = risk_tiers
                    
                    st.markdown("### 📊 Ringkasan Risiko Kohort")
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    
                    high_risk_count = sum(1 for r in risk_tiers if 'Tinggi' in r)
                    med_risk_count = sum(1 for r in risk_tiers if 'Sedang' in r)
                    low_risk_count = sum(1 for r in risk_tiers if 'Rendah' in r)
                    
                    with kpi1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Total Mahasiswa</div>
                            <div class="metric-value">{len(result_df):,}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with kpi2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid #EF4444;">
                            <div class="metric-label" style="color:#DC2626;">Risiko Tinggi (High)</div>
                            <div class="metric-value" style="color:#DC2626;">{high_risk_count} ({high_risk_count/len(result_df)*100:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with kpi3:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid #F59E0B;">
                            <div class="metric-label" style="color:#D97706;">Risiko Sedang (Medium)</div>
                            <div class="metric-value" style="color:#D97706;">{med_risk_count} ({med_risk_count/len(result_df)*100:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with kpi4:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid #10B981;">
                            <div class="metric-label" style="color:#059669;">Risiko Rendah (Low)</div>
                            <div class="metric-value" style="color:#059669;">{low_risk_count} ({low_risk_count/len(result_df)*100:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("### 📋 Daftar Hasil Prediksi")
                    filter_risk = st.multiselect(
                        "Filter berdasarkan Tingkat Risiko:",
                        options=['Tinggi (High)', 'Sedang (Medium)', 'Rendah (Low)'],
                        default=['Tinggi (High)', 'Sedang (Medium)', 'Rendah (Low)']
                    )
                    
                    filtered_df = result_df[result_df['Risk_Level'].isin(filter_risk)]
                    display_cols = [
                        'Predicted_Status', 'Dropout_Probability (%)', 'Risk_Level',
                        'Age_at_enrollment', 'Gender', 'Course', 'Tuition_fees_up_to_date',
                        'Debtor', 'Scholarship_holder', 'Curricular_units_1st_sem_approved',
                        'Curricular_units_2nd_sem_approved'
                    ]
                    available_disp_cols = [c for c in display_cols if c in filtered_df.columns]
                    st.dataframe(filtered_df[available_disp_cols], use_container_width=True)
                    
                    res_csv = result_df.to_csv(index=False, sep=';')
                    st.download_button(
                        label="📥 Unduh Seluruh Hasil Prediksi (.CSV)",
                        data=res_csv,
                        file_name="hasil_prediksi_mahasiswa_jaya_jaya.csv",
                        mime="text/csv",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Gagal memproses file: {str(e)}")


# ---------------------------------------------------------
# MENU 3: Simulasi Kebijakan (What-If Simulation)
# ---------------------------------------------------------
elif menu == "🧪 Simulasi Kebijakan (What-If)":
    st.markdown('<div class="main-title">🧪 Simulasi Kebijakan Intervensi (What-If Sandbox)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Uji dampak berbagai skenario kebijakan bantuan sebelum mengeksekusinya pada mahasiswa secara nyata.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="simulation-box">
        <h4 style="margin:0 0 8px 0; color:#1E3A8A;">💡 Mengapa What-If Simulation Sangat Penting?</h4>
        <p style="margin:0; font-size:0.95rem; color:#374151;">
            Model machine learning tidak hanya berguna untuk diagnosis pasif, melainkan dapat digunakan sebagai <b>alat simulasi analitik preskriptif</b>. 
            Melalui simulasi ini, pimpinan Jaya Jaya Institut dapat melihat secara langsung berapa banyak mahasiswa yang berhasil diselamatkan dari <i>dropout</i> jika intervensi tertentu dijalankan.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load dataset sample for interactive simulation
    df_raw_sim = pd.read_csv('data.csv', sep=';', encoding='utf-8-sig')
    
    st.markdown("### 🎛️ Pengaturan Skenario Simulasi Kebijakan")
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("##### 💳 Skenario Bantuan Finansial:")
        opt_fin = st.checkbox("Aktifkan Program Relaksasi SPP & Penyelesaian Utang", value=True,
                              help="Mengubah status SPP menjadi lunas dan menghapus catatan utang bagi mahasiswa berisiko.")
        opt_scholar = st.checkbox("Berikan Beasiswa Bantuan Darurat untuk Mahasiswa Rentan", value=True,
                                  help="Menjadikan mahasiswa berisiko tinggi sebagai penerima beasiswa parsial.")
        
    with col_s2:
        st.markdown("##### 📚 Skenario Pendampingan Akademik (Peer Tutoring):")
        add_approved = st.slider("Peningkatan Jumlah Mata Kuliah Lulus di Semester 2:", min_value=0, max_value=4, value=2,
                                 help="Estimasi penambahan mata kuliah lulus berkat program tutor sebaya intensif.")
        add_grade = st.slider("Peningkatan Nilai Rata-rata Semester 2 (Poin, skala 0-20):", min_value=0.0, max_value=5.0, value=2.5, step=0.5,
                              help="Estimasi kenaikan nilai berkat klinik belajar gratis.")

    if st.button("🚀 Jalankan Simulasi What-If Sekarang", type="primary", use_container_width=True):
        with st.spinner("Menjalankan simulasi skenario counterfactual pada seluruh populasi mahasiswa..."):
            # 1. Baseline
            prep_base = engineer_features(df_raw_sim)
            cols = model_meta['input_features']
            y_base_pred = model.predict(prep_base[cols])
            y_base_prob = model.predict_proba(prep_base[cols])
            
            classes_list = list(model.classes_)
            dropout_idx = classes_list.index(1) if 1 in classes_list else (classes_list.index('Dropout') if 'Dropout' in classes_list else 1)
            p_base_dropout = y_base_prob[:, dropout_idx]
            base_dropout_count = int((y_base_pred == 1).sum() if 1 in y_base_pred else (y_base_pred == 'Dropout').sum())
            
            # 2. Simulated DataFrame
            df_sim = df_raw_sim.copy()
            
            if opt_fin:
                df_sim['Tuition_fees_up_to_date'] = 1
                df_sim['Debtor'] = 0
            if opt_scholar:
                # Apply scholarship to debtors or unpaid tuition
                debtor_mask = (df_raw_sim['Tuition_fees_up_to_date'] == 0) | (df_raw_sim['Debtor'] == 1)
                df_sim.loc[debtor_mask, 'Scholarship_holder'] = 1
                
            if add_approved > 0 or add_grade > 0:
                df_sim['Curricular_units_2nd_sem_approved'] = np.minimum(
                    df_sim['Curricular_units_2nd_sem_enrolled'],
                    df_sim['Curricular_units_2nd_sem_approved'] + add_approved
                )
                df_sim['Curricular_units_2nd_sem_grade'] = np.minimum(
                    20.0, df_sim['Curricular_units_2nd_sem_grade'] + add_grade
                )
                
            prep_sim = engineer_features(df_sim)
            y_sim_pred = model.predict(prep_sim[cols])
            y_sim_prob = model.predict_proba(prep_sim[cols])
            p_sim_dropout = y_sim_prob[:, dropout_idx]
            sim_dropout_count = int((y_sim_pred == 1).sum() if 1 in y_sim_pred else (y_sim_pred == 'Dropout').sum())
            
            saved_count = base_dropout_count - sim_dropout_count
            reduction_pct = (saved_count / base_dropout_count * 100) if base_dropout_count > 0 else 0
            
            # Financial Quantification
            tuition_fee = 6_000_000
            rem_sems = 4
            saved_revenue = saved_count * tuition_fee * rem_sems
            
            st.markdown("---")
            st.markdown("### 📈 Hasil Simulasi Intervensi Kebijakan")
            
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dropout Sebelum Kebijakan</div>
                    <div class="metric-value" style="color:#DC2626;">{base_dropout_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dropout Setelah Kebijakan</div>
                    <div class="metric-value" style="color:#059669;">{sim_dropout_count:,}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #10B981;">
                    <div class="metric-label" style="color:#059669;">Mahasiswa Diselamatkan</div>
                    <div class="metric-value" style="color:#059669;">{saved_count:,} ({reduction_pct:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with k4:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #3B82F6;">
                    <div class="metric-label" style="color:#2563EB;">Estimasi SPP Terselamatkan</div>
                    <div class="metric-value" style="color:#2563EB; font-size:1.3rem;">Rp {saved_revenue:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visual comparison
            st.markdown("##### 📊 Visualisasi Pergeseran Tingkat Risiko:")
            chart_df = pd.DataFrame({
                'Skenario': ['Sebelum Intervensi (Baseline)', 'Setelah Intervensi (What-If)'],
                'Jumlah Mahasiswa Dropout': [base_dropout_count, sim_dropout_count],
                'Rata-rata Probabilitas Dropout (%)': [round(p_base_dropout.mean() * 100, 1), round(p_sim_dropout.mean() * 100, 1)]
            })
            
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(chart_df.set_index('Skenario')['Jumlah Mahasiswa Dropout'], color="#1E3A8A")
            with c2:
                st.bar_chart(chart_df.set_index('Skenario')['Rata-rata Probabilitas Dropout (%)'], color="#10B981")


# ---------------------------------------------------------
# MENU 4: Performa Model & Fitur
# ---------------------------------------------------------
elif menu == "📊 Performa Model & Fitur":
    st.markdown('<div class="main-title">📊 Evaluasi Performa Model Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Transparansi metrik evaluasi model Random Forest Classifier, audit keadilan (fairness), dan faktor penentu prediksi.</div>', unsafe_allow_html=True)
    
    metrics = model_meta['metrics']
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Akurasi Keseluruhan", f"{metrics['accuracy']*100:.2f}%", help="Proporsi prediksi benar untuk seluruh kelas")
    with col_m2:
        st.metric("Dropout Recall", f"{metrics['dropout_recall']*100:.2f}%", help="Sensitivitas model dalam mendeteksi mahasiswa yang berisiko dropout")
    with col_m3:
        st.metric("Dropout Presisi", f"{metrics['dropout_precision']*100:.2f}%", help="Ketepatan prediksi mahasiswa yang dilabeli dropout")
    with col_m4:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}", help="Kemampuan diskriminasi probabilitas model klasifikasi biner")
        
    st.markdown("---")
    
    col_feat1, col_feat2 = st.columns([1.2, 1])
    with col_feat1:
        st.markdown("### 🏆 Top 10 Fitur Paling Berpengaruh")
        st.caption("Tingkat kepentingan relatif fitur (Gini Feature Importance & SHAP Values) dalam memprediksi status mahasiswa.")
        top_f_df = pd.DataFrame(model_meta['top_features']).head(10)
        st.bar_chart(top_f_df.set_index('Feature')['Importance'], color="#1E3A8A")
        
    with col_feat2:
        st.markdown("### ⚖️ Audit Keadilan Model (Fairness)")
        st.markdown("""
        Pemeriksaan bias dilakukan terhadap atribut demografis yang dilindungi:
        - **Bias Gender**: Model memprediksi tingkat dropout 40,3% pada laki-laki (aktual 46,0%) dan 23,2% pada perempuan (aktual 24,4%). Model tidak memperbesar risiko pada gender tertentu dan mematuhi prinsip keadilan.
        - **Keseimbangan Recall (Equal Opportunity)**: Recall deteksi dropout pada laki-laki (77,2%) dan perempuan (73,4%) seimbang (< 4% selisih).
        - **Bias Usia**: Distribusi probabilitas bervariasi kontinu mencerminkan beban kerja dan keluarga nyata, bukan bias algoritmik.
        """)


# ---------------------------------------------------------
# MENU 5: Rekomendasi Bisnis Institusi
# ---------------------------------------------------------
elif menu == "💡 Rekomendasi Bisnis Institusi":
    st.markdown('<div class="main-title">💡 Rekomendasi Aksi Strategis (Action Items)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Panduan langkah konkret berbasis data bagi manajemen Jaya Jaya Institut untuk mereduksi angka putus studi secara berkelanjutan.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. 🛡️ Pembentukan Sistem Peringatan Dini Akademik (Academic Early Warning System)
    - **Tindakan**: Integrasikan model machine learning ini ke dalam Sistem Informasi Akademik (SIAKAD) kampus.
    - **Mekanisme**: Setiap akhir semester 1 dan semester 2, jalankan kalkulasi *Dropout Risk Score* secara otomatis.
    - **Target**: Memberikan sinyal otomatis kepada Dosen Pembimbing Akademik (DPA) untuk mahasiswa dengan skor risiko > 50%.
    
    ---
    
    ### 2. 💳 Skema Penyelamatan Finansial Fleksibel (Financial Relief Program)
    - **Tindakan**: Mengingat 86,5% mahasiswa yang menunggak SPP berujung *dropout*, institusi harus menyediakan opsi pembayaran bertahap (cicilan tanpa denda) dan beasiswa darurat (*emergency micro-grants*).
    - **Mekanisme**: Mahasiswa yang terdeteksi memiliki utang atau tunggakan SPP tidak langsung diskors, melainkan dijadwalkan untuk sesi konseling finansial bersama unit beasiswa kampus.
    
    ---
    
    ### 3. 👥 Program Tutorial Sebaya & Klinik Akademik (Peer Tutoring Program)
    - **Tindakan**: Membuka klinik belajar gratis dan mentoring sebaya bagi mahasiswa semester 1-2 yang memiliki nilai di bawah ambang batas (grade < 10 dari skala 20).
    - **Fokus Mata Kuliah**: Fokuskan tutor pada mata kuliah dengan tingkat kegagalan (*failure rate*) tertinggi di setiap program studi.
    
    ---
    
    ### 4. ⏰ Pendampingan Khusus Mahasiswa Dewasa (Mature & Evening Student Support)
    - **Tindakan**: Memberikan dukungan adaptasi akademik dan manajemen waktu untuk mahasiswa yang mendaftar pada usia > 25 tahun atau kelas malam.
    - **Fasilitas**: Penyediaan rekaman perkuliahan (*asynchronous learning*) dan jam bimbingan konseling fleksibel di luar jam kerja reguler.
    
    ---
    
    ### 5. 🎯 Evaluasi Beban SKS Semester Awal
    - **Tindakan**: Menyesuaikan beban kredit maksimum bagi mahasiswa yang memiliki riwayat nilai seleksi masuk rendah. Membatasi pengambilan beban SKS berlebih di semester 2 jika semester 1 belum mencapai rasio kelulusan 75%.
    """)
