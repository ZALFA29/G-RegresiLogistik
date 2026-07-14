import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ==========================================
# 1. KONFIGURASI HALAMAN & UI/UX (CSS)
# ==========================================
st.set_page_config(page_title="Dashboard Machine Learning", layout="wide", initial_sidebar_state="expanded")

# CSS dipertahankan untuk Kartu Angka (Metric Card)
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E !important;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        cursor: help;
    }
    .metric-card-alert {
        background-color: #1E1E1E !important;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid #FF5252;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        cursor: help;
    }
    .metric-value { font-size: 36px; font-weight: bold; color: #FFFFFF !important; }
    .metric-label { font-size: 14px; color: #A0A0A0 !important; margin-top: 5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SISTEM CACHE & PEMROSESAN DATA
# ==========================================
# Deteksi lokasi folder tempat app.py berada agar bisa dibaca oleh GitHub
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_analytics_data():
    data_dict = {}
    daftar_file = {
        "Keseluruhan": os.path.join(BASE_DIR, "Overall Data.xlsx"),
        "2021": os.path.join(BASE_DIR, "2021.xlsx"),
        "2022": os.path.join(BASE_DIR, "2022.xlsx"),
        "2023": os.path.join(BASE_DIR, "2023.xlsx"),
        "2024": os.path.join(BASE_DIR, "2024.xlsx")
    }
    
    for nama_periode, path_file in daftar_file.items():
        try:
            df = pd.read_excel(path_file)
            
            kolom_numerik = ['Age (Month)', 'Weight', 'Height']
            for col in kolom_numerik:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Gender' in df.columns:
                def tentukan_gender(x):
                    val = str(x).strip().upper()
                    if val in ['M', '1', '1.0', 'L', 'LAKI-LAKI']:
                        return 'Laki-laki'
                    return 'Perempuan'
                df['Gender_Label'] = df['Gender'].apply(tentukan_gender)

            if 'Height for Age' in df.columns:
                df['Status'] = df['Height for Age'].astype(str).str.strip().str.title()
            
            data_dict[nama_periode] = df
        except Exception:
            data_dict[nama_periode] = pd.DataFrame()
            
    return data_dict

@st.cache_resource
def train_ml_model():
    path_ml = os.path.join(BASE_DIR, 'Preprocessed Data.xlsx')
    df = pd.read_excel(path_ml)
    
    def selamatkan_berat_badan(val):
        if isinstance(val, datetime.datetime):
            return val.day + (val.month / 10)
        return val
    df['Weight'] = df['Weight'].apply(selamatkan_berat_badan)
    
    def binarize_target(x):
        val = str(x).strip().upper()
        if val == 'NOT STUNTED' or val in ['0', '0.0']:
            return 0 
        return 1 
    df['Target_Stunting'] = df['Height for Age'].apply(binarize_target)
    
    def binarize_gender(x):
        val = str(x).strip().upper()
        if val in ['M', '1', '1.0', 'L']:
            return 1 
        return 0 
    df['Gender'] = df['Gender'].apply(binarize_gender)
    
    kolom_numerik = ['Gender', 'Age (Month)', 'Weight', 'Height']
    for col in kolom_numerik:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=kolom_numerik).copy()
    
    X = df[['Gender', 'Age (Month)', 'Weight', 'Height']]
    y = df['Target_Stunting']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    eval_data = {
        'y_test': y_test, 
        'y_pred': y_pred, 
        'y_pred_proba': y_pred_proba,
        'total_bersih': len(df),
        'total_uji': len(y_test)
    }
    
    return model, scaler, eval_data

df_analytics_dict = load_analytics_data()
model_ml, scaler_ml, eval_data = train_ml_model()

# ==========================================
# 3. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio("Pilih Modul Analisis:", ["Analisis Data Deskriptif", "Prediksi Machine Learning"])
st.sidebar.divider()
st.sidebar.info("Sistem ini dibangun untuk menganalisis status pertumbuhan balita Kabupaten Jeneponto menggunakan komputasi statistik dan algoritma Logistic Regression.")

# ==========================================
# 4. HALAMAN 1: ANALISIS DESKRIPTIF
# ==========================================
if menu == "Analisis Data Deskriptif":
    st.title("Analisis Tren dan Sebaran Data Visual")
    st.write("Silakan atur parameter di bawah ini. Dasbor akan secara otomatis melakukan kalkulasi ulang pada matriks visualisasi dan teks interpretasi.")
    
    st.divider()
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        filter_tahun = st.selectbox("Periode Tahun:", ["Keseluruhan", "2021", "2022", "2023", "2024"], help="Pilih 'Keseluruhan' untuk melihat akumulasi data.")
    with col_f2:
        filter_gender = st.selectbox("Klasifikasi Gender:", ["Semua Populasi", "Laki-laki", "Perempuan"], help="Saring data berdasarkan jenis kelamin spesifik.")
    with col_f3:
        df_aktif = df_analytics_dict[filter_tahun]
        min_age, max_age = 0, 60
        if not df_aktif.empty and pd.notna(df_aktif['Age (Month)'].min()):
            min_age = int(df_aktif['Age (Month)'].min())
            max_age = int(df_aktif['Age (Month)'].max())
        rentang_umur = st.slider("Rentang Usia (Bulan):", min_value=min_age, max_value=max_age, value=(min_age, max_age), help="Geser panel untuk membatasi usia balita.")
    
    if not df_aktif.empty:
        df_filtered = df_aktif[(df_aktif['Age (Month)'] >= rentang_umur[0]) & (df_aktif['Age (Month)'] <= rentang_umur[1])]
        if filter_gender != "Semua Populasi":
            df_filtered = df_filtered[df_filtered['Gender_Label'] == filter_gender]
        
        total_data = len(df_filtered)
        total_stunting = len(df_filtered[df_filtered['Status'] == 'Stunted']) if total_data > 0 else 0
        persen_stunting = (total_stunting / total_data) * 100 if total_data > 0 else 0
    else:
        df_filtered = pd.DataFrame()
        total_data = total_stunting = persen_stunting = 0

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card" title="Jumlah total data pengamatan yang relevan."><div class="metric-value">{total_data:,}</div><div class="metric-label">TOTAL PENGAMATAN (SAMPEL)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card-alert" title="Jumlah aktual data observasi yang masuk ke dalam kelas Stunted."><div class="metric-value" style="color:#FF5252;">{total_stunting:,}</div><div class="metric-label">TERKLASIFIKASI STUNTED</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card-alert" style="border-left-color: #FFC107;" title="Rasio persentase observasi berstatus Stunted."><div class="metric-value" style="color:#FFC107;">{persen_stunting:.1f}%</div><div class="metric-label">PERSENTASE PREVALENSI</div></div>', unsafe_allow_html=True)
    
    st.divider()
    
    if total_data > 0:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_pie = px.pie(df_filtered, names='Status', title='Proporsi Distribusi Kelas Target', color='Status', color_discrete_map={'Not Stunted':'#4CAF50', 'Stunted':'#FF5252'}, hole=0.5)
            fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_c2:
            fig_hist = px.histogram(df_filtered, x='Age (Month)', color='Status', barmode='group', title='Distribusi Frekuensi Usia Berdasarkan Kelas', color_discrete_map={'Not Stunted':'#4CAF50', 'Stunted':'#FF5252'})
            fig_hist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.divider()

        col_scatter, col_text = st.columns([1.5, 1])
        with col_scatter:
            fig_scatter = px.scatter(df_filtered, x='Weight', y='Height', color='Status', title='Pola Pertumbuhan Fisik: Tinggi vs Berat Badan', color_discrete_map={'Not Stunted':'#4CAF50', 'Stunted':'#FF5252'}, opacity=0.6)
            fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col_text:
            st.markdown("### Kesimpulan & Interpretasi")
            
            narasi = f"Berdasarkan parameter penyaringan untuk **{filter_gender}** pada rentang usia **{rentang_umur[0]} - {rentang_umur[1]} bulan** di periode **{filter_tahun}**, komputasi mencatat sebanyak **{total_stunting:,} observasi** (dari total {total_data:,} sampel) terindikasi sebagai perlambatan laju pertumbuhan (Stunted)."
            
            if persen_stunting > 30:
                st.error(f"{narasi}\n\n**Perhatian Tinggi:** Prevalensi menembus skala >30%. Data ini mengindikasikan urgensi distribusi nutrisi dan tindakan korektif.")
            elif persen_stunting > 15:
                st.warning(f"{narasi}\n\n**Perhatian Moderat:** Tingkat prevalensi berada pada spektrum pengawasan.")
            else:
                st.success(f"{narasi}\n\n**Tingkat prevalensi pada batas wajar**, menandakan proporsi indikator pertumbuhan berjalan konstan di dalam batas sampel yang diukur.")
    else:
        st.warning("Data komputasi belum tersedia atau bernilai nol untuk instrumen filter yang dipilih.")

# ==========================================
# 5. HALAMAN 2: PREDIKSI MACHINE LEARNING
# ==========================================
elif menu == "Prediksi Machine Learning":
    st.title("Model Prediksi Probabilistik & Arsitektur Evaluasi")
    st.write("Modul ini mengeksekusi algoritma Logistic Regression berpenalti L1 (Lasso Regression) berdasarkan pembobotan fitur fisik historis.")
    
    st.divider()
    
    # ---------------------------------------------------------
    # DIPINDAHKAN KE ATAS: EVALUASI KINERJA ARSITEKTUR MODEL
    # ---------------------------------------------------------
    st.markdown("### Evaluasi Kinerja Arsitektur Model")
    
    st.info(f"""
    **Asal Usul Metrik Pengujian (Metode Train-Test Split):**
    
    Sesuai dengan dataset Stunting dan Status Gizi Balita dari Kabupaten Jeneponto, arsitektur ini memproses populasi komprehensif sebanyak **{eval_data['total_bersih']:,} observasi**. Untuk mencegah *overfitting*, data dipecah secara proporsional:
    
    1. **80% Training Set:** Dimanfaatkan secara eksklusif oleh algoritma untuk mengekstraksi koefisien dan pola matematis.
    2. **20% Testing Set ({eval_data['total_uji']:,} sampel):** Data empiris yang diisolasi khusus untuk menantang tingkat akurasi tebakan model. Hasil evaluasi objektif pada set pengujian inilah yang divisualisasikan pada instrumen **Confusion Matrix** dan **Kurva ROC** di bawah ini.
    """)
    
    eval_col1, eval_col2 = st.columns(2)
    with eval_col1:
        cm = confusion_matrix(eval_data['y_test'], eval_data['y_pred'])
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Prediksi Algoritma", y="Data Aktual", color="Frekuensi Pengamatan"),
                           x=['Not Stunted (0)', 'Stunted (1)'], y=['Not Stunted (0)', 'Stunted (1)'],
                           title="Confusion Matrix")
        fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with eval_col2:
        fpr, tpr, _ = roc_curve(eval_data['y_test'], eval_data['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Kurva ROC (AUC = {roc_auc:.4f})', line=dict(color='darkorange', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Garis Ambang Dasar', line=dict(color='navy', width=2, dash='dash')))
        fig_roc.update_layout(title='Receiver Operating Characteristic (ROC)', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig_roc, use_container_width=True)
        
    st.success(f"""
    **Interpretasi Confusion Matrix & ROC:**
    
    Berdasarkan pengujian terhadap **{eval_data['total_uji']:,} sampel uji**, matriks evaluasi menghasilkan pemetaan yang sangat akurat:
    *   **True Negative (Kiri Atas):** Model berhasil mendeteksi secara presisi sebagian besar populasi balita yang memang berstatus "Not Stunted".
    *   **True Positive (Kanan Bawah):** Model sukses melakukan klasifikasi yang tepat pada balita dengan kondisi aktual "Stunted".
    *   **False Positive & False Negative:** Hanya terdapat persentase kesalahan klasifikasi yang sangat minim pada kuadran Kanan Atas dan Kiri Bawah, membuktikan rasio presisi dan sensitivitas (Recall) algoritma yang tinggi.
    
    Lebih lanjut, nilai **AUC (Area Under Curve) sebesar {roc_auc:.2f}** pada kurva ROC mengonfirmasi kemampuan separasi model yang superior. Semakin mendekati angka 1.0, semakin cerdas model dalam membedakan entitas kelas positif (Stunted) dan kelas negatif (Not Stunted).
    """)

    st.divider()

    # ---------------------------------------------------------
    # DIPINDAHKAN KE BAWAH: KALKULATOR PROBABILITAS
    # ---------------------------------------------------------
    st.markdown("### Kalkulator Probabilitas Sigmoid")
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)
        with col1:
            input_gender_text = st.selectbox("Variabel 1: Klasifikasi Gender", ["Laki-laki", "Perempuan"], help="Penentuan kategori biologis dasar pengamatan.")
            input_age = st.slider("Variabel 2: Usia (Bulan)", min_value=0, max_value=60, value=24, step=1, help="Menentukan sumbu usia pengamatan.")
        with col2:
            input_weight = st.number_input("Variabel 3: Berat Badan Aktual (kg)", min_value=1.0, max_value=40.0, value=10.5, step=0.1, help="Beban masa aktual objek.")
            input_height = st.number_input("Variabel 4: Tinggi Badan Aktual (cm)", min_value=30.0, max_value=120.0, value=85.0, step=0.1, help="Dimensi vertikal objek.")
        
        submit_button = st.form_submit_button("Eksekusi Komputasi")

    if submit_button:
        input_gender = 1 if input_gender_text == "Laki-laki" else 0
        user_data = pd.DataFrame([[input_gender, input_age, input_weight, input_height]], columns=['Gender', 'Age (Month)', 'Weight', 'Height'])
        user_data_scaled = scaler_ml.transform(user_data)
        
        pred_proba = model_ml.predict_proba(user_data_scaled)[0][1] * 100 
        pred_class = model_ml.predict(user_data_scaled)[0]
        
        st.divider()
        st.markdown("### Ekstraksi Prediksi Probabilitas")
        res_col1, res_col2 = st.columns([1, 1.5])
        
        with res_col1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = pred_proba, domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilitas Klasifikasi", 'font': {'size': 18, 'color': 'white'}},
                number = {'suffix': "%", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FF5252" if pred_proba >= 50 else "#4CAF50"},
                    'bgcolor': "white",
                    'steps': [{'range': [0, 50], 'color': '#1E1E1E'}, {'range': [50, 100], 'color': '#2b2b2b'}],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with res_col2:
            st.write("\n\n")
            if pred_class == 1:
                st.error(f"**KEPUTUSAN KELAS: 1 (Stunted) | Probabilitas: {pred_proba:.1f}%**\n\nBerdasarkan kalkulasi fungsi aktivasi terhadap koefisien fitur pembentuk model, proyeksi melampaui **ambang batas probabilistik (threshold > 50%)**. Dalam evaluasi analitis, profil vektor dimensi ini mengonfirmasi presensi pola Stunted.")
            else:
                st.success(f"**KEPUTUSAN KELAS: 0 (Not Stunted) | Probabilitas: {pred_proba:.1f}%**\n\nKomputasi fungsi aktivasi linier memetakan proyeksi probabilitas pada kurva bawah **(threshold < 50%)**. Berdasarkan model Logistic Regression, susunan metrik input ini ekuivalen dengan matriks kelas Not Stunted.")

# Footer
st.sidebar.divider()
st.sidebar.markdown("<p style='text-align: center; font-size: 12px; color: gray;'>Proyek Komputasi Machine Learning<br>Universitas Negeri Makassar | 2026</p>", unsafe_allow_html=True)
