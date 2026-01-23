import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from google import genai
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Running Coach", layout="wide")

# --- 2. FUNGSI AUTHENTICATION (OAUTH 2.0) ---
def strava_oauth_button():
    # Coba ambil secrets, jika tidak ada (user biasa), return None tanpa error
    try:
        client_id = st.secrets.get("STRAVA_CLIENT_ID")
        client_secret = st.secrets.get("STRAVA_CLIENT_SECRET")
        if not client_id or not client_secret: return None
    except:
        return None
    
    # GANTI DENGAN LINK APP KAMU JIKA SUDAH ONLINE
    # redirect_uri = "https://ai-running-coach.streamlit.app" 
    redirect_uri = "http://localhost:8501" # Pakai ini kalau masih di laptop
    
    query_params = st.query_params
    auth_url = f"https://www.strava.com/oauth/authorize?client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&approval_prompt=force&scope=activity:read_all"
    
    if "strava_token" not in st.session_state:
        if "code" in query_params:
            code = query_params["code"]
            tukar_token_otomatis(code, client_id, client_secret)
            st.rerun()
        else:
            st.link_button("🟠 Login dengan Akun Strava (Admin)", auth_url)
    else:
        st.sidebar.success("✅ Terhubung via API")
        if st.sidebar.button("Logout API"):
            del st.session_state["strava_token"]
            st.rerun()
        return st.session_state["strava_token"]
    return None

def tukar_token_otomatis(auth_code, client_id, client_secret):
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': auth_code,
        'grant_type': 'authorization_code'
    }
    try:
        res = requests.post("https://www.strava.com/oauth/token", data=payload)
        if res.status_code == 200:
            st.session_state["strava_token"] = res.json()['access_token']
            st.query_params.clear()
        else:
            st.error("Gagal login API.")
    except Exception as e:
        st.error(f"Error koneksi: {e}")

# --- 3. FUNGSI PENGOLAH DATA (CLEANING) ---
def process_dataframe(df_raw, source_type="api"):
    """Membersihkan data baik dari API maupun CSV"""
    df = df_raw.copy()
    
    # 1. Normalisasi Nama Kolom (Agar CSV dan API dianggap sama)
    col_mapping = {
        'Distance': 'distance', 
        'Moving Time': 'moving_time', 
        'Elapsed Time': 'moving_time',
        'Activity Date': 'start_date_local',
        'Activity Type': 'type',
        'Average Speed': 'average_speed'
    }
    df.rename(columns=col_mapping, inplace=True)
    
    # 2. Filter hanya Lari
    if 'type' in df.columns:
        df = df[df['type'].astype(str).str.contains('Run|Lari', case=False, na=False)]
    
    # 3. Pastikan kolom numerik aman (Hapus koma jika ada string "1,000")
    for col in ['distance', 'moving_time']:
        if col in df.columns and df[col].dtype == object:
             df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    # 4. Feature Engineering
    if 'distance' in df.columns and 'moving_time' in df.columns:
        df['distance_km'] = df['distance'] / 1000
        df['duration_min'] = df['moving_time'] / 60
        df['pace'] = df.apply(lambda x: x['duration_min'] / x['distance_km'] if x['distance_km'] > 0 else 0, axis=1)
    
    # 5. Tanggal
    if 'start_date_local' in df.columns:
        df['date'] = pd.to_datetime(df['start_date_local'], errors='coerce').dt.date
        df['date_obj'] = pd.to_datetime(df['start_date_local'], errors='coerce')
        df = df.dropna(subset=['date_obj']).sort_values(by='date_obj').reset_index(drop=True)
        
    return df

def ambil_data_strava_api(token):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {'Authorization': f'Bearer {token}'}
    params = {'per_page': 50, 'page': 1}
    try:
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200:
            return process_dataframe(pd.DataFrame(res.json()), source_type="api")
    except Exception as e:
        st.error(f"Error API: {e}")
    return None

def prediksi_pace_ml(df):
    if len(df) < 2: return 0, 0, None, None, None
    
    df_ml = df.copy()
    df_ml = df_ml[df_ml['pace'] > 0] # Hapus pace 0
    df_ml = df_ml[df_ml['pace'] < 30] # Hapus data error
    
    df_ml['run_index'] = np.arange(len(df_ml))
    X = df_ml[['run_index']]
    y = df_ml['pace']
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_index = len(df_ml)
    pred_pace = model.predict([[next_index]])[0]
    trend = model.coef_[0]
    
    return pred_pace, trend, model, X, y

def tanya_gemini(data_summary, ml_prediction):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Analisis data lari ini:
        [DATA]: {data_summary}
        [PREDIKSI ML]: {ml_prediction} min/km
        Berikan 3 saran latihan spesifik untuk minggu depan.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Maaf, AI sedang sibuk atau limit habis. Error: {e}"

# --- 4. TAMPILAN UTAMA (FRONTEND) ---
st.title("🏃‍♂️ AI Running Coach: Upload & Analyze")
st.markdown("Analisis performa lari Anda menggunakan **Machine Learning** dan **Gemini AI**.")

# --- SIDEBAR UTAMA ---
with st.sidebar:
    st.header("📂 Input Data")
    
    # OPSI 1: UPLOAD CSV
    uploaded_file = st.file_uploader("Upload File CSV Strava", type=["csv"])
    
    st.divider()
    st.markdown("**Atau**")
    
    # OPSI 2: LOGIN API (Opsional)
    token_api = strava_oauth_button()
    
    # Tombol Download Template
    st.divider()
    sample_data = pd.DataFrame({
        'start_date_local': ['2024-01-01 07:00:00', '2024-01-03 17:00:00'],
        'type': ['Run', 'Run'],
        'distance': [5000, 10000],
        'moving_time': [1800, 3600],
        'average_heartrate': [145, 150]
    })
    st.download_button(
        label="📥 Download Contoh Format CSV",
        data=sample_data.to_csv(index=False),
        file_name="contoh_data_lari.csv",
        mime="text/csv"
    )

# --- LOGIKA PEMILIHAN DATA ---
df = None

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        df = process_dataframe(df_raw, source_type="csv")
        st.info("📂 Menggunakan data dari File CSV yang diupload.")
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")

elif token_api:
    with st.spinner("Mengambil data dari Strava API..."):
        df = ambil_data_strava_api(token_api)
        st.success("☁️ Menggunakan data Live dari Strava API.")

# --- TAMPILKAN HASIL (GABUNGAN LAYOUT) ---
if df is not None and not df.empty:
    
    # Hitung ML
    pred_pace, trend, model, X, y = prediksi_pace_ml(df)
    
    # --- BAGIAN INI ADALAH LAYOUT YANG KAMU REQUEST ---
    tab1, tab2, tab3 = st.tabs(["📊 Statistik", "🔮 Prediksi ML", "🤖 Rekomendasi AI"])
    
    with tab1:
        # Metrik dengan Caption Lengkap
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Lari", f"{len(df)} kali")
            st.caption("Jumlah total sesi lari yang berhasil terekam dan dianalisis dari data Anda.")

        with c2:
            st.metric("Jarak Total", f"{df['distance_km'].sum():.1f} km")
            st.caption("Akumulasi seluruh jarak lari yang telah Anda tempuh sejauh ini.")

        with c3:
            st.metric("Pace Rata-rata", f"{df['pace'].mean():.2f} min/km")
            st.caption("Rata-rata kecepatan lari Anda. Ingat: Angka makin KECIL berarti makin CEPAT.")
        
        st.divider()
        
        # Grafik Pace dengan Penjelasan
        st.subheader("Grafik Perkembangan Pace")
        st.info("""
        **Cara Membaca Grafik ini:**
        * **Garis Biru:** Menunjukan perubahan **Pace** Anda dari waktu ke waktu.
        * **Arah:** Karena sumbu Y dibalik, jika garisnya bergerak ke **ATAS**, artinya lari Anda semakin **CEPAT**. Sebaliknya, jika turun, artinya melambat.
        * **Konsistensi:** Jika garisnya mendatar (rata), berarti performa Anda stabil.
        """)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df, x='date', y='pace', marker='o', ax=ax)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        with st.expander("🔍 Lihat Data Mentah"):
            st.dataframe(df)

    with tab2:
        st.subheader("🔮 Prediksi Performa Masa Depan")
        
        if model:
            # Kolom Hasil Prediksi
            col_pred1, col_pred2 = st.columns(2)
            
            # Logika Warna & Pesan Tren
            delta_color = "normal"
            if trend < 0: # Negatif = Pace turun (cepat)
                pesan_tren = "Tren MEMBAIK (Makin Cepat) 🚀"
                delta_color = "inverse" # Hijau
            else:
                pesan_tren = "Tren MEMBURUK (Melambat) 🐢"
                delta_color = "off"
            
            with col_pred1:
                st.metric("Prediksi Pace Lari Selanjutnya", f"{pred_pace:.2f} min/km", delta=f"Tren: {trend:.4f}", delta_color=delta_color)
            
            with col_pred2:
                st.info(f"Analisis Tren: **{pesan_tren}**")
            
            st.divider()
            
            # Grafik ML dengan Penjelasan Markdown
            st.write("### Visualisasi Garis Regresi (Machine Learning)")
            st.markdown("""
            Grafik ini menunjukkan bagaimana **Machine Learning** bekerja:
            * 🔵 **Titik Biru (Data Asli):** Sebaran data lari Anda yang sebenarnya.
            * 🔴 **Garis Merah Putus-putus (Trendline):** Garis lurus terbaik yang ditarik mesin untuk melihat pola kemajuan Anda.
            * 🟢 **Titik Hijau Besar (Prediksi):** Tebakan mesin untuk lari ke-n selanjutnya.
            """)
            
            fig_ml, ax_ml = plt.subplots(figsize=(10, 4))
            ax_ml.scatter(X, y, color='blue', label='Data Aktual')
            ax_ml.plot(X, model.predict(X), color='red', linestyle='--', label='Garis Tren ML')
            ax_ml.scatter(len(df), pred_pace, color='green', s=100, label='Prediksi Next Run', zorder=5)
            
            ax_ml.set_xlabel("Urutan Lari")
            ax_ml.set_ylabel("Pace (min/km)")
            ax_ml.invert_yaxis()
            ax_ml.legend()
            ax_ml.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(fig_ml)
            
            st.caption("*Garis merah putus-putus menunjukkan arah perkembangan lari Anda. Jika garisnya miring ke atas (nilai pace di sumbu Y mengecil), berarti Anda semakin cepat.*")
        else:
            st.warning("Data lari kurang cukup untuk prediksi (Minimal 2 data).")

    with tab3:
        st.subheader("🤖 Coach Gemini Analysis")
        if st.button("Minta Analisis AI"):
            with st.spinner("AI sedang membaca data Anda..."):
                summary_json = df[['distance_km', 'pace']].describe().to_json()
                hasil_ai = tanya_gemini(summary_json, f"{pred_pace:.2f}")
                st.markdown(hasil_ai)

elif uploaded_file is None and not token_api:
    # Tampilan Awal (Landing Page)
    st.markdown("""
    ### 👋 Selamat Datang!
    Untuk memulai analisis, silakan pilih salah satu metode di menu sebelah kiri (Sidebar):
    
    1. **Upload CSV:** Jika Anda punya file data lari (Bisa digunakan oleh SIAPA SAJA).
    2. **Login Strava:** Khusus Admin/Developer untuk koneksi langsung.
    """)