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
    # Ambil credential APLIKASI dari Secrets
    try:
        client_id = st.secrets["STRAVA_CLIENT_ID"]
        client_secret = st.secrets["STRAVA_CLIENT_SECRET"]
    except KeyError:
        st.error("❌ Secrets belum disetting! Masukkan Client ID & Secret di Streamlit Cloud.")
        return None
    
    # URL Redirect (Otomatis deteksi apakah di localhost atau cloud)
    # Trik: Mengambil URL asli dari browser user
    # Jika error di local, ganti manual jadi "http://localhost:8501"
    redirect_uri = "https://stravatraining.streamlit.app" # GANTI DENGAN LINK APP KAMU JIKA SUDAH ONLINE
    
    # Cek query params untuk menangkap kode dari Strava
    query_params = st.query_params
    
    auth_url = f"https://www.strava.com/oauth/authorize?client_id={client_id}&response_type=code&redirect_uri={redirect_uri}&approval_prompt=force&scope=activity:read_all"
    
    # Jika belum ada token di session, tampilkan tombol login
    if "strava_token" not in st.session_state:
        # Cek apakah Strava baru saja melempar balik dengan 'code'?
        if "code" in query_params:
            code = query_params["code"]
            tukar_token_otomatis(code, client_id, client_secret)
            # Rerun agar tampilan refresh dan tombol login hilang
            st.rerun()
        else:
            # TAMPILAN TOMBOL LOGIN
            st.link_button("🟠 Connect with Strava (Login)", auth_url)
            st.caption("Klik tombol di atas. Anda akan diarahkan ke Strava untuk izin akses.")
            return None
    else:
        # TAMPILAN JIKA SUDAH LOGIN
        st.sidebar.success("✅ Terhubung sebagai User")
        if st.sidebar.button("Logout"):
            del st.session_state["strava_token"]
            st.rerun()
        return st.session_state["strava_token"]

def tukar_token_otomatis(auth_code, client_id, client_secret):
    """Menukar Authorization Code dengan Access Token User di background"""
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': auth_code,
        'grant_type': 'authorization_code'
    }
    with st.spinner("Sedang login ke Strava..."):
        res = requests.post("https://www.strava.com/oauth/token", data=payload)
        if res.status_code == 200:
            data = res.json()
            # Simpan token USER ke session sementara (hilang saat tab ditutup)
            st.session_state["strava_token"] = data['access_token']
            # Bersihkan URL bar
            st.query_params.clear()
        else:
            st.error(f"Gagal login. Error Strava: {res.text}")

# --- 3. FUNGSI LOGIKA (BACKEND) ---
def ambil_data_strava(token):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {'Authorization': f'Bearer {token}'}
    params = {'per_page': 50, 'page': 1}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if not data: return None
            
            df = pd.DataFrame(data)
            if 'type' in df.columns:
                df = df[df['type'] == 'Run'].copy()
            
            cols = ['name', 'start_date_local', 'distance', 'moving_time', 'average_speed', 'total_elevation_gain']
            for c in cols:
                if c not in df.columns: df[c] = 0
            df = df[cols].copy()
            
            # Feature Engineering
            df['distance_km'] = df['distance'] / 1000
            df['duration_min'] = df['moving_time'] / 60
            df['pace'] = df.apply(lambda x: x['duration_min'] / x['distance_km'] if x['distance_km'] > 0 else 0, axis=1)
            df['date'] = pd.to_datetime(df['start_date_local']).dt.date
            df['date_obj'] = pd.to_datetime(df['start_date_local'])
            df = df.sort_values(by='date_obj').reset_index(drop=True)
            return df
        else:
            st.error(f"Error API Strava: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Terjadi kesalahan koneksi: {e}")
        return None

def prediksi_pace_ml(df):
    if len(df) < 2: return 0, 0, None, None, None # Data terlalu sedikit
    
    df_ml = df.copy()
    df_ml['run_index'] = np.arange(len(df_ml))
    X = df_ml[['run_index']]
    y = df_ml['pace']
    model = LinearRegression()
    model.fit(X, y)
    next_run_index = len(df_ml)
    pred_pace = model.predict([[next_run_index]])[0]
    trend = model.coef_[0]
    return pred_pace, trend, model, X, y

def tanya_gemini(data_summary, ml_prediction):
    # Ambil kunci APLIKASI (Punyamu) dari Secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Sebagai pelatih lari AI, analisis data ini:
        [STATISTIK]: {data_summary}
        [PREDIKSI ML]: Pace Lari Berikutnya: {ml_prediction} min/km
        Berikan evaluasi tren, 3 saran spesifik, dan jadwal latihan.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Maaf, AI sedang sibuk atau limit habis. Error: {e}"

# --- 4. TAMPILAN UTAMA (FRONTEND) ---
st.title("🏃‍♂️ AI Running Coach")
st.markdown("Login dengan Strava Anda, dan biarkan AI menganalisis performa lari secara otomatis.")

# --- SIDEBAR: LOGIN AREA ---
with st.sidebar:
    st.header("🔓 Akses Data")
    token = strava_oauth_button()
    st.divider()
    st.caption("Aplikasi ini menggunakan Strava API & Google Gemini AI.")

# --- MAIN CONTENT ---
if token: 
    with st.spinner("Mengambil data lari Anda..."):
        df = ambil_data_strava(token)
        
    if df is not None and not df.empty:
        # TABS
        tab1, tab2, tab3 = st.tabs(["📊 Statistik", "🔮 Prediksi ML", "🤖 Rekomendasi AI"])
        
        with tab1:
            col1, col2 = st.columns(2)
            col1.metric("Total Lari", f"{len(df)} kali")
            col1.metric("Pace Rata-rata", f"{df['pace'].mean():.2f} min/km")
            st.subheader("Grafik Pace")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=df, x='date', y='pace', marker='o', ax=ax)
            ax.invert_yaxis()
            st.pyplot(fig)
            
        with tab2:
            pred_pace, trend, model, X, y = prediksi_pace_ml(df)
            if model:
                st.metric("Prediksi Pace Selanjutnya", f"{pred_pace:.2f} min/km")
                fig_ml, ax_ml = plt.subplots(figsize=(10, 4))
                ax_ml.scatter(X, y, color='blue', alpha=0.5)
                ax_ml.plot(X, model.predict(X), color='red', linestyle='--')
                ax_ml.invert_yaxis()
                st.pyplot(fig_ml)
            else:
                st.warning("Data belum cukup untuk prediksi Machine Learning.")

        with tab3:
            st.subheader("🤖 Coach Gemini Analysis")
            if model:
                with st.spinner("AI sedang berpikir..."):
                    summary_json = df[['distance_km', 'pace']].describe().to_json()
                    hasil_ai = tanya_gemini(summary_json, f"{pred_pace:.2f}")
                    st.markdown(hasil_ai)
    elif df is not None:
        st.warning("Data ditemukan tapi tidak ada aktivitas lari (Run).")
else:
    st.info("👈 Silakan login menggunakan tombol di Sidebar untuk memulai.")