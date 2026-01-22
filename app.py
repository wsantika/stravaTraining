import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from google import genai
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression # Import Machine Learning

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AI Running Coach", layout="wide")

st.title("🏃‍♂️ AI Running Coach: Analisis & Prediksi Pace")
st.markdown("""
Aplikasi ini membantu pelari pemula menganalisis data Strava, mendapatkan rekomendasi 
latihan dari **Gemini AI**, dan memprediksi performa masa depan dengan **Machine Learning**.
""")

# --- FUNGSI BANTUAN AUTO-LOAD TOKEN ---
def load_tokens_from_file():
    if os.path.exists('strava_keys.json'):
        try:
            with open('strava_keys.json', 'r') as f:
                data = json.load(f)
                return data.get('access_token', '')
        except:
            return ""
    return ""

# --- 2. SIDEBAR (INPUT) ---
st.sidebar.header("🔑 Konfigurasi API")
default_token = load_tokens_from_file()
strava_token = st.sidebar.text_input("Strava Access Token", value=default_token, type="password")
gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
btn_proses = st.sidebar.button("🚀 Mulai Analisis & Prediksi")

# --- 3. FUNGSI LOGIKA (BACKEND) ---
def ambil_data_strava(token):
    url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {'Authorization': f'Bearer {token}'}
    params = {'per_page': 50, 'page': 1} # Ambil 50 data terakhir
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        
        # Filter hanya lari
        if 'type' in df.columns:
            df = df[df['type'] == 'Run'].copy()
        
        # Kolom yang diperlukan
        cols = ['name', 'start_date_local', 'distance', 'moving_time', 'average_speed', 'total_elevation_gain', 'average_heartrate']
        for c in cols:
            if c not in df.columns: df[c] = 0
            
        df = df[cols].copy()
        
        # Feature Engineering
        df['distance_km'] = df['distance'] / 1000
        df['duration_min'] = df['moving_time'] / 60
        df['pace'] = df.apply(lambda x: x['duration_min'] / x['distance_km'] if x['distance_km'] > 0 else 0, axis=1)
        df['date'] = pd.to_datetime(df['start_date_local']).dt.date
        df['date_obj'] = pd.to_datetime(df['start_date_local']) # Untuk sorting
        
        # Urutkan dari yang terlama ke terbaru (Penting untuk ML Time Series)
        df = df.sort_values(by='date_obj').reset_index(drop=True)
        
        return df
    else:
        st.error(f"Gagal ambil data Strava. Error: {response.status_code}")
        return None

def prediksi_pace_ml(df):
    """
    Fungsi ini melakukan prediksi pace menggunakan Linear Regression
    Berdasarkan logika dari file prediksi_pace.py
    """
    # Siapkan data untuk Machine Learning
    df_ml = df.copy()
    
    # Kita gunakan index urutan lari sebagai 'waktu' (Lari ke-1, ke-2, dst)
    df_ml['run_index'] = np.arange(len(df_ml))
    
    X = df_ml[['run_index']] # Fitur: Urutan Lari
    y = df_ml['pace']        # Target: Pace
    
    # Latih Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Prediksi untuk lari berikutnya (Next Run)
    next_run_index = len(df_ml)
    prediksi_pace_besok = model.predict([[next_run_index]])[0]
    
    # Hitung Tren (Koefisien kemiringan garis)
    # Jika negatif = Pace makin kecil (makin cepat) = BAGUS
    trend = model.coef_[0]
    
    return prediksi_pace_besok, trend, model, X, y

def tanya_gemini(data_summary, ml_prediction, api_key):
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
        Sebagai pelatih lari AI, analisis data ini:
        
        [STATISTIK HISTORIS]:
        {data_summary}
        
        [PREDIKSI MACHINE LEARNING]:
        - Prediksi Pace Lari Berikutnya: {ml_prediction} menit/km
        
        Tugas:
        1. Evaluasi apakah tren performa pelari ini membaik atau memburuk.
        2. Berikan 3 saran spesifik untuk mencapai prediksi tersebut atau melampauinya.
        3. Buat jadwal latihan minggu depan.
        """
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"Error Gemini: {e}"

# --- 4. TAMPILAN UTAMA (FRONTEND) ---
if btn_proses:
    if not strava_token:
        st.warning("⚠️ Masukkan Token Strava dulu!")
    else:
        with st.spinner("Sedang mengambil & menganalisis data..."):
            df = ambil_data_strava(strava_token)
            
        if df is not None:
            # --- TABBED LAYOUT (Supaya Rapi) ---
            tab1, tab2, tab3 = st.tabs(["📊 Statistik", "🔮 Prediksi ML", "🤖 Rekomendasi AI"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Lari", f"{len(df)} kali")
                col1.metric("Jarak Total", f"{df['distance_km'].sum():.1f} km")
                col2.metric("Pace Rata-rata", f"{df['pace'].mean():.2f} min/km")
                
                st.subheader("Grafik Performa Pace")
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(data=df, x='date', y='pace', marker='o', ax=ax, label='Pace Aktual')
                ax.invert_yaxis() # Pace kecil = Cepat (di atas)
                ax.set_title("Grafik Pace Lari (Semakin ke atas semakin cepat)")
                st.pyplot(fig)

            with tab2:
                st.subheader("🔮 Prediksi Performa Masa Depan")
                
                # Jalankan fungsi ML
                pred_pace, trend, model, X, y = prediksi_pace_ml(df)
                
                # Tampilkan Hasil Prediksi
                col_pred1, col_pred2 = st.columns(2)
                
                # Warna teks dinamis (Hijau kalau makin cepat, Merah kalau melambat)
                delta_color = "normal"
                if trend < 0: # Negatif artinya pace turun (makin cepat)
                    pesan_tren = "Tren MEMBAIK (Makin Cepat) 🚀"
                    delta_color = "inverse" # Streamlit metric inverse: turun = hijau
                else:
                    pesan_tren = "Tren MEMBURUK (Melambat) 🐢"
                    delta_color = "off"
                
                col_pred1.metric("Prediksi Pace Lari Selanjutnya", f"{pred_pace:.2f} min/km", delta=f"Tren: {trend:.4f}", delta_color=delta_color)
                col_pred2.info(f"Analisis Tren: **{pesan_tren}**")
                
                # Visualisasi Regresi Linear
                st.write("Visualisasi Garis Regresi (Machine Learning):")
                fig_ml, ax_ml = plt.subplots(figsize=(10, 4))
                
                # Plot titik data asli
                ax_ml.scatter(X, y, color='blue', label='Data Aktual')
                
                # Plot garis prediksi
                ax_ml.plot(X, model.predict(X), color='red', linestyle='--', label='Garis Tren ML')
                
                # Plot titik prediksi masa depan
                ax_ml.scatter(len(df), pred_pace, color='green', s=100, label='Prediksi Next Run', zorder=5)
                
                ax_ml.set_xlabel("Urutan Lari")
                ax_ml.set_ylabel("Pace (min/km)")
                ax_ml.invert_yaxis()
                ax_ml.legend()
                st.pyplot(fig_ml)
                
                st.caption("*Garis merah putus-putus menunjukkan arah perkembangan lari Anda. Jika garisnya miring ke atas (nilai pace di sumbu Y mengecil), berarti Anda semakin cepat.*")

            with tab3:
                st.subheader("🤖 Coach Gemini Analysis")
                if not gemini_key:
                    st.warning("Masukkan API Key Gemini di sidebar untuk melihat rekomendasi pelatih AI.")
                else:
                    with st.spinner("Gemini sedang berpikir..."):
                        # Siapkan ringkasan data
                        summary_json = df[['distance_km', 'pace', 'total_elevation_gain', 'average_heartrate']].describe().to_json()
                        
                        # Minta rekomendasi
                        hasil_ai = tanya_gemini(summary_json, f"{pred_pace:.2f}", gemini_key)
                        
                        st.markdown(hasil_ai)