import pandas as pd
import numpy as np
import json
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI KUNCI API ---
# ⚠️ GANTI NILAI DI BAWAH INI DENGAN KUNCI GEMINI API ANDA
GEMINI_API_KEY = "#"


# --- FUNGSI LLM REKOMENDASI ---
def get_llm_recommendation(json_data, api_key):
    """Fungsi untuk mengirim JSON ke Gemini dan mendapatkan rekomendasi teks."""
    
    # 1. Inisialisasi Klien dengan Kunci yang Disediakan
    try:
        # Kita masukkan kunci API langsung ke klien
        client = genai.Client(api_key=api_key)
    except Exception as e:
        print("ERROR: Kunci API Gemini tidak valid atau terjadi masalah koneksi.")
        print(f"Detail: {e}")
        return
    
    # 2. Buat Prompt Template 
    system_prompt = (
        "Anda adalah Pelatih Lari Data-Driven, spesialis pelari pemula. "
        "Analisis data JSON di bawah ini. Jelaskan hasil korelasi secara singkat. "
        "Berdasarkan korelasi, berikan 3-4 Poin Rencana Latihan MINGGUAN yang SPESIFIK dan memotivasi untuk meningkatkan pace."
    )
    
    final_prompt = f"{system_prompt}\n\n[DATA ANALISIS JSON]:\n{json_data}"
    
    print("\n--- Mengirim data ke Gemini... (Menunggu Rekomendasi) ---")
    
    try:
        # Panggil API menggunakan model gemini-2.5-flash
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt
        )
        
        print("\n\n--- REKOMENDASI FINAL DARI LLM (COACH) ---")
        print(response.text)
        
    except Exception as e:
        print(f"\nERROR PANGGILAN API GEMINI: Gagal saat generate content.")
        print(f"Detail: {e}")

# --- KODE UTAMA (LOGIKA PYTHON) ---

# Pastikan Anda sudah menginstal pandas, numpy, dan google-genai
# import pandas as pd # Sudah diimpor
# import numpy as np # Sudah diimpor
# import json # Sudah diimpor


# 1. Load Data CSV
df = pd.read_csv('data_lari_saya.csv')

# --- TAHAP 1: KLASIFIKASI DATA (FEATURE ENGINEERING) ---

# Hapus baris dengan pace tidak valid dan pastikan kolom 'date' diolah
df.dropna(subset=['pace_decimal'], inplace=True) 
df['date'] = pd.to_datetime(df['date'], errors='coerce') 
df.dropna(subset=['date'], inplace=True) # Hapus baris jika tanggal tidak valid

# Hitung Quartile Pace (Pace Q1 dan Q3)
pace_Q1 = df['pace_decimal'].quantile(0.25)
pace_Q3 = df['pace_decimal'].quantile(0.75) 

def classify_smart(pace):
    if pace <= pace_Q1:
        return 'SPEED/INTERVAL'
    elif pace > pace_Q1 and pace < pace_Q3:
        return 'TEMPO/MODERATE'
    else:
        return 'EASY/RECOVERY'

df['workout_type_smart'] = df['pace_decimal'].apply(classify_smart)

print("--- Hasil Klasifikasi Data ---")
print(f"Pace Q1 (Batas Cepat): {pace_Q1:.2f} min/km")
print(f"Pace Q3 (Batas Santai): {pace_Q3:.2f} min/km")
print("\nDistribusi Latihan:")
print(df['workout_type_smart'].value_counts())
print("-" * 30)

# --- TAHAP 2: KORELASI MINGGUAN (AI LOGIC) ---

df['week'] = df['date'].dt.isocalendar().week 
df['year'] = df['date'].dt.isocalendar().year 

weekly_data = df.groupby(['year', 'week']).agg(
    avg_pace=('pace_decimal', 'mean'),
    speed_runs=('workout_type_smart', lambda x: (x == 'SPEED/INTERVAL').sum()),
    easy_runs=('workout_type_smart', lambda x: (x == 'EASY/RECOVERY').sum())
).reset_index()

weekly_data['pace_delta'] = weekly_data['avg_pace'].shift(-1) - weekly_data['avg_pace']

# Hitung Koefisien Korelasi
correlation_speed_vs_next_pace = weekly_data['speed_runs'].shift(-1).corr(weekly_data['pace_delta'])
correlation_easy_vs_next_pace = weekly_data['easy_runs'].shift(-1).corr(weekly_data['pace_delta'])

# --- TAHAP 3: PEMBUATAN JSON INPUT (LLM BRIDGE) ---

avg_pace_change = weekly_data['pace_delta'].mean() 
total_runs_in_data = len(df)
most_frequent_workout = df['workout_type_smart'].mode()[0]

analysis_data = {
    "user_profile": "Pelari Pemula",
    "goal": "Peningkatan Pace Berbasis Data",
    "personalized_pace_zones": {
        "Q1_speed_limit": f"{pace_Q1:.2f} min/km",
        "Q3_easy_limit": f"{pace_Q3:.2f} min/km"
    },
    "training_summary": {
        "total_runs_last_month": total_runs_in_data,
        "most_frequent_workout": most_frequent_workout,
        "avg_pace_change_vs_prev_period": f"{avg_pace_change:.3f}"
    },
    "core_ai_analysis": {
        "correlation_speed_runs_vs_pace_improvement": f"{correlation_speed_vs_next_pace:.3f}",
        "correlation_easy_runs_vs_pace_improvement": f"{correlation_easy_vs_next_pace:.3f}"
    }
}

json_string = json.dumps(analysis_data, indent=4)
print("\n--- JSON INPUT UNTUK LLM GENERATIF ---")
print(json_string)

# --- TAHAP 4: PANGGILAN LLM ---
# Panggil fungsi rekomendasi menggunakan kunci API yang Anda hardcode
get_llm_recommendation(json_string, GEMINI_API_KEY)


# --- VISUALISASI 1: TREN PACE RATA-RATA MINGGUAN ---

# 1. Siapkan Sumbu X (Timeline)
# Buat indeks numerik untuk grafik yang berkelanjutan
weekly_data['time_index'] = weekly_data.index 

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=weekly_data, 
    x='time_index', 
    y='avg_pace', 
    marker='o', 
    color='darkblue'
)

# Label & Judul
plt.title('Tren Pace Rata-rata Mingguan (Peningkatan Pace Terlihat dari Garis yang Menurun)', fontsize=14)
plt.xlabel('Minggu Latihan (Indeks Waktu)', fontsize=12)
plt.ylabel('Pace Rata-rata (menit/km)', fontsize=12)
plt.gca().invert_yaxis() # Invers Y-axis agar Pace yang LEBIH CEPAT (angka kecil) terlihat di ATAS
plt.grid(True, linestyle='--', alpha=0.6)

# Simpan Grafik
plt.savefig('grafik_tren_pace.png')
print("\n[INFO] Grafik Tren Pace disimpan di: grafik_tren_pace.png")


# --- VISUALISASI 2: KORELASI LATIHAN KERAS VS. PENINGKATAN PACE (VALIDASI AI) ---

plt.figure(figsize=(8, 6))

# Menggunakan regplot untuk scatter plot dengan garis regresi
sns.regplot(
    data=weekly_data,
    x='speed_runs',
    y='pace_delta',
    scatter_kws={'alpha':0.6, 's': 70}, # Atur transparansi dan ukuran titik
    line_kws={'color': 'red', 'linewidth': 2}
)

# Label & Judul
plt.title(f'Validasi AI: Korelasi Speed Runs vs. Perubahan Pace (r={correlation_speed_vs_next_pace:.3f})', fontsize=14)
plt.xlabel('Jumlah Latihan Speed/Interval Minggu Sebelumnya', fontsize=12)
plt.ylabel('Pace Delta (Perubahan Pace Minggu Ini vs Minggu Lalu)', fontsize=12)

# Penjelasan hasil di grafik
# Note: Korelasi Negatif berarti Garis Turun (GOOD), Korelasi Positif berarti Garis Naik (BAD)
plt.axhline(0, color='gray', linestyle='--') # Garis di y=0 (Tidak ada perubahan pace)
plt.text(weekly_data['speed_runs'].max() * 0.5, 0.01, 'Zona Pace Melambat', color='black', fontsize=10)
plt.text(weekly_data['speed_runs'].max() * 0.5, -0.01, 'Zona Pace Meningkat (Lebih Cepat)', color='darkgreen', fontsize=10)

plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('grafik_korelasi.png')
print("[INFO] Grafik Korelasi disimpan di: grafik_korelasi.png")