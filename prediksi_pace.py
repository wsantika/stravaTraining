import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- 1. LOAD & PREPARE DATA (DARI RAW KE WEEKLY) ---

# Load data mentah
df = pd.read_csv('data_lari_saya.csv')

# Konversi ke format tanggal yang benar
df['date'] = pd.to_datetime(df['date'])

# Buat kolom Tahun dan Minggu (Inilah yang tadi hilang/error)
df['year'] = df['date'].dt.isocalendar().year
df['week'] = df['date'].dt.isocalendar().week

# Kita butuh logika klasifikasi lagi untuk menghitung volume latihan
# (Apakah ini lari santai atau cepat?)
pace_Q1 = df['pace_decimal'].quantile(0.25) 
pace_Q3 = df['pace_decimal'].quantile(0.75)

def classify_simple(pace):
    if pace <= pace_Q1: return 'SPEED'
    elif pace > pace_Q1 and pace < pace_Q3: return 'TEMPO'
    else: return 'EASY'

df['workout_type'] = df['pace_decimal'].apply(classify_simple)

# SEKARANG BARU KITA GROUPING MENJADI DATA MINGGUAN
weekly_data = df.groupby(['year', 'week']).agg(
    avg_pace=('pace_decimal', 'mean'),
    # Hitung total sesi lari (Volume)
    total_runs=('workout_type', 'count') 
).reset_index()

# Urutkan data berdasarkan waktu
weekly_data = weekly_data.sort_values(by=['year', 'week'])

print("Data berhasil diubah menjadi format mingguan!")
print(weekly_data.head())

# --- 2. FEATURE ENGINEERING (MEMBUAT DATA UNTUK PREDIKSI) ---

# X (Input): Performa Minggu Ini
weekly_data['prev_week_pace'] = weekly_data['avg_pace'] 
weekly_data['prev_week_volume'] = weekly_data['total_runs']

# y (Target): Pace Minggu Depan
# Geser data avg_pace ke atas (-1) untuk jadi target
weekly_data['target_next_week_pace'] = weekly_data['avg_pace'].shift(-1)

# Hapus baris terakhir (karena tidak punya target minggu depan)
df_model = weekly_data.dropna().copy()

# --- 3. SPLIT DATA & TRAINING ---

X = df_model[['prev_week_pace', 'prev_week_volume']]
y = df_model['target_next_week_pace']

# Split data (tanpa shuffle karena time-series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Training
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Time-Series Siap!")
print(f"Koefisien (Pengaruh Pace Lalu): {model.coef_[0]:.2f}")
print(f"Koefisien (Pengaruh Volume): {model.coef_[1]:.2f}")

# --- 4. PREDIKSI MINGGU DEPAN ---

# Ambil data minggu terakhir (Real time) dari data asli
last_stats = weekly_data.iloc[-1]
current_pace = last_stats['avg_pace']
current_vol = last_stats['total_runs']

# Prediksi
prediksi_pace = model.predict([[current_pace, current_vol]])

print("-" * 30)
print(f"Performa Minggu Ini: Pace {current_pace:.2f} min/km, Latihan {int(current_vol)} kali")
print(f"PREDIKSI AI: Pace kamu minggu depan kemungkinan adalah {prediksi_pace[0]:.2f} min/km")

# Logika sederhana untuk pesan tambahan
if prediksi_pace[0] < current_pace:
    print("Trend Positif! Pace diprediksi makin cepat. 🚀")
else:
    print("Trend Melambat. Mungkin butuh istirahat atau variasi latihan. ⚠️")