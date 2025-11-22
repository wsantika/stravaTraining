import requests
import pandas as pd
import os

# --- KONFIGURASI ---
# Masukkan Access Token yang kamu copy dari website Strava tadi di sini:
access_token = "#"

# URL API Strava untuk mengambil aktivitas atlet
url = "https://www.strava.com/api/v3/athlete/activities"

# --- 1. REQUEST DATA DARI STRAVA ---
print("Sedang mengambil data dari Strava...")

# Kita minta 50 aktivitas terakhir
payload = {'per_page': 50, 'page': 1} 
headers = {'Authorization': f'Bearer {access_token}'}

response = requests.get(url, headers=headers, params=payload)

if response.status_code == 200:
    data = response.json()
    print(f"Berhasil! Mendapatkan {len(data)} data aktivitas.")
else:
    print(f"Gagal mengambil data. Error: {response.status_code}")
    print(response.text)
    exit()

# --- 2. FILTER & BERSIHKAN DATA (DATA CLEANING) ---
if not data:
    print("Data kosong/tidak ada aktivitas.")
    exit()

# Ubah JSON ke DataFrame Pandas
df = pd.DataFrame(data)

# Kita hanya butuh jenis olahraga "Run"
df = df[df['type'] == 'Run'].copy()

# Pilih kolom yang penting saja untuk AI
cols_to_keep = [
    'name', 'start_date_local', 'distance', 'moving_time', 
    'total_elevation_gain', 'average_speed', 'max_speed', 'average_heartrate'
]

# Handle jika kolom heartrate tidak ada (misal user gak pakai smartwatch)
for col in cols_to_keep:
    if col not in df.columns:
        df[col] = 0 # Isi 0 jika data tidak ada

df_clean = df[cols_to_keep].copy()

# --- 3. FEATURE ENGINEERING (KONVERSI SATUAN) ---
# Strava kasih data dalam meter & detik. Kita butuh KM & Menit.

# Jarak: Meter -> Kilometer
df_clean['distance_km'] = df_clean['distance'] / 1000

# Durasi: Detik -> Menit
df_clean['duration_min'] = df_clean['moving_time'] / 60

# Pace: Meter/Detik -> Menit/KM
# Rumus Pace = (1 / speed_in_mps) * (1000 / 60)  <-- Rumus konversi fisika
# Cara gampang: Pace = Waktu (menit) / Jarak (km)
df_clean['pace_decimal'] = df_clean['duration_min'] / df_clean['distance_km']

# Format Tanggal agar rapi (Hanya ambil tanggal, jam dibuang)
df_clean['date'] = pd.to_datetime(df_clean['start_date_local']).dt.date

# --- 4. SIMPAN HASILNYA ---
nama_file = "data_lari_saya.csv"
df_clean.to_csv(nama_file, index=False)

print("-" * 30)
print("CONTOH 5 DATA TERATAS:")
print(df_clean[['date', 'distance_km', 'pace_decimal', 'average_heartrate']].head())
print("-" * 30)
print(f"Data lengkap sudah disimpan di file: {nama_file}")