import requests
import json

# --- 1. GANTI DENGAN DATA APLIKASIMU ---
CLIENT_ID = "#"        # Angka (ID Aplikasi)
CLIENT_SECRET = "#" # Kode Rahasia Panjang
AUTHORIZATION_CODE = "#" # Kode 39d06b... tadi

# --- 2. ENDPOINT UNTUK PENUKARAN TOKEN ---
token_url = "https://www.strava.com/oauth/token"

payload = {
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
    'code': AUTHORIZATION_CODE,
    'grant_type': 'authorization_code'
}

# --- 3. LAKUKAN POST REQUEST ---
print("Sedang menukar kode dengan token...")
response = requests.post(token_url, data=payload)
data = response.json()

if response.status_code == 200:
    print("Penukaran berhasil! Simpan data berikut:")
    
    # Simpan Token & Refresh Token ke dalam file
    with open('strava_keys.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\nAccess Token Baru: {data['access_token']}")
    print(f"Refresh Token: {data['refresh_token']}")
    print("\nData Token sudah disimpan di strava_keys.json")

    # Access Token ini berlaku 6 jam. Refresh Token berlaku selamanya.
    print("\nSekarang, ganti token di file ambil_data_strava.py dengan Access Token BARU ini!")
else:
    print(f"Gagal menukar token. Error: {response.status_code}")
    print(data)