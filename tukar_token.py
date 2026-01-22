import requests
import json

# --- INPUT USER ---
print("="*40)
print("  STRAVA TOKEN EXCHANGER  ")
print("="*40)

CLIENT_ID = input("Masukkan Client ID Anda: ").strip() # Masukan Client ID dari Strava
CLIENT_SECRET = input("Masukkan Client Secret Anda: ").strip() # Secret Key (Hati-hati, idealnya jangan di-share)

# Minta user paste kode dari browser
auth_code = input("Masukkan CODE dari URL Browser (setelah code=): ").strip()

# --- REQUEST TOKEN ---
token_url = "https://www.strava.com/oauth/token"
payload = {
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
    'code': auth_code,
    'grant_type': 'authorization_code'
}

print("Sedang menukar kode dengan token...")
response = requests.post(token_url, data=payload)
data = response.json()

if response.status_code == 200:
    print("\nBERHASIL! Token baru didapatkan.")
    print(f"Access Token: {data['access_token']}")
    print(f"Refresh Token: {data['refresh_token']}")
    
    # Simpan ke JSON
    with open('strava_keys.json', 'w') as f:
        json.dump(data, f, indent=4)
    print("Disimpan ke strava_keys.json")
else:
    print(f"Gagal. Error: {response.status_code}")
    print(data)