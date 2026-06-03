# AI Running Coach - Strava Training

Aplikasi Streamlit untuk menganalisis data lari dari Strava. Project ini bisa membaca data dari file CSV export Strava atau dari Strava API, lalu menampilkan statistik lari, grafik perkembangan pace, prediksi pace berikutnya dengan Machine Learning, dan rekomendasi latihan menggunakan Gemini AI.

## Fitur Utama

- Upload file CSV data lari Strava.
- Login Strava API untuk mengambil data aktivitas terbaru.
- Cleaning data otomatis untuk format CSV Strava.
- Filter aktivitas lari saja.
- Hitung jarak dalam kilometer, durasi dalam menit, dan pace dalam menit/km.
- Tampilkan statistik total lari, total jarak, dan pace rata-rata.
- Visualisasi perkembangan pace.
- Prediksi pace lari berikutnya menggunakan Linear Regression.
- Rekomendasi latihan dari Gemini AI.
- Download hasil data yang sudah dibersihkan sebagai CSV siap training.

## Struktur Project

```text
stravaTraining/
|-- app.py                    # Aplikasi utama Streamlit
|-- requirements.txt          # Daftar dependency Python
|-- modules/                  # Script eksperimen/pendukung
|   |-- ambil_data_strava.py  # Ambil data dari Strava API lewat terminal
|   |-- tukar_token.py        # Tukar authorization code Strava jadi token
|   |-- analisis_pace.py      # Analisis pace + rekomendasi Gemini versi script
|   |-- analisis_mingguan.py  # Analisis mingguan dan output grafik
|   `-- prediksi_pace.py      # Prediksi pace mingguan versi script
|-- data/                     # Data lokal, token, dan CSV pribadi
|-- outputs/                  # Hasil grafik dan ringkasan analisis
|-- .streamlit/               # Konfigurasi Streamlit dan secrets lokal
`-- .devcontainer/            # Konfigurasi Codespaces/devcontainer
```

Catatan: logic utama aplikasi sekarang ada di `app.py`. File di folder `modules/` lebih banyak berisi script terpisah/eksperimen lama.

## Cara Menjalankan Project

1. Buat virtual environment.

```bash
python -m venv .venv
```

2. Aktifkan virtual environment.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

3. Install dependency.

```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi.

```bash
streamlit run app.py
```

5. Buka aplikasi di browser.

```text
http://localhost:8501
```

## Cara Pakai Aplikasi

### Opsi 1: Upload CSV

1. Export data aktivitas dari Strava.
2. Upload file CSV lewat sidebar aplikasi.
3. Aplikasi akan membersihkan data dan mengambil aktivitas lari saja.
4. Buka tab:
   - Statistik
   - Prediksi ML
   - Rekomendasi AI

### Opsi 2: Login Strava API

Login API hanya bekerja jika `STRAVA_CLIENT_ID` dan `STRAVA_CLIENT_SECRET` sudah tersedia di Streamlit secrets.

Setelah login berhasil, aplikasi menyimpan access token sementara di `st.session_state`.

## Format CSV yang Didukung

Aplikasi berusaha membaca format export Strava dan beberapa nama kolom alternatif. Kolom yang paling penting:

```text
name
start_date_local
type
distance
moving_time
total_elevation_gain
average_speed
max_speed
average_heartrate
```

Format export Strava yang juga didukung:

```text
Activity Date
Activity Name
Activity Type
Elapsed Time
Moving Time
Distance
Elevation Gain
Average Speed
Max Speed
Average Heart Rate
```

Aplikasi akan membuat kolom turunan:

```text
distance_km
duration_min
pace
date
date_obj
```

## Konfigurasi Secrets

Buat file `.streamlit/secrets.toml` jika ingin memakai Strava API dan Gemini AI.

Contoh:

```toml
STRAVA_CLIENT_ID = "isi_client_id_strava"
STRAVA_CLIENT_SECRET = "isi_client_secret_strava"
GEMINI_API_KEY = "isi_api_key_gemini"
```

File `.streamlit/secrets.toml` sudah masuk `.gitignore`, jadi jangan commit file ini.

## File yang Tidak Perlu Di-commit

Project ini menyimpan beberapa file pribadi dan hasil generate. File berikut sudah diabaikan oleh `.gitignore`:

```text
.streamlit/secrets.toml
data/strava_keys.json
data/*.csv
outputs/
__pycache__/
*.pyc
```

Jangan commit token Strava, API key Gemini, atau data CSV pribadi.

## Cara Kerja Singkat

1. Data masuk dari upload CSV atau Strava API.
2. `process_dataframe()` membersihkan kolom, menghapus duplikat, memfilter lari, dan menghitung pace.
3. Tab Statistik menampilkan metrik dan grafik pace.
4. `prediksi_pace_ml()` melatih Linear Regression sederhana berdasarkan urutan sesi lari.
5. Tab Prediksi ML menampilkan prediksi pace lari berikutnya dan trendline.
6. `tanya_gemini()` mengirim ringkasan data ke Gemini untuk membuat saran latihan.

## Catatan Pengembangan

- Model prediksi saat ini masih sederhana: hanya memakai urutan lari sebagai fitur.
- Untuk hasil prediksi yang lebih kuat, fitur bisa ditambah seperti jarak, elevasi, heart rate, volume mingguan, dan tipe latihan.
- Beberapa script di `modules/` bisa dirapikan atau digabung ke `app.py` jika ingin struktur project lebih bersih.
- Jika emoji terlihat rusak seperti `ðŸ...`, kemungkinan file perlu disimpan ulang dengan encoding UTF-8.
