# Aplikasi Deteksi Penyakit Daun Selada Hidroponik dengan YOLOv8s

Aplikasi web berbasis **Flask** dan **YOLOv8s** untuk mendeteksi kondisi daun selada hidroponik. Model melakukan deteksi objek dan menampilkan bounding box, label kelas, serta nilai confidence untuk tiga kelas berikut:

- **Sehat** (`Healthy`)
- **Klorosis** (`Chlorosis`)
- **Busuk** (`Rot`)

Aplikasi mendukung deteksi melalui unggah gambar, unggah video, dan webcam secara real-time.

## Struktur Folder

```text
YOLO-LETTUCE-APP/
├── models/
│   ├── best.pt
│   └── PLACEHOLDER.txt
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── uploads/
│   └── .gitkeep
├── .gitignore
├── app.py
├── Procfile
├── README.md
├── requirements.txt
└── runtime.txt
```

## Persyaratan

- Python 3.10.12
- `pip`
- Model YOLO hasil pelatihan bernama `best.pt`
- FFmpeg untuk pemrosesan video, terutama apabila fitur unggah video digunakan

## Menjalankan Aplikasi Secara Lokal

### 1. Buka terminal pada folder proyek

```bash
cd YOLO-LETTUCE-APP
```

### 2. Buat virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Perbarui `pip`

```bash
python -m pip install --upgrade pip
```

### 4. Instal dependensi

```bash
pip install -r requirements.txt
```

### 5. Letakkan model YOLO

Pastikan file model berada pada lokasi berikut:

```text
models/best.pt
```

### 6. Buat file `.env` bila diperlukan

Contoh isi `.env` untuk pengembangan lokal:

```env
PORT=5000
FLASK_DEBUG=true
```

File `.env` tidak akan masuk ke Git karena sudah dikecualikan melalui `.gitignore`.

### 7. Jalankan aplikasi

```bash
python app.py
```

Buka alamat berikut pada browser:

```text
http://127.0.0.1:5000
```

Status aplikasi dan model dapat diperiksa melalui:

```text
http://127.0.0.1:5000/health
```

## Menjalankan dengan Gunicorn Secara Lokal

Gunicorn digunakan sebagai server produksi pada Render atau Railway.

```bash
gunicorn app:app
```

Pada Windows, Gunicorn umumnya tidak digunakan secara langsung. Jalankan `python app.py` untuk pengujian lokal, atau gunakan WSL/Docker untuk menguji Gunicorn.

## Deploy ke Render

### 1. Siapkan repositori GitHub

Inisialisasi Git dan unggah proyek ke GitHub:

```bash
git init
git add .
git commit -m "Prepare Flask YOLO app for deployment"
git branch -M main
git remote add origin https://github.com/USERNAME/NAMA-REPOSITORY.git
git push -u origin main
```

Folder `uploads/` tetap tersedia melalui `uploads/.gitkeep`, tetapi file hasil unggahan tidak dimasukkan ke repositori.

### 2. Buat Web Service di Render

1. Masuk ke dashboard Render.
2. Pilih **New** lalu **Web Service**.
3. Hubungkan akun GitHub dan pilih repositori proyek.
4. Pilih runtime **Python 3**.
5. Isi konfigurasi berikut:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

6. Pilih branch `main`.
7. Pilih instance sesuai kebutuhan memori model YOLO.

### 3. Tentukan versi Python di Render

Tambahkan environment variable pada halaman **Environment**:

```text
Key: PYTHON_VERSION
Value: 3.10.12
```

File `runtime.txt` tetap disertakan untuk kompatibilitas platform yang membacanya, tetapi pada Render versi Python sebaiknya ditentukan melalui `PYTHON_VERSION`.

### 4. Tambahkan environment variable aplikasi

Render memberikan variabel `PORT` secara otomatis. Aplikasi membaca nilainya dengan:

```python
port = int(os.environ.get("PORT", 5000))
```

Untuk produksi, jangan mengaktifkan debug. Jika perlu, tambahkan:

```text
Key: FLASK_DEBUG
Value: false
```

### 5. Jalankan deployment

Pilih **Create Web Service**. Render akan menjalankan build, memasang dependensi, kemudian menjalankan aplikasi menggunakan Gunicorn.

Setelah deployment selesai, buka URL `onrender.com` yang diberikan. Periksa endpoint `/health` untuk memastikan nilai `model_ready` adalah `true`.

## Penempatan File `best.pt`

### Pilihan A — Model di bawah 100 MB

Model `best.pt` pada paket proyek ini berukuran sekitar 22,5 MB, sehingga dapat dimasukkan ke repositori Git melalui command line. Pastikan struktur repositori tetap seperti berikut sebelum melakukan push:

```text
models/best.pt
```

Catatan: unggahan melalui antarmuka web GitHub dibatasi lebih kecil daripada unggahan melalui command line. Untuk model ini, gunakan Git command line atau GitHub Desktop.

### Pilihan B — Model di atas 100 MB

`.gitignore` tidak dapat mengecualikan file berdasarkan ukuran. Jika `best.pt` melebihi 100 MB, gunakan salah satu cara berikut:

1. **Git LFS**

   ```bash
   git lfs install
   git lfs track "*.pt"
   git add .gitattributes models/best.pt
   git commit -m "Track YOLO model with Git LFS"
   git push
   ```

2. **Penyimpanan eksternal**

   Simpan model pada object storage atau lokasi unduhan privat, kemudian unduh model ke `models/best.pt` pada tahap build atau saat aplikasi dimulai. Jangan menyimpan token atau URL rahasia secara langsung di kode; gunakan environment variable Render.

3. **Persistent Disk Render**

   Pada layanan Render berbayar, persistent disk dapat dipasang pada direktori model, misalnya:

   ```text
   /opt/render/project/src/models
   ```

   Setelah disk terpasang, unggah `best.pt` ke direktori tersebut melalui Shell/SCP, kemudian restart service. Metode ini tidak tersedia untuk Free Web Service.

Aplikasi tidak langsung berhenti apabila model belum tersedia. Endpoint `/health` akan menampilkan `model_ready: false` dan pesan kesalahan pada `model_error`. Setelah model ditempatkan pada `models/best.pt`, restart atau redeploy service agar model dimuat kembali.

## Catatan Penyimpanan Folder `uploads`

Render menggunakan filesystem sementara secara default. File gambar atau video yang disimpan ke folder `uploads/` dapat hilang ketika service restart, spin down, atau redeploy. Hal ini tidak menjadi masalah apabila folder tersebut hanya digunakan untuk file sementara. Gunakan persistent disk atau object storage apabila hasil deteksi harus disimpan secara permanen.

## Troubleshooting

### Model tidak ditemukan

Pastikan nama dan lokasi file tepat:

```text
models/best.pt
```

Kemudian periksa:

```text
/health
```

### Import OpenCV gagal di server

Project menggunakan `opencv-python-headless`, bukan `opencv-python`, agar OpenCV dapat berjalan pada server tanpa antarmuka grafis.

### Proses video gagal

Pastikan FFmpeg tersedia pada environment deployment. Fitur gambar dan webcam tidak memerlukan proses konversi video yang sama seperti unggah video.

### Build kehabisan memori

Ultralytics akan memasang PyTorch dan dependensi terkait. Pilih instance Render yang menyediakan memori memadai apabila build atau proses inferensi gagal karena keterbatasan RAM.
