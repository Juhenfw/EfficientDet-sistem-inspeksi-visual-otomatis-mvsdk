# Software Inspeksi Visual Otomatis (QC Pianika)

Direktori ini berisi rangkaian perangkat lunak operasional untuk **Sistem Inspeksi Visual Otomatis** pada lini produksi Pianika. Perangkat lunak ini mengintegrasikan pengambilan gambar dari kamera industri, inferensi AI real-time menggunakan EfficientDet, dan manajemen data produksi.

Sistem ini terbagi menjadi empat modul utama yang berfungsi untuk kalibrasi perangkat keras, konfigurasi logika spasial, dan operasional inspeksi di dua stasiun kerja yang berbeda.

---

## Modul Perangkat Lunak

### 1. Tool Kalibrasi Kamera ([Calibrate_Camera.py](./Calibrate_Camera.py))
Instrumen ini digunakan untuk mengatur parameter optik kamera industri MindVision guna mendapatkan kualitas gambar yang optimal bagi model AI.
* **Fitur Utama**: Pengaturan waktu eksposur (*exposure time*), gamma, kontras, gain analog, dan White Balance (Manual RGB, Auto, atau Preset D65).
* **Output**: Menyimpan konfigurasi ke dalam folder `configs/` sebagai referensi bagi sistem utama.

### 2. Tool Konfigurasi Zona ([Calibrate_Zone.py](./Calibrate_Zone.py))
Alat untuk menetapkan "Zona Ideal" atau *Region of Interest* (ROI) bagi setiap komponen yang akan dideteksi agar sistem dapat memverifikasi posisi objek secara akurat.
* **Fitur Utama**: Inferensi AI otomatis untuk mendeteksi posisi objek referensi dan menyimpannya sebagai batasan koordinat spasial.
* **Fungsi**: Memastikan komponen tidak hanya ada, tetapi juga berada di posisi yang benar sesuai standar QC.

### 3. Sistem Inspeksi Stasiun 1 ([main_system_station1.py](./main_system_station1.py))
Perangkat lunak operasional untuk **Stasiun Aksesoris**.
* **Target Deteksi**: Label, *hose*, *mouthpiece*, *leaflet*, dan buku manual.
* **Fitur**: Manajemen operator melalui pemindaian No. Tag, pencatatan log produksi harian, dan sistem antrean cerdas (IPC).

### 4. Sistem Inspeksi Stasiun 2 ([main_system_station2.py](./main_system_station2.py))
Perangkat lunak operasional untuk **Stasiun Unit Utama**.
* **Target Deteksi**: Unit Pianika (Biru/Pink) dan Tas/Case (Biru/Pink).
* **Fitur**: Sinkronisasi data antrean dari Stasiun 1 dan verifikasi kelengkapan unit utama sebelum pengemasan akhir.

---

## Arsitektur Pendukung

* **[GUI_v5.py](./GUI_v5.py)**: Pustaka inti antarmuka pengguna berbasis *CustomTkinter* yang menyediakan layout responsif dan visualisasi status produksi (PASS/FAIL).
* **[mvsdk.py](./mvsdk.py)**: Driver Python SDK untuk integrasi langsung dengan kamera industri MindVision.

---

## Struktur Direktori Operasional

Sistem mengharapkan struktur folder berikut agar dapat berjalan dengan stabil:

```text
.
├── configs/                 # File hasil kalibrasi (.json)
├── models/                  # File bobot AI (best_loss_d1.pth)
├── assets/                  # Gambar panduan operator (.jpg)
├── hasil_inspeksi/          # Database rekam gambar produksi
├── projects/                # Definisi kelas objek (pianika.yml)
├── Training-EfficientDet/   # Akses ke modul pelatihan AI
└── Software-Inspeksi/       # Direktori operasional ini
```

---

## Panduan Penggunaan

### Langkah 1: Kalibrasi Hardware
Pastikan pencahayaan dan fokus kamera sudah optimal. Jalankan:
```bash
python Calibrate_Camera.py
```
Simpan konfigurasi untuk masing-masing index kamera yang terpasang.

### Langkah 2: Konfigurasi Logika Spasial
Tetapkan zona deteksi untuk setiap model pianika (P32E/P32EP) di setiap stasiun:
```bash
python Calibrate_Zone.py
```

### Langkah 3: Operasional Produksi
Jalankan sistem utama sesuai dengan penempatan stasiun kerja:
* **Stasiun 1**: `python main_system_station1.py`
* **Stasiun 2**: `python main_system_station2.py`

---

## Integrasi Data
Setiap hasil inspeksi akan dicatat secara otomatis ke dalam:
1.  **Statistik Harian**: `statistik_station_X.json`.
2.  **Log Produksi**: `log_produksi_station_X.csv` yang mencatat waktu, status, komponen yang kurang, latensi sistem, hingga identitas operator.

---
*Proyek ini merupakan bagian dari penelitian tugas akhir oleh Juhen FW - FTMM Universitas Airlangga.*
