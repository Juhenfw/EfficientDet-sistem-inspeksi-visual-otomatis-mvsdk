# Sistem Inspeksi Visual Otomatis Berbasis EfficientDet

Repositori ini merupakan hasil implementasi dari penelitian Tugas Akhir (Skripsi) yang berjudul **"Implementasi EfficientDet untuk Deteksi Komponen dan Klasifikasi Kelengkapan Alat Musik Pianika pada Sistem Inspeksi di PT. XYZ"**.

Sistem ini dirancang untuk mengotomatisasi pengendalian mutu pada tahap pengepakan akhir produk pianika guna meminimalisir risiko kesalahan identifikasi akibat kelelahan manusia (*human error*).

---

## Ringkasan Proyek
Penelitian ini mengembangkan sistem inspeksi visual otomatis berbasis kamera ganda yang menggunakan arsitektur *Deep Learning* EfficientDet untuk mendeteksi sembilan kategori komponen dan mengklasifikasikan kelengkapan paket secara *real-time*.

### Kategori Objek Deteksi:
* **Aksesori**: *Hose* (selang), *Mouthpiece* (corong tiup), Label, Buku Manual, dan *Leaflet*.
* **Unit Utama**: Pianika (Biru/Pink) dan *Case* (Biru/Pink).

---

## Fitur Utama Sistem
* **Integrasi Dual-Station**: Sistem dirancang selaras dengan SOP industri yang mencakup Stasiun 1 (verifikasi aksesori) dan Stasiun 2 (verifikasi unit utama serta tas).
* **Logika Spasial (IoU)**: Selain deteksi objek, sistem menerapkan filter logika spasial berbasis *Intersection over Union* (IoU) dengan ambang batas 0,3 untuk memastikan presisi tata letak komponen sesuai standar perusahaan.
* **Pendekatan Data-Centric**: Menggunakan strategi isolasi objek dan *hard negative samples* (wadah kosong) untuk meningkatkan ketangguhan model terhadap pantulan cahaya dan gangguan visual di pabrik.
* **Performa Kecepatan Tinggi**: Rata-rata waktu tanggap (*latency*) sistem berada di kisaran 0,55 hingga 0,56 detik, memenuhi target produktivitas industri di bawah 2 detik per paket.

---

## Performa dan Hasil Eksperimen
Berdasarkan evaluasi terhadap 2.441 sampel citra operasional di lapangan:
* **Varian Model Terbaik**: EfficientDet-D1.
* **mAP@0.50**: 0,9932.
* **F1-Score**: 0,9927.
* **Akurasi Lapangan**: 99,4% (Stasiun Aksesori) dan 97,8% (Stasiun Unit Utama).

---

## Struktur Repositori
Proyek ini terbagi menjadi dua modul fungsional utama:

1.  **[Training-EfficientDet](./Training-EfficientDet/)**
    Berisi infrastruktur pengembangan model AI, termasuk strategi akuisisi data adaptif, skrip pelatihan yang dioptimalkan dengan *Automatic Mixed Precision* (AMP) dan *Gradient Accumulation*.

2.  **[Software-Inspeksi](./Software-Inspeksi/)**
    Paket perangkat lunak operasional yang mencakup alat kalibrasi kamera industri, konfigurasi zona referensi (ROI), dan antarmuka pengguna (GUI) utama berbasis *CustomTkinter*.

---

## Spesifikasi Sistem (Hardware Reference)
* **Kamera**: 2x Kamera Industri HT-SUA501GC-TIV-C (Sensor 2/3" CMOS, 5MP, 40 FPS).
* **Unit Pemrosesan**: Laptop/Mini PC (Intel Core i7, RAM 16GB, GPU NVIDIA RTX 4060 8GB).

---

## Sitasi (Citation)

Jika Anda menggunakan kode atau hasil penelitian dari repositori ini, harap berikan atribusi sesuai format berikut:

**Bahasa Indonesia:**
Wildan, J. F. (2026). Implementasi EfficientDet untuk Deteksi Komponen dan Klasifikasi Kelengkapan Alat Musik Pianika pada Sistem Inspeksi di PT. XYZ. Skripsi. Surabaya: Universitas Airlangga.

**English:**
Wildan, J. F. (2026). Implementation of EfficientDet for Component Detection and Completeness Classification of Melodica Musical Instruments in the Inspection System at PT. XYZ. Undergraduate Thesis. Surabaya: Universitas Airlangga.

---

## Penulis
**Juhen Fashikha Wildan**\
Program Sarjana Teknik Robotika dan Kecerdasan Buatan\
Fakultas Teknologi Maju dan Multidisiplin\
**Universitas Airlangga**

---
*Penelitian ini didukung oleh PT. XYZ sebagai bagian dari upaya peningkatan efisiensi proses pengepakan dan pengendalian mutu produk.*
