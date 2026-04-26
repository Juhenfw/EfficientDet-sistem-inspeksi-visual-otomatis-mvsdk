# Yet Another EfficientDet Pytorch (Optimized Implementation)

Implementasi ulang EfficientDet menggunakan PyTorch ini dikembangkan untuk mereproduksi algoritma resmi Google AutoML secara akurat. Proyek ini secara khusus telah diadopsi dan dioptimalkan untuk mendukung [Sistem Inspeksi Visual Otomatis](../Software-Inspeksi/), yang bertujuan untuk melakukan deteksi komponen secara real-time pada lini produksi instrumen musik (studi kasus: Pianika).

---

## Performa dan Bobot Terlatih

Hasil pengujian pada dataset COCO menunjukkan tingkat presisi yang kompetitif. Bobot pretrained yang tersedia dapat diunduh melalui tautan berikut:

| Koefisien | Link Download | Img Size | Memori GPU (MB) | FPS | mAP 0.5:0.95 (Resmi) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 512 | 1049 | 36.20 | 33.8 |
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 640 | 1159 | 29.69 | 39.6 |
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 768 | 1321 | 26.50 | 43.0 |
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 896 | 1647 | 22.73 | 45.8 |

---

## Fitur Optimasi Pelatihan

Repositori ini menyertakan skrip pelatihan khusus ([train_d0.py](./train_d0.py)) dan ([train_d1.py](./train_d1.py)) yang memiliki fitur unggulan sebagai berikut:

* **Manajemen Memori**:
    * **Gradient Accumulation**: Mendukung penggunaan batch size efektif yang besar melalui akumulasi gradien sebelum pembaruan bobot.
    * **Automatic Mixed Precision (AMP)**: Mengoptimalkan kecepatan dan penggunaan memori menggunakan operasi presisi campuran melalui `torch.amp`.
* **Generalisasi Model**: Mengintegrasikan **Mixup Augmentation** untuk membantu model melakukan generalisasi yang lebih baik pada variasi data baru.
* **Metrik Deteksi Lengkap**: Pelacakan mAP, F1-Score, Precision, Recall, serta statistik TP, FP, dan FN.
* **Visualisasi Otomatis**: Menghasilkan grafik tren Loss, Learning Rate, dan performa deteksi secara otomatis (`.png`).
* **Keamanan Operasional**: Mekanisme **Auto-stop** jika terdeteksi nilai `NaN` pada Loss.

---

## Spesifikasi Sistem (Hardware Reference)

Proses pelatihan dan pengujian repositori ini dilakukan menggunakan spesifikasi perangkat keras berikut:
* **GPU**: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM).
* **Catatan**: Fitur AMP dan Gradient Accumulation diaktifkan dalam skrip ini untuk memastikan kestabilan pelatihan pada kapasitas 8GB VRAM.

---

## Instalasi

Gunakan langkah-langkah berikut untuk menyiapkan lingkungan pengembangan:

1. Pastikan Anda menggunakan Python 3.7 atau versi yang lebih baru.
2. Instal seluruh dependensi melalui file [requirements.txt](./requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

---

## Konteks Aplikasi: Inspeksi Visual Otomatis

Implementasi ini diarahkan untuk standarisasi inspeksi visual pada lini produksi. Model dikonfigurasi untuk mengenali berbagai komponen pianika guna mendeteksi kelengkapan atau kecacatan produksi:

* **Objek Deteksi**: label, pianika_biru, hose, mouthpiece, case_biru, leaflet, buku_manual, case_pink, pianika_pink.
* **Integrasi**: Sistem Computer Vision ini terintegrasi langsung dengan perangkat lunak pendukung di folder [Software-Inspeksi](../Software-Inspeksi/).

---

## Struktur Direktori Dataset

Pastikan dataset Anda disusun secara relatif agar skrip dapat mendeteksi jalur file secara otomatis:

```text
.
├── datasets/
│   └── pianika_1/
│       ├── train/          # Gambar pelatihan (.jpg)
│       ├── valid/          # Gambar validasi (.jpg)
│       └── annotations/    # File .json format COCO
├── projects/
│   └── pianika.yml         # Konfigurasi parameter proyek
├── train_d0.py             # Skrip pelatihan EfficientDet-D0
└── train_d1.py             # Skrip pelatihan EfficientDet-D1
```

---

## Panduan Pelatihan

Gunakan skrip yang sesuai dengan koefisien model yang ingin Anda latih:

* **EfficientDet-D0**: Jalankan [train_d0.py](./train_d0.py)
    ```bash
    python train_d0.py -p pianika --batch_size 2 --grad_accumulation_steps 4 --use_amp True
    ```
* **EfficientDet-D1**: Jalankan [train_d1.py](./train_d1.py)
    ```bash
    python train_d1.py -p pianika --batch_size 2 --grad_accumulation_steps 4 --use_amp True
    ```

Sistem akan otomatis menyimpan laporan metrik dan grafik performa di direktori `logs/` setelah proses selesai.

---

## Analisis Perbandingan Teknis

Proyek ini memperbaiki beberapa ketidaksesuaian algoritma agar selaras dengan arsitektur asli TensorFlow:
1. **Batch Normalization**: Penyesuaian momentum sistem.
2. **Depthwise-Separable Conv2D**: Penempatan BiasAdd yang akurat.
3. **Integrasi BiFPN**: Jalur fitur dan bobot koneksi yang presisi.
4. **Padding**: Implementasi Static Same Padding.

---

## Referensi
- [google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [signatrix/efficientdet](https://github.com/signatrix/efficientdet)
