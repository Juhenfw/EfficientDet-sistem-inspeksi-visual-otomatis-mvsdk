import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visual nilai Confusion Matrix
# Definisi: Lengkap (Positive / Pass) dan Tidak Lengkap (Negative / Fail)

# --- Stasiun 1: Fokus Kelengkapan Aksesori ---
cm_st1 = np.array([[689, 0],    # (TN, FP)
                   [3, 538]])   # (FN, TP)

# --- Stasiun 2: Fokus Verifikasi Varian Produk ---
cm_st2 = np.array([[669, 19],    # (TN, FP)
                   [12, 511]])   # (FN, TP)

labels = ['OK', 'NG']

# Pengaturan visualisasi
sns.set_theme(style="white")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Visualisasi Stasiun 1
sns.heatmap(cm_st1, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
ax1.set_title('Confusion Matrix Stasiun 1\n(Kelengkapan Aksesori)', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Prediksi', fontsize=12)
ax1.set_ylabel('Aktual', fontsize=12)

# Visualisasi Stasiun 2
sns.heatmap(cm_st2, annot=True, fmt='d', cmap='Greens', ax=ax2, 
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
ax2.set_title('Confusion Matrix Stasiun 2\n(Verifikasi Varian Produk)', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Prediksi', fontsize=12)
ax2.set_ylabel('Aktual', fontsize=12)

plt.tight_layout()

# Menyimpan output dalam format PDF
plt.savefig('Confusion_Matrix_Stasiun_1_dan_2.pdf')
plt.show()