import pandas as pd

# 1. Membaca file CSV
# file_name = 'data_skripsi_log_produksi_station_1_v2.csv'
file_name = 'data_skripsi_log_produksi_station_2_v2.csv'
df = pd.read_csv(file_name)

# 2. Menghitung statistik dari kolom Latency_ms
jumlah_data = len(df)
rata_rata_latency = df['Latency_ms'].mean()
std_latency = df['Latency_ms'].std()  # Standar Deviasi
min_latency = df['Latency_ms'].min()  # Nilai Minimum
max_latency = df['Latency_ms'].max()  # Nilai Maksimum

# 3. Menampilkan hasil
print(f"--- Hasil Analisis Latency ---")
print(f"Total Data Terproses : {jumlah_data} data")
print(f"Rata-rata Latency    : {rata_rata_latency:.2f} ms")
print(f"Standar Deviasi      : {std_latency:.2f} ms")
print(f"Nilai Minimum        : {min_latency:.2f} ms")
print(f"Nilai Maksimum       : {max_latency:.2f} ms")

print(df['Latency_ms'].describe())