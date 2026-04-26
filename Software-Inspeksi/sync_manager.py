import json
import os
import time

class SyncManager:
    def __init__(self, file_path="shared_status.json"):
        self.file_path = file_path
        self._init_file()

    def _init_file(self):
        """Membuat file default jika belum ada"""
        if not os.path.exists(self.file_path):
            self.write_status({"tahap_1_pass": False, "waktu": "", "id_produk": ""})

    def read_status(self):
        """Membaca status saat ini (Digunakan oleh Software 2)"""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Gagal membaca status: {e}")
            return {"tahap_1_pass": False}

    def write_status(self, data_dict):
        """Menulis status baru (Digunakan oleh Software 1 & 2)"""
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data_dict, f, indent=4)
            return True
        except Exception as e:
            print(f"Gagal menulis status: {e}")
            return False

    def reset_status(self):
        """Mereset status setelah Tahap 2 selesai (Digunakan oleh Software 2)"""
        self.write_status({"tahap_1_pass": False, "waktu": "", "id_produk": ""})