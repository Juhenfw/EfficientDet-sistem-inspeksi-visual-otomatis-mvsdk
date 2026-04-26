import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import json
import os
import time
from datetime import datetime
import csv

# --- IMPORT AI EFFICIENTDET ---
import torch
import yaml
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

# --- IMPORT MINDVISION KAMERA ---
import mvsdk 

# Import UI terbaru (Pastikan nama file dan foldernya benar)
from GUI_v5 import InspectionUI, STATUS_COLORS, get_scaled_size

# =====================================================================
# PENGATURAN MODE STATION & KAMERA
# =====================================================================
STATION_MODE = 1  # 1 untuk Aksesoris, 2 untuk Case & Pianika
KAMERA_INDEX = 0  # 0 untuk kamera pertama yg tercolok, 1 untuk yg kedua
SHARED_FILE = "antrean_produksi.json" 
CONFIG_FILE = "configs/camera1_config.json"
OPERATOR_FILE = "configs/operator.json" # <--- File baru untuk NIK & Nama

# =====================================================================
# KELAS MANAJEMEN ANTREAN CERDAS (SMART QUEUE IPC)
# =====================================================================
class SyncManager:
    def __init__(self, file_path=SHARED_FILE):
        self.file_path = file_path
        self.default_data = {
            "Pianika Model P32E (Biru)": 0,
            "Pianika Model P32EP (Pink)": 0
        }
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.file_path):
            self.write_data(self.default_data)

    def read_data(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except:
            return self.default_data

    def write_data(self, data_dict):
        try:
            with open(self.file_path, 'w') as f:
                json.dump(data_dict, f, indent=4)
        except Exception as e:
            print(f"[IPC ERROR] Gagal menulis sinkronisasi: {e}")

    def get_queue(self, model_name):
        return self.read_data().get(model_name, 0)

    def tambah_antrean(self, model_name):
        data = self.read_data()
        data[model_name] = data.get(model_name, 0) + 1
        self.write_data(data)

    def kurangi_antrean(self, model_name):
        data = self.read_data()
        current = data.get(model_name, 0)
        data[model_name] = max(0, current - 1)
        self.write_data(data)

# =====================================================================
# KELAS DATA MANAJER (STATISTIK HARIAN & LOGGING CSV)
# =====================================================================
class DataManager:
    def __init__(self, station_id):
        self.station_id = station_id
        self.stats_file = f"statistik_station_{station_id}.json"
        self.log_file = f"log_produksi_station_{station_id}.csv"
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.base_img_path = os.path.join("hasil_inspeksi", f"Station_{station_id}", today)
        
        os.makedirs(self.base_img_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_img_path, "PASS"), exist_ok=True)
        os.makedirs(os.path.join(self.base_img_path, "FAIL"), exist_ok=True)
        os.makedirs(os.path.join(self.base_img_path, "MANUAL"), exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Waktu", "Station", "Model", "Status", "Komponen Kurang", "No_Tag", "NIK", "Nama_Operator", "Latency_ms"])

    def load_daily_stats(self):
        today = datetime.now().strftime("%Y-%m-%d")
        default_stats = {
            "tanggal": today,
            "Pianika Model P32E (Biru)": {"total": 0, "pass": 0, "fail": 0},
            "Pianika Model P32EP (Pink)": {"total": 0, "pass": 0, "fail": 0}
        }
        
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                if data.get("tanggal") == today:
                    return data
            except: 
                pass
            
        self.save_daily_stats(default_stats)
        return default_stats

    def save_daily_stats(self, stats_dict):
        with open(self.stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=4)

    def catat_log_csv(self, model, is_pass, missing_items, operator_data, latency_ms):
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_teks = "PASS" if is_pass else "FAIL"
        kurang_teks = ", ".join(missing_items) if missing_items else "Lengkap"
        
        notag = operator_data.get("notag", "UNKNOWN")
        nik = operator_data.get("nik", "UNKNOWN")
        nama = operator_data.get("nama", "UNKNOWN")
        
        try:
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([waktu, f"Station {self.station_id}", model, status_teks, kurang_teks, notag, nik, nama, f"{latency_ms:.2f}"])
        except Exception as e:
            print(f"[LOG ERROR] Gagal menulis ke CSV: {e}")

# =====================================================================
# KELAS LOGIKA AI & INSPEKSI (TETAP SAMA SEPERTI SEBELUMNYA)
# =====================================================================
class InspectionEngine:
    def __init__(self):
        self.compound_coef = 1
        self.threshold = 0.5
        self.iou_threshold = 0.5
        self.max_input_size = 640 
        self.model_path = r'EfficientDet-sistem-inspeksi-visual-otomatis-mvsdk\Software-Inspeksi\models\best_loss_d1.pth'
        self.project_yml = r'EfficientDet-sistem-inspeksi-visual-otomatis-mvsdk\Software-Inspeksi\projects\pianika.yml'
        
        # Target Stage 1
        self.target_stage_1 = ['label', 'hose', 'mouthpiece', 'leaflet', 'buku_manual']
        
        # Target stage 2
        self.target_stage_2_p32e = ['pianika_biru', 'case_biru']
        self.target_stage_2_p32ep = ['pianika_pink', 'case_pink']
        
        self.config_dir = r'EfficientDet-sistem-inspeksi-visual-otomatis-mvsdk\Software-Inspeksi\configs'
        self.all_zones = {
            "station_1_p32e": self.load_json_zone('zones_station_1_P32E.json'),
            "station_2_p32e": self.load_json_zone('zones_station_2_P32E.json'),
            "station_1_p32ep": self.load_json_zone('zones_station_1_P32EP.json'),
            "station_2_p32ep": self.load_json_zone('zones_station_2_P32EP.json')
        }
        
        try:
            with open(self.project_yml, 'r') as f:
                params = yaml.safe_load(f)
                self.obj_list = params['obj_list']
                ratios = eval(params['anchors_ratios'])
                scales = eval(params['anchors_scales'])
        except Exception as e:
            print(f"[AI ERROR] {e}")
            return

        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list), ratios=ratios, scales=scales)
        
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.requires_grad_(False)
            self.model.eval()
            self.model.to(self.device)
            print("[AI] Model Siap! ✓")
        except Exception as e: pass

    def load_json_zone(self, filename):
        path = os.path.join(self.config_dir, filename)
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[ZONA ERROR] {e}")
        return {} 

    def get_active_zones(self, current_model):
        is_pink = "Pink" in current_model or "P32EP" in current_model
        if STATION_MODE == 1:
            return self.all_zones["station_1_p32ep"] if is_pink else self.all_zones["station_1_p32e"]
        else: 
            return self.all_zones["station_2_p32ep"] if is_pink else self.all_zones["station_2_p32e"]

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if float(boxAArea + boxBArea - interArea) == 0: return 0.0
        return interArea / float(boxAArea + boxBArea - interArea)

    def run_inference(self, image_np, current_model):
        # ==========================================
        # KONFIGURASI VISUAL
        # ==========================================
        box_thickness = 3          # Ketebalan kotak deteksi AI
        zone_thickness = 2         # Ketebalan kotak zona kalibrasi
        label_font_scale = 0.6     # Ukuran font label hasil deteksi
        label_font_thick = 2       # Ketebalan font label hasil deteksi
        zone_font_scale = 0.5      # Ukuran font label zona
        # ==========================================

        detected_items_status = {}
        result_image = image_np.copy() 
        img_h, img_w = result_image.shape[:2]
        
        if not hasattr(self, 'model'): return detected_items_status, result_image

        active_zones = self.get_active_zones(current_model)

        # 1. Gambar zona kalibrasi
        for obj_name, rel_coords in active_zones.items():
            rx1, ry1 = int(rel_coords[0] * img_w), int(rel_coords[1] * img_h)
            rx2, ry2 = int(rel_coords[2] * img_w), int(rel_coords[3] * img_h)
            # Gunakan zone_thickness
            cv2.rectangle(result_image, (rx1, ry1), (rx2, ry2), (0, 255, 0), zone_thickness)
            # Gunakan zone_font_scale
            cv2.putText(result_image, f"Zone {obj_name}", (rx1, ry1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, zone_font_scale, (0, 255, 0), 1)

        temp_path = "temp_inference.bmp"
        cv2.imwrite(temp_path, result_image)

        try:
            ori_imgs, framed_imgs, framed_metas = preprocess(temp_path, max_size=self.max_input_size)
            x = torch.from_numpy(framed_imgs[0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()

            with torch.no_grad():
                features, regression, classification, anchors = self.model(x)
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, self.threshold, self.iou_threshold)

            out = invert_affine(framed_metas, out)

            if len(out[0]['rois']) > 0:
                is_pink_mode = "Pink" in current_model or "P32EP" in current_model
                
                for i in range(len(out[0]['rois'])):
                    score = float(out[0]['scores'][i])
                    class_id = int(out[0]['class_ids'][i])
                    class_name = self.obj_list[class_id]
                    
                    if is_pink_mode and "biru" in class_name.lower(): continue
                    if not is_pink_mode and "pink" in class_name.lower(): continue
                    if class_name in detected_items_status: continue

                    x1, y1 = int(out[0]['rois'][i][0]), int(out[0]['rois'][i][1])
                    x2, y2 = int(out[0]['rois'][i][2]), int(out[0]['rois'][i][3])
                    det_box = [x1, y1, x2, y2]
                    
                    posisi_ok = True 
                    if class_name in active_zones:
                        rel_c = active_zones[class_name]
                        ref_box = [int(rel_c[0]*img_w), int(rel_c[1]*img_h), int(rel_c[2]*img_w), int(rel_c[3]*img_h)]
                        iou_score = self.calculate_iou(det_box, ref_box)
                        if iou_score < 0.3: 
                            posisi_ok = False
                            
                    status_teks = "OK" if posisi_ok else "SALAH"
                    detected_items_status[class_name] = status_teks
                    
                    # --- LOGIKA VISUALISASI DENGAN VARIABEL ---
                    box_color = (255, 255, 0) if posisi_ok else (0, 0, 255) 
                    
                    # A. Gambar Bounding Box utama
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), box_color, box_thickness)
                    
                    # B. Hitung ukuran background teks secara dinamis
                    label_text = f"{class_name} {score:.2f} [{status_teks}]"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                label_font_scale, label_font_thick)
                    
                    # C. Gambar background label (kotak penuh warna)
                    y_bg_top = max(0, y1 - text_h - baseline - 10)
                    cv2.rectangle(result_image, (x1, y_bg_top), (x1 + text_w, y1), box_color, -1)
                    
                    # D. Gambar teks label
                    y_text = max(text_h + 5, y1 - 7)
                    cv2.putText(result_image, label_text, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 
                                label_font_scale, (0, 0, 0), label_font_thick, cv2.LINE_AA)

        except Exception as e: 
            print(f"[AI ERROR] {e}")
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

        return detected_items_status, result_image

    def evaluasi_hasil(self, detected_items_status, current_model):
        if STATION_MODE == 1:
            targets = self.target_stage_1
        else:
            if "Pink" in current_model or "P32EP" in current_model:
                targets = self.target_stage_2_p32ep
            else:
                targets = self.target_stage_2_p32e

        missing_items = []
        hasil_display = []

        for target in targets:
            if target in detected_items_status:
                status_posisi = detected_items_status[target]
                if status_posisi == "OK":
                    hasil_display.append((target.replace('_', ' ').title(), "Lolos", "✓", STATUS_COLORS.get("pass", "#2ECC71")))
                else:
                    # Kalau terdeteksi tapi IoU-nya meleset (Salah Posisi)
                    hasil_display.append((target.replace('_', ' ').title(), "NG (Posisi/Hilang)", "⚠", STATUS_COLORS.get("fail", "#E74C3C")))
                    missing_items.append(f"{target} (NG)")
            else:
                # Kalau AI buta karena barang ditaruh sembarangan (Tidak Ada)
                hasil_display.append((target.replace('_', ' ').title(), "NG (Posisi/Hilang)", "✗", STATUS_COLORS.get("fail", "#E74C3C")))
                missing_items.append(f"{target} (NG)")
                
        is_pass = len(missing_items) == 0
        return is_pass, hasil_display, missing_items


# =====================================================================
# KONTROLER UTAMA
# =====================================================================
class MainController:
    def __init__(self, root):
        self.root = root
        self.ui = InspectionUI(root)
        self.sync_mgr = SyncManager()
        self.data_mgr = DataManager(STATION_MODE)
        self.engine = InspectionEngine()
        
        # Variabel Status Kamera & Timer
        self.hCamera = None
        self.pFrameBuffer = None
        self.camera_running = False
        self.current_frame = None
        self.is_inspecting = False 
        self.auto_reset_timer = None 
        
        # Variabel Cache untuk Menyimpan Hasil Evaluasi Sementara
        self.last_eval_pass = None
        self.last_missing_items = None
        self.last_result_image = None
        self.last_model_inspected = None
        
        self.stats = self.data_mgr.load_daily_stats() 
        
        # --- BINDING UI BARU ---
        self.ui.btn_inspect.configure(command=self.mulai_inspeksi)  # Dulu ambil gambar
        self.ui.btn_save.configure(command=self.simpan_hasil)       # Dulu mulai inspeksi
        self.ui.btn_reset.configure(command=self.reset_sistem)
        self.ui.exit_btn.configure(command=self.exit_aplikasi)
        self.ui.root.protocol("WM_DELETE_WINDOW", self.exit_aplikasi)
        
        # Binding saat dropdown model berubah
        self.ui.model_dropdown.configure(command=self.on_model_change)
        
        # Variabel Status Operator
        self.operator_aktif = None 
        # Kunci tombol saat aplikasi baru dibuka
        self.kunci_tombol_inspeksi()
        
        # Setup Awal UI
        self.setup_ui_station()
        # --- LOGIKA OPERATOR BARU ---
        self.operator_db = []
        self.load_operator_db() # Load database JSON
        self.ui.entry_notag.bind("<Return>", self.proses_scan_notag) # Deteksi jika alat scan menekan Enter
        self.ui.btn_ok_tag.configure(command=self.proses_scan_notag) # Untuk Touchscreen
        self.ui.entry_notag.focus() # Auto-fokus kursor ke kolom input saat aplikasi dibuka
        # ----------------------------
        self.update_ui_stats()
        self.update_guide_image(self.ui.model_dropdown.get()) # Load Gambar Panduan Awal
        
        # INISIALISASI LIVE CAMERA
        self.init_camera()
        if STATION_MODE == 2: self.pantau_status_station1()

    # --- Fungsi Bantuan untuk Mengunci/Membuka Tombol ---
    def kunci_tombol_inspeksi(self):
        self.ui.btn_inspect.configure(state="disabled", fg_color="#7F8C8D")
        self.ui.btn_save.configure(state="disabled", fg_color="#7F8C8D")
        self.ui.update_status("Sistem Terkunci. Silakan Scan Tag Operator!", STATUS_COLORS["fail"])

    def buka_tombol_inspeksi(self):
        self.ui.btn_inspect.configure(state="normal", fg_color=STATUS_COLORS["accent"])
        self.ui.btn_save.configure(state="normal", fg_color=STATUS_COLORS["pass"])

    # --- FUNGSI BARU UNTUK UI 3 KOLOM ---
    def load_operator_db(self):
        """Membaca file JSON yang berisi daftar operator (berbentuk List)"""
        try:
            if not os.path.exists(OPERATOR_FILE):
                os.makedirs(os.path.dirname(OPERATOR_FILE), exist_ok=True)
                dummy_data = [
                    {"notag": "00012345", "nik": "1012304", "nama": "Juhen FW"},
                    {"notag": "00098765", "nik": "1012305", "nama": "Budi Santoso"}
                ]
                with open(OPERATOR_FILE, 'w') as f:
                    json.dump(dummy_data, f, indent=4)
                    
            with open(OPERATOR_FILE, 'r') as f:
                self.operator_db = json.load(f)
        except Exception as e:
            print(f"[OPERATOR WARN] Gagal meload database operator: {e}")

    def proses_scan_notag(self, event=None):
        """Mendeteksi NIK dan Nama berdasarkan input No Tag dari Scanner/Tombol"""
        scanned_tag = self.ui.entry_notag.get().strip()
        
        operator_ditemukan = False
        for op in self.operator_db:
            if op.get("notag") == scanned_tag:
                self.ui.set_operator_info(op.get("nik"), op.get("nama"))
                operator_ditemukan = True
                
                # --- REVISI 3: Simpan Data Operator & Buka Kunci Tombol ---
                self.operator_aktif = op 
                self.buka_tombol_inspeksi()
                self.ui.update_status(f"Login Sukses: {op.get('nama')}. Siap Inspeksi.", STATUS_COLORS["pass"])
                break
                
        if not operator_ditemukan:
            self.ui.set_operator_info("TIDAK VALID", "TIDAK TERDAFTAR")
            self.operator_aktif = None
            self.kunci_tombol_inspeksi() # Tetap kunci jika scan salah
            
        self.ui.entry_notag.delete(0, tk.END)

    def update_guide_image(self, model_name):
        """Memperbarui gambar panduan di panel kiri sesuai model yang dipilih"""
        is_pink = "Pink" in model_name or "P32EP" in model_name
        
        # Pastikan Anda meletakkan gambar panduan di dalam folder assets
        img_path = "assets/guide_station_1_pink.jpg" if is_pink else "assets/guide_station_1_biru.jpg"
        
        if STATION_MODE == 2:
            img_path = "assets/guide_station_2_pink.jpg" if is_pink else "assets/guide_station_2_biru.jpg"
            
        self.ui.set_guide_image(img_path)

    def on_model_change(self, selected_model):
        """Event ketika operator mengganti dropdown model"""
        self.update_ui_stats()
        self.update_guide_image(selected_model)

    def update_ui_stats(self):
        current_model = self.ui.model_dropdown.get()
        if current_model not in self.stats:
            self.stats[current_model] = {"total": 0, "pass": 0, "fail": 0}
            
        data = self.stats[current_model]
        self.ui.lbl_stat_total.configure(text=str(data["total"]))
        self.ui.lbl_stat_pass.configure(text=str(data["pass"]))
        self.ui.lbl_stat_fail.configure(text=str(data["fail"]))
        
    def simpan_stats(self):
        self.data_mgr.save_daily_stats(self.stats)

    def setup_ui_station(self):
        if STATION_MODE == 1:
            teks_judul = "LIVE CAMERA FEED (STATION 1: AKSESORIS)"
        else:
            teks_judul = "LIVE CAMERA FEED (STATION 2: UNIT UTAMA)"
            self.ui.btn_inspect.configure(state="disabled", fg_color="#7F8C8D")
            self.ui.update_status("Menunggu Antrean Station 1...", STATUS_COLORS["warning"])
            
        for widget in self.ui.cam_status_frame.master.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and "LIVE CAMERA" in widget.cget("text"):
                widget.configure(text=teks_judul)

    def init_camera(self):
        try:
            DevList = mvsdk.CameraEnumerateDevice()
            if len(DevList) <= KAMERA_INDEX:
                print(f"[KAMERA ERROR] Kamera index {KAMERA_INDEX} tidak ditemukan!")
                self.ui.lbl_cam_status.configure(text="● OFFLINE", text_color=STATUS_COLORS["fail"])
                self.ui.update_status("Kamera Tidak Terdeteksi!", STATUS_COLORS["fail"])
                self.ui.draw_camera_placeholder()
                return

            DevInfo = DevList[KAMERA_INDEX]
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            cap = mvsdk.CameraGetCapability(self.hCamera)
            
            modeMono = (cap.sIspCapacity.bMonoSensor != 0)
            fmt = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if modeMono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
            mvsdk.CameraSetIspOutFormat(self.hCamera, fmt)
            
            if os.path.exists(CONFIG_FILE):
                try:
                    with open(CONFIG_FILE, 'r') as f:
                        cfg = json.load(f)
                    mvsdk.CameraSetExposureTime(self.hCamera, cfg.get("exposure_time", 1000))
                    mvsdk.CameraSetGamma(self.hCamera, cfg.get("gamma", 20))
                    mvsdk.CameraSetContrast(self.hCamera, cfg.get("contrast", 120))
                    gain_val = int(cfg.get("gain", 16.5) / 0.125)
                    mvsdk.CameraSetAnalogGain(self.hCamera, gain_val)
                    wb_mode = cfg.get("white_balance_mode", 0)
                    mvsdk.CameraSetWbMode(self.hCamera, wb_mode)
                    if wb_mode == 0:
                        mvsdk.CameraSetGain(self.hCamera, cfg.get("r_gain", 100), cfg.get("g_gain", 100), cfg.get("b_gain", 100))
                except Exception as e: pass
            
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            mvsdk.CameraSetAeState(self.hCamera, 0) 
            mvsdk.CameraPlay(self.hCamera)
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if modeMono else 3), 16)
            
            self.camera_running = True
            self.ui.lbl_cam_status.configure(text="● ONLINE", text_color="white")
            self.ui.update_status("Kamera Siap.", STATUS_COLORS["info"])
            
            # --- PERBAIKAN 1: Matikan Placeholder ---
            # Mencegah canvas terus-menerus menggambar ulang teks "Menunggu Kamera" saat ukuran jendela berubah
            self.ui.camera_canvas.unbind("<Configure>")
            # ----------------------------------------
            
            self.update_video_feed()
            
        except Exception as e:
            print(f"[KAMERA ERROR] {e}")
            self.ui.lbl_cam_status.configure(text="● ERROR", text_color=STATUS_COLORS["fail"])

    def update_video_feed(self):
        if not self.camera_running or self.hCamera is None: return

        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            
            if os.name == 'nt': mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)
            
            # --- PERBAIKAN 2: Penentuan Ukuran Buffer Array yang Akurat ---
            channels = 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
            buffer_size = FrameHead.iHeight * FrameHead.iWidth * channels
            frame_data = (mvsdk.c_ubyte * buffer_size).from_address(self.pFrameBuffer)
            
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((FrameHead.iHeight, FrameHead.iWidth, channels))
            self.current_frame = frame.copy()
            
            if not self.is_inspecting:
                canvas_w = self.ui.camera_canvas.winfo_width()
                canvas_h = self.ui.camera_canvas.winfo_height()
                
                if canvas_w > 10 and canvas_h > 10:
                    h, w = frame.shape[:2]
                    aspect = w / h
                    if canvas_w / canvas_h > aspect:
                        new_h = canvas_h
                        new_w = int(canvas_h * aspect)
                    else:
                        new_w = canvas_w
                        new_h = int(canvas_w / aspect)
                        
                    frame_resized = cv2.resize(frame, (max(1, new_w), max(1, new_h)))
                    
                    # --- PERBAIKAN 3: Konversi Warna Aman ---
                    if channels == 3:
                        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    else:
                        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2RGB)
                        
                    img_pil = Image.fromarray(img_rgb)
                    self.tk_image = ImageTk.PhotoImage(image=img_pil)
                    
                    self.ui.camera_canvas.delete("all")
                    self.ui.camera_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_image, anchor="center")
                    
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT: 
                print(f"[KAMERA SDK] Skip frame: {e}")
                
        # --- PERBAIKAN 4: Tangkap Semua Error ---
        except Exception as e:
            print(f"[VIDEO LOOP ERROR] {e}") # Print error jika numpy/cv2 gagal
            
        finally:
            # --- PERBAIKAN 5: Pastikan Loop Selalu Berjalan ---
            # Dengan taruh di 'finally', meskipun ada 1 frame yang corrupt, 
            # aplikasi tidak akan mati dan akan terus mengambil frame berikutnya.
            if self.camera_running:
                self.root.after(30, self.update_video_feed)

    def pantau_status_station1(self):
        self.update_ui_stats()
        current_model = self.ui.model_dropdown.get()
        antrean_model_ini = self.sync_mgr.get_queue(current_model)
        
        model_singkat = "P32EP" if "Pink" in current_model or "P32EP" in current_model else "P32E"
        for widget in self.ui.cam_status_frame.master.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and "OFFLINE MODE" in widget.cget("text"):
                widget.configure(text=f"OFFLINE MODE (STATION 2) - Antrean {model_singkat}: {antrean_model_ini}")

        # --- PERBAIKAN LOGIKA INTERLOCK STATION 2 ---
        if self.operator_aktif is None:
            # Syarat 1 Gagal: Operator belum scan -> Paksa tombol terkunci
            if self.ui.btn_inspect.cget("state") == "normal":
                self.kunci_tombol_inspeksi()
        else:
            # Operator SUDAH login. Sekarang cek antreannya.
            if antrean_model_ini > 0:
                # Syarat 1 dan 2 Terpenuhi -> Nyalakan tombol
                if self.ui.btn_inspect.cget("state") == "disabled":
                    self.buka_tombol_inspeksi()
                    self.ui.update_status("Siap Inspeksi Tahap 2", STATUS_COLORS["pass"])
            else:
                # Syarat 2 Gagal: Antrean kosong -> Matikan tombol & tunggu antrean
                if self.ui.btn_inspect.cget("state") == "normal":
                    self.ui.btn_inspect.configure(state="disabled", fg_color="#7F8C8D")
                    self.ui.btn_save.configure(state="disabled", fg_color="#7F8C8D")
                    self.ui.update_status(f"Menunggu Antrean {model_singkat}...", STATUS_COLORS["warning"])
                
        self.root.after(500, self.pantau_status_station1)

    # =====================================================================
    # LOGIKA BARU: MULAI INSPEKSI (Hanya Analisis & Tahan Layar)
    # =====================================================================
    # =====================================================================
    # LOGIKA BARU: MULAI INSPEKSI (Hanya Analisis & Tahan Layar)
    # =====================================================================
    def mulai_inspeksi(self):
        # Batalkan timer reset yang mungkin masih berjalan dari sesi sebelumnya
        if self.auto_reset_timer is not None:
            self.root.after_cancel(self.auto_reset_timer)
            self.auto_reset_timer = None 

        if self.operator_aktif is None:
            self.ui.update_status("Ditolak: Operator belum login!", STATUS_COLORS["fail"])
            return

        if self.current_frame is None:
            self.ui.update_status("Gagal: Belum memuat gambar!", STATUS_COLORS["fail"])
            return
            
        current_model = self.ui.model_dropdown.get()
        self.ui.update_status("Sedang Menganalisis...", STATUS_COLORS["warning"])
        self.is_inspecting = True # Tahan layar kamera
        self.root.update_idletasks() 
        
        # =========================================================
        # [START TIMER]: Catat waktu mulai dengan presisi tinggi
        # =========================================================
        start_time = time.perf_counter()
        
        # 0. Simpan gambar mentah (raw) saat ini ke folder MANUAL
        waktu_raw = datetime.now().strftime("%H%M%S")
        nama_file_raw = f"{current_model.split(' ')[2]}_{waktu_raw}_RAW.jpg"
        path_simpan_raw = os.path.join(self.data_mgr.base_img_path, "MANUAL", nama_file_raw)
        
        try:
            cv2.imwrite(path_simpan_raw, self.current_frame)
            print(f"[LOG] Gambar raw tersimpan: {nama_file_raw}")
        except Exception as e:
            print(f"[LOG ERROR] Gagal menyimpan gambar raw: {e}")
        
        # 1. Jalankan Inferensi AI
        detected_items, result_image_bgr = self.engine.run_inference(self.current_frame, current_model)
        
        # 2. Tampilkan Hasil Visual ke Canvas
        try:
            canvas_w = self.ui.camera_canvas.winfo_width()
            canvas_h = self.ui.camera_canvas.winfo_height()
            if canvas_w > 10 and canvas_h > 10:
                h, w = result_image_bgr.shape[:2]
                aspect = w / h
                if canvas_w / canvas_h > aspect:
                    new_h = canvas_h
                    new_w = int(canvas_h * aspect)
                else:
                    new_w = canvas_w
                    new_h = int(canvas_w / aspect)
                    
                frame_resized = cv2.resize(result_image_bgr, (max(1, new_w), max(1, new_h)))
                img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                self.tk_image_result = ImageTk.PhotoImage(image=img_pil)
                
                self.ui.camera_canvas.delete("all")
                self.ui.camera_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_image_result, anchor="center")
        except Exception as e: 
            print(f"[UI ERROR] Gagal merender freeze frame: {e}")

        # 3. Evaluasi Hasil Kelengkapan & Tampilkan List
        is_pass, hasil_display, missing_items = self.engine.evaluasi_hasil(detected_items, current_model)
        self.update_list_komponen(hasil_display)
        
        # =========================================================
        # [STOP TIMER]: Catat waktu selesai dan hitung selisihnya
        # =========================================================
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000 # Konversi ke milidetik

        # 4. SIMPAN STATE KE VARIABEL CACHE (Untuk dipakai saat tekan tombol Simpan)
        self.last_eval_pass = is_pass
        self.last_missing_items = missing_items
        self.last_result_image = result_image_bgr
        self.last_model_inspected = current_model
        self.last_latency_ms = latency_ms # [REVISI TIMER]: Simpan nilai latency ke dalam cache

        # 5. Tampilkan Informasi ke Terminal dan UI
        print(f"[LATENCY] Waktu Eksekusi Inspeksi: {latency_ms:.2f} ms")
        status_text = f"Inspeksi Selesai ({latency_ms:.0f} ms). Silakan klik SIMPAN HASIL."
        self.ui.update_status(status_text, STATUS_COLORS["info"])

    # =====================================================================
    # LOGIKA BARU: SIMPAN HASIL (Statistik, IPC, & Rekam Gambar)
    # =====================================================================
    def simpan_hasil(self):
        # Proteksi: Pastikan operator sudah menekan 'Mulai Inspeksi' sebelumnya
        if not self.is_inspecting or self.last_result_image is None:
            self.ui.update_status("Lakukan 'Mulai Inspeksi' terlebih dahulu!", STATUS_COLORS["warning"])
            return
        
        if self.operator_aktif is None:
            return

        # Tarik data dari cache
        is_pass = self.last_eval_pass
        missing_items = self.last_missing_items
        result_image_bgr = self.last_result_image
        current_model = self.last_model_inspected
        latency_ms = getattr(self, 'last_latency_ms', 0.0)

        # 1. Update Statistik JSON
        if current_model not in self.stats:
            self.stats[current_model] = {"total": 0, "pass": 0, "fail": 0}
            
        self.stats[current_model]["total"] += 1
        if is_pass:
            self.stats[current_model]["pass"] += 1
            status_teks = "LULUS"
            # Antrean IPC
            if STATION_MODE == 1: self.sync_mgr.tambah_antrean(current_model)
        else:
            self.stats[current_model]["fail"] += 1
            status_teks = "TIDAK LENGKAP"

        if STATION_MODE == 2: self.sync_mgr.kurangi_antrean(current_model)

        self.update_ui_stats()
        self.simpan_stats()
        # --- REVISI 5: Kirim self.operator_aktif ke pencatat CSV ---
        self.data_mgr.catat_log_csv(current_model, is_pass, missing_items, self.operator_aktif, latency_ms)

        # 2. Simpan Gambar AI ke Folder PASS/FAIL
        waktu_file = datetime.now().strftime("%H%M%S")
        sub_folder = "PASS" if is_pass else "FAIL"
        nama_file_ai = f"{current_model.split(' ')[2]}_{waktu_file}_{sub_folder}.jpg"
        path_simpan_ai = os.path.join(self.data_mgr.base_img_path, sub_folder, nama_file_ai)
        cv2.imwrite(path_simpan_ai, result_image_bgr)

        # 3. Beri notifikasi berhasil dan jalankan auto-reset
        notif_color = STATUS_COLORS["pass"] if is_pass else STATUS_COLORS["fail"]
        self.ui.update_status(f"Disimpan ({status_teks}). Auto-reset 3s...", notif_color)
        
        if self.auto_reset_timer: 
            self.root.after_cancel(self.auto_reset_timer)
        # self.auto_reset_timer = self.root.after(3000, self.reset_sistem)
        self.auto_reset_timer = self.root.after(1000, self.reset_sistem)

    def update_list_komponen(self, hasil_display):
        for widget in self.ui.comp_scroll.winfo_children(): widget.destroy()
        item_height = get_scaled_size(60, self.ui.screen_width, self.ui.screen_height)
        lolos_count = 0
        total_count = len(hasil_display)
        
        for idx, (komponen, status, icon, color) in enumerate(hasil_display):
            if status == "Lolos": lolos_count += 1
            item_frame = ctk.CTkFrame(self.ui.comp_scroll, height=item_height, corner_radius=10, fg_color="#1e1e1e" if idx % 2 == 0 else "#252525")
            item_frame.pack(fill="x", pady=5, padx=5)
            item_frame.pack_propagate(False)
            ctk.CTkLabel(item_frame, text=icon, font=("Segoe UI", 24, "bold"), text_color=color, width=40).pack(side="left", padx=(15, 10))
            ctk.CTkLabel(item_frame, text=komponen, font=("Segoe UI", 14), anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(item_frame, text=status, font=("Segoe UI", 11, "bold"), text_color="white", fg_color=color, corner_radius=6, width=100, height=28).pack(side="right", padx=15)
            
        self.ui.comp_count_label.configure(text=f"{lolos_count}/{total_count}")

    def reset_sistem(self):
        if self.auto_reset_timer is not None:
            self.root.after_cancel(self.auto_reset_timer)
            self.auto_reset_timer = None
            
        # Kosongkan cache inspeksi
        self.last_eval_pass = None
        self.last_missing_items = None
        self.last_result_image = None
        self.last_model_inspected = None
        self.is_inspecting = False 
        
        self.ui.update_status("Sistem Direset. Menunggu...", STATUS_COLORS["info"])
        for widget in self.ui.comp_scroll.winfo_children(): widget.destroy()
        self.ui.comp_count_label.configure(text="0/0")

    def exit_aplikasi(self):
        print("\n[SISTEM] Menutup aplikasi secara paksa...")
        self.camera_running = False
        try: self.simpan_stats()
        except: pass
        try: self.ui.root.withdraw()
        except: pass
        time.sleep(0.2)
        if self.hCamera:
            try:
                mvsdk.CameraStop(self.hCamera)
                mvsdk.CameraUnInit(self.hCamera)
                if self.pFrameBuffer: mvsdk.CameraAlignFree(self.pFrameBuffer)
            except: pass
        os._exit(0)

if __name__ == "__main__":
    root = ctk.CTk()
    app = MainController(root)
    root.mainloop()