import cv2
import numpy as np
import mvsdk
import time
import platform
import threading
import customtkinter as ctk
from tkinter import messagebox, filedialog
import json
import os
import torch
import yaml
from PIL import Image, ImageTk

# Import EfficientDet components
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

class ZoneCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spatial Logic Calibrator Tool - Undergraduate Thesis Juhen's Edition")
        self.root.geometry("1250x850")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # State Hardware & Config
        self.hCamera = None
        self.pFrameBuffer = None
        self.running = False
        self.current_frame = None
        self.station_id = 1
        self.camera_index = 0 
        self.current_model = "P32E" 
        self.dev_list = []
        self.saved_zones = {}
        self.conf_threshold = 0.50 
        
        # --- STATE BARU: Mencegah Freeze saat AI Berjalan ---
        self.is_ai_busy = False 
        
        # UI Setup
        self.status_text = ctk.StringVar(value="SISTEM: Inisialisasi...")
        self.init_ai()
        self.build_gui()
        self.enumerate_cameras()
        self.load_zones_from_json()

    def get_cam_config_path(self):
        return os.path.join(CONFIG_DIR, f"camera{self.camera_index + 1}_config.json")

    def get_zone_path(self):
        return os.path.join(CONFIG_DIR, f"zones_station_{self.station_id}_{self.current_model}.json")

    def init_ai(self):
        try:
            project_yml = os.path.join(BASE_DIR, "projects", "pianika.yml")
            model_path = os.path.join(BASE_DIR, "models", "best_loss_d1.pth")
            with open(project_yml, 'r') as f:
                params = yaml.safe_load(f)
                self.obj_list = params['obj_list']
            self.model = EfficientDetBackbone(compound_coef=1, num_classes=len(self.obj_list), 
                                              ratios=eval(params['anchors_ratios']), 
                                              scales=eval(params['anchors_scales']))
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            self.model.eval()
            print("[AI] Model Ready.")
        except Exception as e: print(f"[AI ERROR] {e}")

    def enumerate_cameras(self):
        self.dev_list = mvsdk.CameraEnumerateDevice()
        if not self.dev_list:
            self.status_text.set("SISTEM: Kamera tidak ditemukan!")
        else:
            self.status_text.set(f"SISTEM: {len(self.dev_list)} Kamera terdeteksi.")

    def load_hardware_config(self):
        path = self.get_cam_config_path()
        if os.path.exists(path) and self.hCamera:
            try:
                with open(path, 'r') as f: cfg = json.load(f)
                mvsdk.CameraSetAeState(self.hCamera, 0)
                mvsdk.CameraSetExposureTime(self.hCamera, cfg.get("exposure_time", 10000))
                mvsdk.CameraSetGamma(self.hCamera, cfg.get("gamma", 100))
                mvsdk.CameraSetContrast(self.hCamera, cfg.get("contrast", 100))
                mvsdk.CameraSetAnalogGain(self.hCamera, int(cfg.get("gain", 1.0) / 0.125))
                mvsdk.CameraSetWbMode(self.hCamera, cfg.get("white_balance_mode", 0))
                if cfg.get("white_balance_mode", 0) == 0:
                    mvsdk.CameraSetGain(self.hCamera, int(cfg.get("r_gain", 100)), 
                                        int(cfg.get("g_gain", 100)), int(cfg.get("b_gain", 100)))
            except: pass

    def start_camera(self):
        if self.running: return
        try:
            if not self.dev_list: self.enumerate_cameras()
            if self.camera_index >= len(self.dev_list):
                messagebox.showerror("Error", "Kamera tidak tersedia!")
                return
            self.hCamera = mvsdk.CameraInit(self.dev_list[self.camera_index], -1, -1)
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            self.load_hardware_config()
            mvsdk.CameraPlay(self.hCamera)
            cap = mvsdk.CameraGetCapability(self.hCamera)
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3, 16)
            self.running = True
            self.btn_cam.configure(text="Stop Stream", fg_color="#e74c3c")
            
            threading.Thread(target=self.stream_worker, daemon=True).start()
            self.update_video_loop()
        except Exception as e: messagebox.showerror("Error", f"Start fail: {e}")

    def update_video_loop(self):
        """Loop rendering stabil di Main Thread"""
        if self.running and self.current_frame is not None:
            self.render_to_ui(self.current_frame.copy())
        
        if self.running:
            self.root.after(30, self.update_video_loop)

    def stream_worker(self):
        while self.running:
            pRaw = None
            try:
                pRaw, Head = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRaw, self.pFrameBuffer, Head)
                if platform.system() == "Windows": mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, Head, 1)
                
                frame_data = (mvsdk.c_ubyte * Head.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((Head.iHeight, Head.iWidth, 3))
                self.current_frame = frame.copy()
            except: pass
            finally:
                if pRaw is not None: mvsdk.CameraReleaseImageBuffer(self.hCamera, pRaw)
            time.sleep(0.01)

    def render_to_ui(self, display_img):
        h_img, w_img = display_img.shape[:2]
        for name, z in self.saved_zones.items():
            px1, py1 = int(z[0]*w_img), int(z[1]*h_img)
            px2, py2 = int(z[2]*w_img), int(z[3]*h_img)
            cv2.rectangle(display_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(display_img, name.upper(), (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        win_w, win_h = self.video_label.winfo_width(), self.video_label.winfo_height()
        if win_w > 100:
            scale = min(win_w/w_img, win_h/h_img)
            new_w, new_h = int(w_img*scale), int(h_img*scale)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            img_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(new_w, new_h))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label._image = ctk_img

    # --- PERBAIKAN: AI Inference di Background agar Tidak Freeze ---
    def start_learning_thread(self):
        if self.is_ai_busy: return
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Tidak ada gambar untuk dianalisis!")
            return
        
        # Matikan tombol agar tidak diklik berkali-kali
        self.btn_learn.configure(state="disabled", text="ANALYZING...")
        self.is_ai_busy = True
        self.status_text.set("SISTEM: Menjalankan AI di latar belakang...")
        
        # Jalankan proses AI di thread terpisah
        threading.Thread(target=self.run_learning_process, daemon=True).start()

    def run_learning_process(self):
        try:
            # Pakai copy frame saat ini agar thread aman
            frame_to_process = self.current_frame.copy()
            h_img, w_img = frame_to_process.shape[:2]
            temp_path = f"temp_calib_{int(time.time())}.bmp"
            cv2.imwrite(temp_path, frame_to_process)

            target_stage_1 = ['label', 'hose', 'mouthpiece', 'leaflet', 'buku_manual']
            target_stage_2 = ['pianika_biru', 'case_biru', 'pianika_pink', 'case_pink']
            active_targets = target_stage_1 if self.station_id == 1 else target_stage_2

            _, framed_imgs, framed_metas = preprocess(temp_path, max_size=640)
            x = torch.from_numpy(framed_imgs[0]).permute(2, 0, 1).unsqueeze(0).float()
            
            with torch.no_grad():
                _, reg, cls, anc = self.model(x)
                out = postprocess(x, anc, reg, cls, BBoxTransform(), ClipBoxes(), self.conf_threshold, 0.5)
            out = invert_affine(framed_metas, out)
            
            new_zones_data = {}
            if len(out[0]['rois']) > 0:
                is_pink_mode = "Pink" in self.current_model or "P32EP" in self.current_model
                for i in range(len(out[0]['rois'])):
                    class_name = self.obj_list[int(out[0]['class_ids'][i])]
                    if class_name not in active_targets: continue
                    if is_pink_mode and "biru" in class_name.lower(): continue
                    if not is_pink_mode and "pink" in class_name.lower(): continue
                    if class_name in new_zones_data: continue

                    x1, y1, x2, y2 = out[0]['rois'][i]
                    rx1, ry1, rx2, ry2 = float(x1/w_img), float(y1/h_img), float(x2/w_img), float(y2/h_img)
                    pad = 0.02 
                    zx1, zy1, zx2, zy2 = max(0.0, rx1-pad), max(0.0, ry1-pad), min(1.0, rx2+pad), min(1.0, ry2+pad)
                    new_zones_data[class_name] = [round(zx1, 3), round(zy1, 3), round(zx2, 3), round(zy2, 3)]

            # Simpan hasil (harus panggil thread-safe update ke UI di akhir)
            if new_zones_data:
                with open(self.get_zone_path(), 'w') as f: json.dump(new_zones_data, f, indent=4)
                self.root.after(0, lambda: self.finish_ai_process(True, len(new_zones_data)))
            else:
                self.root.after(0, lambda: self.finish_ai_process(False, 0))

            if os.path.exists(temp_path): os.remove(temp_path)

        except Exception as e:
            print(f"AI ERROR: {e}")
            self.root.after(0, lambda: self.finish_ai_process(False, 0, str(e)))

    def finish_ai_process(self, success, count, err_msg=""):
        self.is_ai_busy = False
        self.btn_learn.configure(state="normal", text="LEARN IDEAL POSITIONS")
        if success:
            self.load_zones_from_json()
            messagebox.showinfo("Success", f"Berhasil kalibrasi {count} zona.")
        else:
            self.status_text.set(f"GAGAL: {err_msg if err_msg else 'Objek tidak terdeteksi.'}")

    # =========================================================================

    def on_closing(self):
        self.running = False 
        time.sleep(0.3) 
        if self.hCamera:
            try:
                mvsdk.CameraStop(self.hCamera)
                mvsdk.CameraUnInit(self.hCamera)
                if self.pFrameBuffer: mvsdk.CameraAlignFree(self.pFrameBuffer)
            except: pass
        self.root.destroy()
        os._exit(0)

    def calibrate_from_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.bmp")])
        if filepath:
            img = cv2.imread(filepath)
            if img is not None:
                self.stop_camera()
                self.current_frame = img.copy()
                self.render_to_ui(self.current_frame.copy())
                self.start_learning_thread()

    def stop_camera(self):
        self.running = False
        self.btn_cam.configure(text="Start Stream", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        time.sleep(0.3)
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
            if self.pFrameBuffer: mvsdk.CameraAlignFree(self.pFrameBuffer)
            self.hCamera = None

    def toggle_camera(self):
        if not self.running: self.start_camera()
        else: self.stop_camera()

    def load_zones_from_json(self):
        path = self.get_zone_path()
        for widget in self.coord_scroll.winfo_children(): widget.destroy()
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: self.saved_zones = json.load(f)
                self.status_text.set(f"SISTEM: Load {os.path.basename(path)} Berhasil.")
                for name, z in self.saved_zones.items():
                    card = ctk.CTkFrame(self.coord_scroll, fg_color="#252525", corner_radius=8)
                    card.pack(fill="x", pady=4, padx=5)
                    ctk.CTkLabel(card, text=name.upper(), font=ctk.CTkFont(weight="bold", size=13), text_color="#2ecc71").pack(pady=(5,0), padx=10, anchor="w")
                    ctk.CTkLabel(card, text=f"Area: {z}", font=ctk.CTkFont(size=11), text_color="#aaaaaa").pack(pady=(0,5), padx=10, anchor="w")
            except: self.saved_zones = {}
        else:
            self.saved_zones = {}
            self.status_text.set(f"SISTEM: Belum ada zona.")
        if not self.running and self.current_frame is not None:
            self.render_to_ui(self.current_frame.copy())

    def build_gui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self.root, width=350, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        ctk.CTkLabel(self.sidebar, text="ZONE CONFIGURATOR", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=25)
        group_frame = ctk.CTkFrame(self.sidebar, fg_color="#1e1e1e", corner_radius=10)
        group_frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(group_frame, text="Select Station:").pack(pady=(10,0))
        self.st_combo = ctk.CTkComboBox(group_frame, values=["Station 1", "Station 2"], command=self.change_station, width=200)
        self.st_combo.pack(pady=5, padx=15); self.st_combo.set("Station 1")
        ctk.CTkLabel(group_frame, text="Select Model:").pack(pady=(5,0))
        self.mod_combo = ctk.CTkComboBox(group_frame, values=["P32E (Biru)", "P32EP (Pink)"], command=self.change_model, width=200)
        self.mod_combo.pack(pady=5, padx=15); self.mod_combo.set("P32E (Biru)")
        ctk.CTkLabel(group_frame, text="Select Camera:").pack(pady=(5,0))
        self.cam_combo = ctk.CTkComboBox(group_frame, values=["Camera 1", "Camera 2"], command=self.change_camera_index, width=200)
        self.cam_combo.pack(pady=5, padx=15); self.cam_combo.set("Camera 1")
        
        self.lbl_thresh = ctk.CTkLabel(group_frame, text=f"AI Confidence: {int(self.conf_threshold*100)}%")
        self.lbl_thresh.pack(pady=(10,0))
        self.slider_thresh = ctk.CTkSlider(group_frame, from_=0.1, to=1.0, command=self.change_threshold)
        self.slider_thresh.set(self.conf_threshold); self.slider_thresh.pack(pady=(0, 15), padx=20)
        
        self.coord_scroll = ctk.CTkScrollableFrame(self.sidebar, height=250, fg_color="#161616", corner_radius=10)
        self.coord_scroll.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.action_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.action_frame.pack(fill="x", padx=20, pady=10)
        self.btn_cam = ctk.CTkButton(self.action_frame, text="Start Live Stream", height=40, font=ctk.CTkFont(weight="bold"), command=self.toggle_camera)
        self.btn_cam.pack(fill="x", pady=5)
        self.btn_file = ctk.CTkButton(self.action_frame, text="Kalibrasi via Gambar Lokal", fg_color="#f39c12", hover_color="#d68910", height=40, font=ctk.CTkFont(weight="bold"), command=self.calibrate_from_file)
        self.btn_file.pack(fill="x", pady=5)
        
        # Gunakan fungsi perbaikan start_learning_thread
        self.btn_learn = ctk.CTkButton(self.action_frame, text="LEARN IDEAL POSITIONS", fg_color="#2ecc71", hover_color="#27ae60", height=50, font=ctk.CTkFont(weight="bold"), command=self.start_learning_thread)
        self.btn_learn.pack(fill="x", pady=10)
        
        self.preview_frame = ctk.CTkFrame(self.root, corner_radius=15, fg_color="#0a0a0a")
        self.preview_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.preview_frame.grid_rowconfigure(0, weight=1); self.preview_frame.grid_columnconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(self.preview_frame, text="CAMERA PREVIEW OFFLINE", font=ctk.CTkFont(size=18))
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        self.status_bar = ctk.CTkLabel(self.root, textvariable=self.status_text, anchor="w", fg_color="#1a1a1a", height=35)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def change_threshold(self, value):
        self.conf_threshold = float(value)
        self.lbl_thresh.configure(text=f"AI Confidence: {int(self.conf_threshold*100)}%")

    def change_station(self, choice):
        self.station_id = 1 if "1" in choice else 2
        self.load_zones_from_json()
        
    def change_model(self, choice):
        self.current_model = "P32EP" if "Pink" in choice else "P32E"
        self.load_zones_from_json()

    def change_camera_index(self, choice):
        self.camera_index = 0 if "1" in choice else 1
        if self.running:
            self.stop_camera()
            self.start_camera()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ZoneCalibrationApp(root)
    root.mainloop()