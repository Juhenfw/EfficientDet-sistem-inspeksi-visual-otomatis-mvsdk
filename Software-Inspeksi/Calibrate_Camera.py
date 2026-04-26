import cv2
import numpy as np
import mvsdk
import time
import platform
import threading
import customtkinter as ctk
from tkinter import messagebox
import json
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# =====================================================================
# PEMBUATAN FOLDER OTOMATIS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

class CameraCalibrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Calibration Tool - Undergraduate Thesis Juhen's Edition")
        
        # Responsive window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1400, int(screen_width * 0.9))
        window_height = min(850, int(screen_height * 0.9))
        
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(900, 600)
        
        # State Management
        self.hCamera = None
        self.pFrameBuffer = None
        self.running = False
        self.stop_event = threading.Event()
        
        self.dev_list = []
        self.cam_names = []
        self.current_cam_index = 0
        
        # Tambahan Setting RGB Gain (white_balance_mode: 0=Manual, 1=Auto, 2=D65)
        self.settings = {
            "exposure_time": 1000,
            "gamma": 20,
            "contrast": 120,
            "gain": 16.5,
            "white_balance_mode": 0,
            "r_gain": 100,  # 100 berarti 1.0x (normal)
            "g_gain": 100,
            "b_gain": 100
        }
        
        self.enumerate_cameras()
        self.build_gui()
        
        if self.dev_list:
            self.init_camera_device(0)
            self.load_settings()
        else:
            self.status_label.configure(text="No cameras connected. Please plug in a camera and click Refresh.")

    def get_config_path(self):
        return os.path.join(CONFIG_DIR, f"camera{self.current_cam_index + 1}_config.json")

    def enumerate_cameras(self):
        self.dev_list = mvsdk.CameraEnumerateDevice()
        self.cam_names = []
        for i, dev in enumerate(self.dev_list):
            try:
                name = dev.acFriendlyName.decode('utf-8')
            except:
                name = "Unknown Model"
            self.cam_names.append(f"Camera {i+1} ({name})")

    def refresh_camera_list(self):
        self.enumerate_cameras()
        if self.cam_names:
            self.cam_combo.configure(values=self.cam_names)
            if self.cam_combo.get() not in self.cam_names:
                self.cam_combo.set(self.cam_names[0])
                self.switch_camera(self.cam_names[0])
        else:
            self.cam_combo.configure(values=["No Camera Found"])
            self.cam_combo.set("No Camera Found")
        
        self.status_label.configure(text="Camera list refreshed.")

    def init_camera_device(self, index):
        try:
            DevInfo = self.dev_list[index]
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
            cap = mvsdk.CameraGetCapability(self.hCamera)
            
            modeMono = (cap.sIspCapacity.bMonoSensor != 0)
            if modeMono:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            mvsdk.CameraSetAeState(self.hCamera, 0)
            mvsdk.CameraPlay(self.hCamera)
            
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(
                cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if modeMono else 3), 16
            )
            print(f"[INFO] Camera {index + 1} initialized successfully")
            self.status_label.configure(text=f"Connected to Camera {index + 1}")
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to initialize Camera {index + 1}: {e}")
            self.hCamera = None

    def switch_camera(self, choice):
        if choice == "No Camera Found": return
        try:
            idx = self.cam_names.index(choice)
        except ValueError:
            return

        if idx == self.current_cam_index and self.hCamera is not None:
            return
            
        was_running = self.running
        self.stop_stream()
        time.sleep(0.3)
        
        if self.hCamera:
            try:
                mvsdk.CameraUnInit(self.hCamera)
                if self.pFrameBuffer:
                    mvsdk.CameraAlignFree(self.pFrameBuffer)
            except Exception as e:
                print(f"[WARNING] Error cleaning up old camera: {e}")
            self.hCamera = None
            self.pFrameBuffer = None
            
        self.current_cam_index = idx
        self.init_camera_device(idx)
        self.load_settings()
        
        if was_running and self.hCamera:
            self.start_stream()

    # ==================== GUI BUILDER ====================
    def build_gui(self):
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=1) 
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_columnconfigure(0, weight=1) 
        
        # Header
        header_frame = ctk.CTkFrame(self.root, height=60, corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.pack_propagate(False)
        ctk.CTkLabel(header_frame, text="Multi-Camera Calibration Tool", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=15)
        
        # Main Layout
        main_container = ctk.CTkFrame(self.root, corner_radius=0, fg_color="transparent")
        main_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1) 
        main_container.grid_columnconfigure(1, weight=0, minsize=400) # Diperlebar sedikit agar muat RGB
        
        # Left: Video Frame
        video_frame = ctk.CTkFrame(main_container, corner_radius=10)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        video_frame.grid_rowconfigure(1, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(video_frame, text="Camera Preview", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(10, 5))
        
        video_display_frame = ctk.CTkFrame(video_frame, fg_color="#1a1a1a", corner_radius=8)
        video_display_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=(5, 15))
        video_display_frame.pack_propagate(False) 
        
        self.video_label = ctk.CTkLabel(video_display_frame, text="")
        self.video_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Right: Controls Sidebar
        controls_container = ctk.CTkFrame(main_container, corner_radius=10, width=400)
        controls_container.grid(row=0, column=1, sticky="nsew")
        controls_container.grid_propagate(False) 
        
        controls_container.grid_rowconfigure(2, weight=1) 
        controls_container.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(controls_container, text="Camera Parameters", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, pady=(15, 5), sticky="ew")
        
        # Camera Selection
        cam_select_frame = ctk.CTkFrame(controls_container, fg_color="transparent")
        cam_select_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(cam_select_frame, text="Device:", font=ctk.CTkFont(size=12, weight="bold")).pack(side="left", padx=(5, 5))
        self.cam_combo = ctk.CTkComboBox(
            cam_select_frame, 
            values=self.cam_names if self.cam_names else ["No Camera Found"],
            command=self.switch_camera,
            width=200
        )
        self.cam_combo.pack(side="left", fill="x", expand=True)
        if self.cam_names:
            self.cam_combo.set(self.cam_names[0])
            
        btn_refresh = ctk.CTkButton(cam_select_frame, text="↻", width=30, fg_color="#f39c12", hover_color="#e67e22", command=self.refresh_camera_list)
        btn_refresh.pack(side="left", padx=(5, 5))

        # Scrollable Parameters Area
        scrollable_frame = ctk.CTkScrollableFrame(controls_container, corner_radius=8, fg_color="transparent")
        scrollable_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        scrollable_frame.grid_columnconfigure(0, weight=1)
        
        param_style = {"font": ctk.CTkFont(size=12, weight="bold")}
        
        # Exposure
        exp_frame = self._create_param_frame(scrollable_frame, 0)
        ctk.CTkLabel(exp_frame, text="Exposure Time (us)", **param_style).pack(anchor="w", pady=(0, 5))
        self.exposure_entry = ctk.CTkEntry(exp_frame, height=30)
        self.exposure_entry.pack(fill="x", pady=(0, 5))
        self.exposure_entry.bind("<Return>", lambda e: self.apply_exposure())
        self.exposure_slider = ctk.CTkSlider(exp_frame, from_=10, to=100000, command=self.update_exposure)
        self.exposure_slider.pack(fill="x", pady=(0, 5))
        
        # Gamma
        gamma_frame = self._create_param_frame(scrollable_frame, 1)
        ctk.CTkLabel(gamma_frame, text="Gamma", **param_style).pack(anchor="w", pady=(0, 5))
        self.gamma_entry = ctk.CTkEntry(gamma_frame, height=30)
        self.gamma_entry.pack(fill="x", pady=(0, 5))
        self.gamma_entry.bind("<Return>", lambda e: self.apply_gamma())
        self.gamma_slider = ctk.CTkSlider(gamma_frame, from_=0, to=250, command=self.update_gamma)
        self.gamma_slider.pack(fill="x", pady=(0, 5))
        
        # Contrast
        contrast_frame = self._create_param_frame(scrollable_frame, 2)
        ctk.CTkLabel(contrast_frame, text="Contrast", **param_style).pack(anchor="w", pady=(0, 5))
        self.contrast_entry = ctk.CTkEntry(contrast_frame, height=30)
        self.contrast_entry.pack(fill="x", pady=(0, 5))
        self.contrast_entry.bind("<Return>", lambda e: self.apply_contrast())
        self.contrast_slider = ctk.CTkSlider(contrast_frame, from_=0, to=200, command=self.update_contrast)
        self.contrast_slider.pack(fill="x", pady=(0, 5))
        
        # Gain
        gain_frame = self._create_param_frame(scrollable_frame, 3)
        ctk.CTkLabel(gain_frame, text="Analog Gain", **param_style).pack(anchor="w", pady=(0, 5))
        self.gain_entry = ctk.CTkEntry(gain_frame, height=30)
        self.gain_entry.pack(fill="x", pady=(0, 5))
        self.gain_entry.bind("<Return>", lambda e: self.apply_gain())
        self.gain_slider = ctk.CTkSlider(gain_frame, from_=2.5, to=16.5, command=self.update_gain)
        self.gain_slider.pack(fill="x", pady=(0, 5))
        
        # White Balance & RGB Panel
        wb_frame = self._create_param_frame(scrollable_frame, 4)
        ctk.CTkLabel(wb_frame, text="Color Correction (White Balance)", **param_style).pack(anchor="w", pady=(0, 5))
        
        wb_controls = ctk.CTkFrame(wb_frame, fg_color="transparent")
        wb_controls.pack(fill="x", pady=(0, 10))
        
        # ----- REVISI D65: Tambahkan Opsi di ComboBox -----
        self.wb_combo_param = ctk.CTkComboBox(wb_controls, values=["Manual RGB", "Auto", "Preset D65"], command=self.apply_wb_mode, height=32, width=120)
        self.wb_combo_param.pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(wb_controls, text="Auto Set Once", command=self.set_wb_once, height=32).pack(side="left", fill="x", expand=True)

        # RGB Sliders
        rgb_container = ctk.CTkFrame(wb_frame, fg_color="#1a1a1a", corner_radius=6)
        rgb_container.pack(fill="x", pady=(0, 5), ipadx=10, ipady=5)

        # Red Gain
        r_frame = ctk.CTkFrame(rgb_container, fg_color="transparent")
        r_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(r_frame, text="R-Gain", font=ctk.CTkFont(size=11, weight="bold"), text_color="#ff4d4d", width=50).pack(side="left")
        self.r_val = ctk.CTkLabel(r_frame, text="100", width=30)
        self.r_val.pack(side="right")
        self.r_slider = ctk.CTkSlider(r_frame, from_=0, to=400, button_color="#ff4d4d", command=lambda v: self.update_rgb('R', v))
        self.r_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Green Gain
        g_frame = ctk.CTkFrame(rgb_container, fg_color="transparent")
        g_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(g_frame, text="G-Gain", font=ctk.CTkFont(size=11, weight="bold"), text_color="#2ecc71", width=50).pack(side="left")
        self.g_val = ctk.CTkLabel(g_frame, text="100", width=30)
        self.g_val.pack(side="right")
        self.g_slider = ctk.CTkSlider(g_frame, from_=0, to=400, button_color="#2ecc71", command=lambda v: self.update_rgb('G', v))
        self.g_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Blue Gain
        b_frame = ctk.CTkFrame(rgb_container, fg_color="transparent")
        b_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(b_frame, text="B-Gain", font=ctk.CTkFont(size=11, weight="bold"), text_color="#3498db", width=50).pack(side="left")
        self.b_val = ctk.CTkLabel(b_frame, text="100", width=30)
        self.b_val.pack(side="right")
        self.b_slider = ctk.CTkSlider(b_frame, from_=0, to=400, button_color="#3498db", command=lambda v: self.update_rgb('B', v))
        self.b_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Action Buttons
        action_frame = ctk.CTkFrame(controls_container, fg_color="transparent")
        action_frame.grid(row=3, column=0, pady=15, padx=10, sticky="ew")
        action_frame.grid_columnconfigure((0, 1), weight=1)
        btn_style = {"height": 38, "font": ctk.CTkFont(size=13, weight="bold")}
        
        btn_row1 = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row1.grid(row=0, column=0, pady=(0, 5), sticky="ew")
        btn_row1.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btn_row1, text="Start Stream", command=self.start_stream, fg_color="#2ecc71", hover_color="#27ae60", **btn_style).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(btn_row1, text="Stop Stream", command=self.stop_stream, fg_color="#e74c3c", hover_color="#c0392b", **btn_style).grid(row=0, column=1, padx=(3, 0), sticky="ew")
        
        btn_row2 = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row2.grid(row=1, column=0, pady=(0, 5), sticky="ew")
        btn_row2.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(btn_row2, text="Save Config", command=self.save_settings, fg_color="#9b59b6", hover_color="#8e44ad", **btn_style).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(btn_row2, text="Load Config", command=self.load_settings, fg_color="#f39c12", hover_color="#d35400", **btn_style).grid(row=0, column=1, padx=(3, 0), sticky="ew")
        
        ctk.CTkButton(action_frame, text="Exit Application", command=self.close_app, fg_color="#95a5a6", hover_color="#7f8c8d", **btn_style).grid(row=2, column=0, sticky="ew")
        
        # Status Bar
        self.status_label = ctk.CTkLabel(self.root, text="System Ready", font=ctk.CTkFont(size=11), anchor="w", fg_color="#34495e", corner_radius=5, height=30)
        self.status_label.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

    def _create_param_frame(self, parent, row):
        frame = ctk.CTkFrame(parent, corner_radius=6, fg_color="#2b2b2b")
        frame.grid(row=row, column=0, sticky="ew", pady=5, padx=5)
        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=12, pady=10)
        return inner

    # ==================== SAVE / LOAD DENGAN RGB ====================
    def save_settings(self):
        try:
            file_path = self.get_config_path()
            with open(file_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            messagebox.showinfo("Success", f"Settings saved to:\n{os.path.basename(file_path)}")
            self.status_label.configure(text=f"Saved config: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def load_settings(self):
        file_path = self.get_config_path()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    self.settings.update(json.load(f))
                
                # Load Basic Params
                self.exposure_entry.delete(0, "end"); self.exposure_entry.insert(0, str(self.settings.get("exposure_time", 1000)))
                self.exposure_slider.set(self.settings.get("exposure_time", 1000))
                self.gamma_entry.delete(0, "end"); self.gamma_entry.insert(0, str(self.settings.get("gamma", 20)))
                self.gamma_slider.set(self.settings.get("gamma", 20))
                self.contrast_entry.delete(0, "end"); self.contrast_entry.insert(0, str(self.settings.get("contrast", 120)))
                self.contrast_slider.set(self.settings.get("contrast", 120))
                self.gain_entry.delete(0, "end"); self.gain_entry.insert(0, f"{self.settings.get('gain', 16.5):.3f}")
                self.gain_slider.set(self.settings.get("gain", 16.5))
                
                # ----- REVISI D65: Load Status WB Mode -----
                wb_mode = self.settings.get("white_balance_mode", 0)
                if wb_mode == 1:
                    self.wb_combo_param.set("Auto")
                elif wb_mode == 2:
                    self.wb_combo_param.set("Preset D65")
                else:
                    self.wb_combo_param.set("Manual RGB")
                
                self.r_slider.set(self.settings.get("r_gain", 100))
                self.g_slider.set(self.settings.get("g_gain", 100))
                self.b_slider.set(self.settings.get("b_gain", 100))
                self.r_val.configure(text=str(self.settings.get("r_gain", 100)))
                self.g_val.configure(text=str(self.settings.get("g_gain", 100)))
                self.b_val.configure(text=str(self.settings.get("b_gain", 100)))
                
                # Apply to Camera
                if self.hCamera:
                    mvsdk.CameraSetExposureTime(self.hCamera, self.settings.get("exposure_time", 1000))
                    mvsdk.CameraSetGamma(self.hCamera, self.settings.get("gamma", 20))
                    mvsdk.CameraSetContrast(self.hCamera, self.settings.get("contrast", 120))
                    mvsdk.CameraSetAnalogGain(self.hCamera, int(self.settings.get("gain", 16.5) / 0.125))
                    
                    # ----- REVISI D65: Terapkan ke Hardware saat Load -----
                    if wb_mode == 1:
                        mvsdk.CameraSetWbMode(self.hCamera, 1) # Mode Auto
                    elif wb_mode == 2:
                        mvsdk.CameraSetWbMode(self.hCamera, 0) 
                        if hasattr(mvsdk, 'CameraSetPresetColorTemperature'):
                            mvsdk.CameraSetPresetColorTemperature(self.hCamera, 0)
                        elif hasattr(mvsdk, 'CameraSetPresetColorTemp'):
                            mvsdk.CameraSetPresetColorTemp(self.hCamera, 0)
                        else:
                            mvsdk.CameraSetGain(self.hCamera, 150, 100, 180)
                    else:
                        mvsdk.CameraSetWbMode(self.hCamera, 0) # Mode Manual
                        self.apply_manual_rgb_to_camera()

                self.status_label.configure(text=f"Loaded config: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path}: {e}")
        else:
            print(f"[INFO] No config found at {file_path}")
            self.status_label.configure(text="No previous config, using defaults.")

    # ==================== PARAMETER UPDATES ====================
    def update_exposure(self, val):
        try:
            val = int(val)
            self.exposure_entry.delete(0, "end"); self.exposure_entry.insert(0, str(val))
            if self.hCamera: mvsdk.CameraSetExposureTime(self.hCamera, val)
            self.settings["exposure_time"] = val
        except Exception as e: print(f"Exposure error: {e}")
    def apply_exposure(self): self.update_exposure(self.exposure_entry.get())

    def update_gamma(self, val):
        try:
            val = int(val)
            self.gamma_entry.delete(0, "end"); self.gamma_entry.insert(0, str(val))
            if self.hCamera: mvsdk.CameraSetGamma(self.hCamera, val)
            self.settings["gamma"] = val
        except Exception as e: print(f"Gamma error: {e}")
    def apply_gamma(self): self.update_gamma(self.gamma_entry.get())

    def update_contrast(self, val):
        try:
            val = int(val)
            self.contrast_entry.delete(0, "end"); self.contrast_entry.insert(0, str(val))
            if self.hCamera: mvsdk.CameraSetContrast(self.hCamera, val)
            self.settings["contrast"] = val
        except Exception as e: print(f"Contrast error: {e}")
    def apply_contrast(self): self.update_contrast(self.contrast_entry.get())

    def update_gain(self, val):
        try:
            val = float(val)
            self.gain_entry.delete(0, "end"); self.gain_entry.insert(0, f"{val:.3f}")
            if self.hCamera: mvsdk.CameraSetAnalogGain(self.hCamera, int(val / 0.125))
            self.settings["gain"] = val
        except Exception as e: print(f"Gain error: {e}")
    def apply_gain(self): self.update_gain(self.gain_entry.get())

    # --- KONTROL RGB & WHITE BALANCE ---
    def apply_wb_mode(self, choice):
        try:
            if choice == "Auto":
                mode = 1
            elif choice == "Preset D65":
                mode = 2
            else:
                mode = 0
                
            if self.hCamera: 
                if mode == 1:
                    mvsdk.CameraSetWbMode(self.hCamera, 1)
                elif mode == 2:
                    mvsdk.CameraSetWbMode(self.hCamera, 0) # Wajib manual dulu
                    
                    # Coba cari nama fungsi yang tersedia di mvsdk.py kamu
                    if hasattr(mvsdk, 'CameraSetPresetColorTemperature'):
                        mvsdk.CameraSetPresetColorTemperature(self.hCamera, 0)
                    elif hasattr(mvsdk, 'CameraSetPresetColorTemp'):
                        mvsdk.CameraSetPresetColorTemp(self.hCamera, 0)
                    else:
                        # FALLBACK JIKA SDK PYTHON TIDAK MENDUKUNG PRESET:
                        # Set manual RGB Gain yang ekuivalen dengan D65 (6500K)
                        # (Nilai ini adalah aproksimasi standar sensor CMOS)
                        mvsdk.CameraSetGain(self.hCamera, 150, 100, 180)
                        print("[INFO] Menggunakan aproksimasi Manual D65 (R:150, G:100, B:180)")
                    
                    # Tarik hasil settingan D65 ke slider
                    self.root.after(500, self._pull_rgb_from_hardware)
                elif mode == 0:
                    mvsdk.CameraSetWbMode(self.hCamera, 0)
                    self.apply_manual_rgb_to_camera() 
                    
            self.settings["white_balance_mode"] = mode
        except Exception as e:
            print(f"WB mode error: {e}")

    def set_wb_once(self):
        """Memerintahkan kamera mencari warna putih, lalu tunda setengah detik sebelum membaca hasil"""
        if self.hCamera:
            try:
                # Set ke manual mode agar bisa diedit
                mvsdk.CameraSetWbMode(self.hCamera, 0)
                # Jalankan kalibrasi 1x tembak
                mvsdk.CameraSetOnceWB(self.hCamera)
                
                self.status_label.configure(text="Menghitung White Balance... (Mohon tunggu)")
                
                # PRO TIP: Beri waktu sensor kamera 500ms (0.5 detik) untuk menyesuaikan warna,
                # baru panggil fungsi untuk membaca nilainya ke slider.
                self.root.after(500, self._pull_rgb_from_hardware)
                
            except Exception as e:
                print(f"WB Set Once error: {e}")

    def _pull_rgb_from_hardware(self):
        """Fungsi internal untuk menarik data dari hardware setelah kamera selesai berpikir"""
        if self.hCamera:
            try:
                # mvsdk.CameraGetGain(hCamera) langsung me-return tuple (R, G, B)
                r_val, g_val, b_val = mvsdk.CameraGetGain(self.hCamera)
                
                # Update Slider dan Label Angka di UI
                self.r_slider.set(r_val)
                self.r_val.configure(text=str(r_val))
                
                self.g_slider.set(g_val)
                self.g_val.configure(text=str(g_val))
                
                self.b_slider.set(b_val)
                self.b_val.configure(text=str(b_val))
                
                # Update Dictionary Settings untuk di-save
                self.settings["r_gain"] = r_val
                self.settings["g_gain"] = g_val
                self.settings["b_gain"] = b_val
                
                # Status berhasil tarik data
                self.status_label.configure(text="Slider RGB telah diperbarui dari Hardware.")
                
            except Exception as e:
                print(f"Gagal membaca nilai RGB dari kamera: {e}")

    def update_rgb(self, channel, val):
        """Handler saat slider R, G, atau B digeser manual"""
        val = int(val)
        
        # ----- REVISI D65: Pindah ke mode manual jika user menggeser slider saat di Auto / D65 -----
        if self.settings["white_balance_mode"] in [1, 2]:
            self.wb_combo_param.set("Manual RGB")
            self.apply_wb_mode("Manual RGB")

        if channel == 'R':
            self.r_val.configure(text=str(val))
            self.settings["r_gain"] = val
        elif channel == 'G':
            self.g_val.configure(text=str(val))
            self.settings["g_gain"] = val
        elif channel == 'B':
            self.b_val.configure(text=str(val))
            self.settings["b_gain"] = val
            
        self.apply_manual_rgb_to_camera()

    def apply_manual_rgb_to_camera(self):
        """Kirim 3 nilai RGB ke hardware"""
        if self.hCamera and self.settings["white_balance_mode"] == 0:
            try:
                mvsdk.CameraSetGain(
                    self.hCamera, 
                    int(self.settings["r_gain"]), 
                    int(self.settings["g_gain"]), 
                    int(self.settings["b_gain"])
                )
            except Exception as e:
                print(f"Gagal mengirim RGB: {e}")

    # ==================== VIDEO STREAM ====================
    def start_stream(self):
        if not self.hCamera: return
        if not self.running:
            self.running = True
            self.stop_event.clear()
            threading.Thread(target=self.update_frame, daemon=True).start()
            self.status_label.configure(text="Camera streaming...")

    def stop_stream(self):
        self.running = False
        self.stop_event.set()
        self.status_label.configure(text="Camera stopped")

    def update_frame(self):
        while self.running and not self.stop_event.is_set():
            if not self.hCamera: break
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
                mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
                
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)
                
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                    (FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3)
                )
                
                label_width = self.video_label.winfo_width()
                label_height = self.video_label.winfo_height()
                if label_width > 10 and label_height > 10:
                    h, w = frame.shape[:2]
                    aspect = w / h
                    if label_width / label_height > aspect:
                        new_h = label_height
                        new_w = int(new_h * aspect)
                    else:
                        new_w = label_width
                        new_h = int(new_w / aspect)
                    new_w, new_h = max(1, new_w), max(1, new_h)
                    frame = cv2.resize(frame, (new_w, new_h))
                else:
                    new_w, new_h = 960, 720
                    frame = cv2.resize(frame, (new_w, new_h))
                
                # Text overlay
                cv2.putText(frame, f"Exp: {self.exposure_entry.get()}us | RGB: {self.settings['r_gain']},{self.settings['g_gain']},{self.settings['b_gain']}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 1. Konversi BGR ke RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Gunakan CTkImage untuk menghilangkan warning HighDPI
                from PIL import Image
                pil_img = Image.fromarray(img_rgb)
                photo = ctk.CTkImage(light_image=pil_img, size=(new_w, new_h))
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT: pass
            except Exception: pass

    def close_app(self):
        self.stop_stream()
        time.sleep(0.3)
        if self.hCamera:
            try:
                mvsdk.CameraUnInit(self.hCamera)
                if self.pFrameBuffer: mvsdk.CameraAlignFree(self.pFrameBuffer)
            except: pass
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = CameraCalibrationApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()