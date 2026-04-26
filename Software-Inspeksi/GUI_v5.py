import tkinter as tk
import customtkinter as ctk
from datetime import datetime
import sys
import ctypes
from PIL import Image, ImageTk
# import cv2
# import numpy as np

# Setup customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Skema warna
STATUS_COLORS = {
    "pass": "#2ECC71",
    "fail": "#E74C3C",
    "warning": "#F39C12",
    "info": "#3498DB",
    "bg_dark": "#1a1a1a",
    "bg_card": "#2b2b2b",
    "accent": "#3498DB"
}

# Fungsi Helper
def get_scaled_size(base_size, screen_width, screen_height, base_width=1920, base_height=1080):
    scale_x = screen_width / base_width
    scale_y = screen_height / base_height
    scale = min(scale_x, scale_y)
    return max(1, int(base_size * scale))

def get_scaled_font(base_family, base_size, weight="normal", screen_width=1920, screen_height=1080):
    scaled_size = get_scaled_size(base_size, screen_width, screen_height)
    font_multiplier = 1.5
    final_size = max(14, int(scaled_size * font_multiplier))
    return (base_family, final_size, weight)

def show_taskbar_windows():
    if sys.platform == "win32":
        SW_SHOW = 5
        hwnd = ctypes.windll.user32.FindWindowW("Shell_TrayWnd", None)
        ctypes.windll.user32.ShowWindow(hwnd, SW_SHOW)
        start_hwnd = ctypes.windll.user32.FindWindowW("Button", None)
        if start_hwnd:
            ctypes.windll.user32.ShowWindow(start_hwnd, SW_SHOW)


class InspectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Inspeksi Visual Otomatis - QC Pianika")
        
        # Variabel Default
        self.total_inspections = 0
        self.passed_today = 0
        self.failed_today = 0
        self.current_status = "Siap"
        
        # Setup Layar
        self.setup_screen()
        
        # Bangun Komponen UI
        self.build_header()
        self.build_main_layout()
        self.build_footer()
        
        # Binding
        self.root.bind('<Escape>', lambda e: self.exit_program())
        self.root.protocol("WM_DELETE_WINDOW", self.exit_program)
        self.root.focus_force()

    def setup_screen(self):
        self.root.update_idletasks()
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        width = int(self.screen_width * 0.8) # Sedikit diperlebar agar 3 kolom muat
        height = int(self.screen_height * 0.75)
        self.root.geometry(f"{width}x{height}+50+50")

        if sys.platform == "win32":
            self.root.state('normal') 
            self.root.overrideredirect(False) 
            self.root.wm_attributes("-topmost", False) 
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        
        self.root.configure(fg_color=STATUS_COLORS["bg_dark"])

    def build_header(self):
        header_height = get_scaled_size(95, self.screen_width, self.screen_height)
        header_frame = ctk.CTkFrame(self.root, height=header_height, corner_radius=0, fg_color=STATUS_COLORS["bg_dark"])
        header_frame.pack(fill="x", padx=0, pady=0)
        header_frame.pack_propagate(False)

        title_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_container.pack(side="left", padx=get_scaled_size(30, self.screen_width, self.screen_height), pady=get_scaled_size(15, self.screen_width, self.screen_height))
        main_title_font = get_scaled_font("Segoe UI", 22, "bold", self.screen_width, self.screen_height)
        main_title = ctk.CTkLabel(title_container, text="SISTEM INSPEKSI VISUAL OTOMATIS", font=main_title_font, text_color="#FFFFFF")
        main_title.pack(anchor="w")
        subtitle_font = get_scaled_font("Segoe UI", 8, "normal", self.screen_width, self.screen_height)
        subtitle = ctk.CTkLabel(title_container, text="Pemeriksaan Kelengkapan - Pianika Production Line", font=subtitle_font, text_color="#95A5A6")
        subtitle.pack(anchor="w")

        datetime_container = ctk.CTkFrame(header_frame, fg_color="#2b2b2b", corner_radius=get_scaled_size(12, self.screen_width, self.screen_height), border_width=1, border_color="#3498DB")
        datetime_container.pack(side="right", padx=get_scaled_size(30, self.screen_width, self.screen_height))
        datetime_inner = ctk.CTkFrame(datetime_container, fg_color="transparent")
        datetime_inner.pack(padx=get_scaled_size(20, self.screen_width, self.screen_height), pady=get_scaled_size(12, self.screen_width, self.screen_height))

        time_font = get_scaled_font("Segoe UI", 20, "bold", self.screen_width, self.screen_height)
        self.time_label = ctk.CTkLabel(datetime_inner, text="", font=time_font, text_color="#3498DB", anchor="center")
        self.time_label.pack()

        divider = ctk.CTkFrame(datetime_inner, height=1, fg_color="#3498DB")
        divider.pack(fill="x", pady=get_scaled_size(6, self.screen_width, self.screen_height))

        date_font = get_scaled_font("Segoe UI", 12, "normal", self.screen_width, self.screen_height)
        self.date_label = ctk.CTkLabel(datetime_inner, text="", font=date_font, text_color="#95A5A6", anchor="center")
        self.date_label.pack()

        self.update_datetime()

    def update_datetime(self):
        now = datetime.now()
        self.time_label.configure(text=now.strftime("%H:%M:%S"))
        self.date_label.configure(text=now.strftime("%A, %d %B %Y"))
        self.root.after(1000, self.update_datetime)

    # ==========================================
    # MAIN LAYOUT (3 KOLOM)
    # ==========================================
    def build_main_layout(self):
        main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=0)

        # --- REVISI 1: Perlebar Kolom Kiri ---
        # Awalnya 0.22 (22%), kita naikkan menjadi 0.26 (26%) atau 0.28 (28%)
        left_width = int(self.screen_width * 0.28) 
        left_column = ctk.CTkFrame(main_container, fg_color="transparent", width=left_width)
        left_column.pack(side="left", fill="y", padx=(0, get_scaled_size(10, self.screen_width, self.screen_height)))
        left_column.pack_propagate(False)

        self.build_operator_card(left_column)
        self.build_model_card(left_column)
        self.build_guide_card(left_column)
        self.build_stats_card(left_column)

        # --- REVISI 2: Persempit Kolom Kanan ---
        # Awalnya 0.25 (25%), kita turunkan menjadi 0.20 (20%) atau 0.18 (18%)
        right_width = int(self.screen_width * 0.20) 
        right_column = ctk.CTkFrame(main_container, fg_color="transparent", width=right_width)
        right_column.pack(side="right", fill="y", padx=(get_scaled_size(10, self.screen_width, self.screen_height), 0))
        right_column.pack_propagate(False)

        self.build_component_card(right_column)

        # 2. KOLOM TENGAH (Kamera & Kontrol Utama) - Sisa space expand=True
        mid_column = ctk.CTkFrame(main_container, fg_color="transparent")
        mid_column.pack(side="left", fill="both", expand=True)

        self.build_camera_card(mid_column)
        self.build_control_card(mid_column)

    # ==========================================
    # ISI KOLOM KIRI
    # ==========================================
    def build_operator_card(self, parent):
        op_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"])
        op_card.pack(fill="x", pady=(0, get_scaled_size(10, self.screen_width, self.screen_height)))

        header_font = get_scaled_font("Segoe UI", 12, "bold", self.screen_width, self.screen_height)
        data_font = get_scaled_font("Segoe UI", 12, "normal", self.screen_width, self.screen_height)
        label_font = get_scaled_font("Segoe UI", 10, "bold", self.screen_width, self.screen_height)

        ctk.CTkLabel(op_card, text="OPERATOR AKTIF", font=header_font, anchor="w", text_color="#95A5A6").pack(padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(get_scaled_size(15, self.screen_width, self.screen_height), 0), anchor="w")
        
        # --- CONTAINER UNTUK ENTRY & TOMBOL OK ---
        entry_container = ctk.CTkFrame(op_card, fg_color="transparent")
        entry_container.pack(
            fill="x", 
            padx=get_scaled_size(15, self.screen_width, self.screen_height), 
            pady=(get_scaled_size(10, self.screen_width, self.screen_height), get_scaled_size(10, self.screen_width, self.screen_height))
        )

        # 1. Entry No Tag (Sebelah Kiri)
        self.entry_notag = ctk.CTkEntry(
            entry_container, 
            placeholder_text="Scan No. Tag di sini...", 
            font=data_font, 
            height=get_scaled_size(35, self.screen_width, self.screen_height)
        )
        # expand=True agar kolom entry mengambil sisa ruang yang ada
        self.entry_notag.pack(side="left", fill="x", expand=True, padx=(0, get_scaled_size(10, self.screen_width, self.screen_height)))

        # 2. Tombol OK (Sebelah Kanan)
        btn_ok_font = get_scaled_font("Segoe UI", 12, "bold", self.screen_width, self.screen_height)
        self.btn_ok_tag = ctk.CTkButton(
            entry_container, 
            text="OK", 
            width=get_scaled_size(50, self.screen_width, self.screen_height), # Lebar tombol dibuat kecil saja
            height=get_scaled_size(35, self.screen_width, self.screen_height), # Samakan tinggi dengan entry
            font=btn_ok_font,
            fg_color=STATUS_COLORS["accent"], 
            hover_color="#2980B9",
            # command=self.process_tag_input # Nanti arahkan ke method untuk memproses tag
        )
        self.btn_ok_tag.pack(side="right")

        # --- CONTAINER BALOK INFO ---
        info_container = ctk.CTkFrame(op_card, fg_color="transparent")
        info_container.pack(fill="x", padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(0, get_scaled_size(15, self.screen_width, self.screen_height)))

        # BALOK 1: NIK
        nik_frame = ctk.CTkFrame(info_container, fg_color="#1e1e1e", corner_radius=8)
        nik_frame.pack(side="left", fill="x", expand=True, padx=(0, get_scaled_size(5, self.screen_width, self.screen_height)))
        ctk.CTkLabel(nik_frame, text="NIK", font=label_font, text_color="#7F8C8D").pack(pady=(get_scaled_size(5, self.screen_width, self.screen_height), 0))
        self.lbl_nik = ctk.CTkLabel(nik_frame, text="-", font=get_scaled_font("Segoe UI", 14, "bold", self.screen_width, self.screen_height), text_color="white")
        self.lbl_nik.pack(pady=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

        # BALOK 2: NAMA
        nama_frame = ctk.CTkFrame(info_container, fg_color="#1e1e1e", corner_radius=8)
        nama_frame.pack(side="left", fill="x", expand=True, padx=(get_scaled_size(5, self.screen_width, self.screen_height), 0))
        ctk.CTkLabel(nama_frame, text="NAMA", font=label_font, text_color="#7F8C8D").pack(pady=(get_scaled_size(5, self.screen_width, self.screen_height), 0))
        self.lbl_nama = ctk.CTkLabel(nama_frame, text="-", font=get_scaled_font("Segoe UI", 14, "bold", self.screen_width, self.screen_height), text_color=STATUS_COLORS["info"])
        self.lbl_nama.pack(pady=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

    def build_model_card(self, parent):
        model_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"])
        model_card.pack(fill="x", pady=(0, get_scaled_size(10, self.screen_width, self.screen_height)))

        model_header_font = get_scaled_font("Segoe UI", 12, "bold", self.screen_width, self.screen_height)
        ctk.CTkLabel(model_card, text="PILIH MODEL PIANIKA", font=model_header_font, anchor="w").pack(padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(get_scaled_size(15, self.screen_width, self.screen_height), get_scaled_size(5, self.screen_width, self.screen_height)), anchor="w")

        self.models = ["Pianika Model P32E (Biru)", "Pianika Model P32EP (Pink)"]
        model_dropdown_font = get_scaled_font("Segoe UI", 12, "normal", self.screen_width, self.screen_height)
        
        self.model_dropdown = ctk.CTkOptionMenu(
            model_card, values=self.models, height=get_scaled_size(35, self.screen_width, self.screen_height),
            font=model_dropdown_font, dropdown_font=model_dropdown_font, corner_radius=get_scaled_size(8, self.screen_width, self.screen_height)
        )
        self.model_dropdown.set(self.models[0])
        self.model_dropdown.pack(fill="x", padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(0, get_scaled_size(15, self.screen_width, self.screen_height)))

    def build_guide_card(self, parent):
        guide_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"])
        guide_card.pack(fill="both", expand=True, pady=(0, get_scaled_size(10, self.screen_width, self.screen_height)))

        guide_header_font = get_scaled_font("Segoe UI", 12, "bold", self.screen_width, self.screen_height)
        ctk.CTkLabel(guide_card, text="GAMBAR PANDUAN", font=guide_header_font, anchor="w").pack(padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(get_scaled_size(10, self.screen_width, self.screen_height), get_scaled_size(5, self.screen_width, self.screen_height)), anchor="w")

        self.guide_canvas_frame = ctk.CTkFrame(guide_card, fg_color="#1a1a1a", corner_radius=get_scaled_size(8, self.screen_width, self.screen_height))
        self.guide_canvas_frame.pack(fill="both", expand=True, padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(0, get_scaled_size(15, self.screen_width, self.screen_height)))
        
        # Label penampung gambar panduan
        self.lbl_guide_img = ctk.CTkLabel(self.guide_canvas_frame, text="Belum ada gambar\npanduan", text_color="#7F8C8D")
        self.lbl_guide_img.pack(fill="both", expand=True)

        self.current_guide_path = None 
        self.guide_canvas_frame.bind("<Configure>", self.resize_guide_image)

    def build_stats_card(self, parent):
        stats_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"])
        stats_card.pack(fill="x", pady=(0, 0))

        stats_title_font = get_scaled_font("Segoe UI", 12, "bold", self.screen_width, self.screen_height)
        ctk.CTkLabel(stats_card, text="STATISTIK HARI INI", font=stats_title_font, anchor="w").pack(padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(get_scaled_size(15, self.screen_width, self.screen_height), get_scaled_size(10, self.screen_width, self.screen_height)), anchor="w")

        stats_container = ctk.CTkFrame(stats_card, fg_color="transparent")
        stats_container.pack(fill="x", padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(0, get_scaled_size(15, self.screen_width, self.screen_height)))

        stat_num_font = get_scaled_font("Segoe UI", 16, "bold", self.screen_width, self.screen_height)
        stat_label_font = get_scaled_font("Segoe UI", 10, "normal", self.screen_width, self.screen_height)

        # Layout kotak kecil agar muat di panel sempit
        stat1 = ctk.CTkFrame(stats_container, fg_color="#1e1e1e", corner_radius=get_scaled_size(8, self.screen_width, self.screen_height))
        stat1.pack(side="left", fill="x", expand=True, padx=(0, get_scaled_size(4, self.screen_width, self.screen_height)))
        self.lbl_stat_total = ctk.CTkLabel(stat1, text="0", font=stat_num_font, text_color=STATUS_COLORS["info"])
        self.lbl_stat_total.pack(pady=(get_scaled_size(5, self.screen_width, self.screen_height), 0))
        ctk.CTkLabel(stat1, text="Total", font=stat_label_font, text_color="#95A5A6").pack(pady=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

        stat2 = ctk.CTkFrame(stats_container, fg_color="#1e1e1e", corner_radius=get_scaled_size(8, self.screen_width, self.screen_height))
        stat2.pack(side="left", fill="x", expand=True, padx=(0, get_scaled_size(4, self.screen_width, self.screen_height)))
        self.lbl_stat_pass = ctk.CTkLabel(stat2, text="0", font=stat_num_font, text_color=STATUS_COLORS["pass"])
        self.lbl_stat_pass.pack(pady=(get_scaled_size(5, self.screen_width, self.screen_height), 0))
        ctk.CTkLabel(stat2, text="Lolos", font=stat_label_font, text_color="#95A5A6").pack(pady=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

        stat3 = ctk.CTkFrame(stats_container, fg_color="#1e1e1e", corner_radius=get_scaled_size(8, self.screen_width, self.screen_height))
        stat3.pack(side="left", fill="x", expand=True, padx=(0, 0))
        self.lbl_stat_fail = ctk.CTkLabel(stat3, text="0", font=stat_num_font, text_color=STATUS_COLORS["fail"])
        self.lbl_stat_fail.pack(pady=(get_scaled_size(5, self.screen_width, self.screen_height), 0))
        ctk.CTkLabel(stat3, text="Gagal", font=stat_label_font, text_color="#95A5A6").pack(pady=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

    # ==========================================
    # ISI KOLOM KANAN (HASIL DETEKSI)
    # ==========================================
    def build_component_card(self, parent):
        component_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"])
        component_card.pack(fill="both", expand=True)

        comp_header_frame = ctk.CTkFrame(component_card, fg_color="transparent")
        comp_header_frame.pack(fill="x", padx=get_scaled_size(20, self.screen_width, self.screen_height), pady=(get_scaled_size(20, self.screen_width, self.screen_height), get_scaled_size(15, self.screen_width, self.screen_height)))

        comp_title_font = get_scaled_font("Segoe UI", 16, "bold", self.screen_width, self.screen_height)
        ctk.CTkLabel(comp_header_frame, text="HASIL DETEKSI", font=comp_title_font, anchor="w").pack(side="left")
        
        comp_count_font = get_scaled_font("Segoe UI", 13, "normal", self.screen_width, self.screen_height)
        self.comp_count_label = ctk.CTkLabel(comp_header_frame, text="0/0", font=comp_count_font, text_color=STATUS_COLORS["info"])
        self.comp_count_label.pack(side="right")

        self.comp_scroll = ctk.CTkScrollableFrame(component_card, fg_color="transparent")
        self.comp_scroll.pack(fill="both", expand=True, padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=(0, get_scaled_size(15, self.screen_width, self.screen_height)))

    # ==========================================
    # ISI KOLOM TENGAH (KAMERA & KONTROL)
    # ==========================================
    def build_camera_card(self, parent):
        camera_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"])
        camera_card.pack(fill="both", expand=True, pady=(0, get_scaled_size(10, self.screen_width, self.screen_height)))

        cam_header_height = get_scaled_size(50, self.screen_width, self.screen_height)
        cam_header = ctk.CTkFrame(camera_card, fg_color="transparent", height=cam_header_height)
        cam_header.pack(fill="x", padx=get_scaled_size(20, self.screen_width, self.screen_height), pady=(get_scaled_size(15, self.screen_width, self.screen_height), get_scaled_size(5, self.screen_width, self.screen_height)))
        cam_header.pack_propagate(False)
        cam_title_font = get_scaled_font("Segoe UI", 16, "bold", self.screen_width, self.screen_height)
        ctk.CTkLabel(cam_header, text="LIVE CAMERA FEED", font=cam_title_font, anchor="w").pack(side="left")

        status_indicator_size = get_scaled_size(30, self.screen_width, self.screen_height)
        self.cam_status_frame = ctk.CTkFrame(cam_header, fg_color=STATUS_COLORS["pass"], corner_radius=status_indicator_size // 2, height=status_indicator_size)
        self.cam_status_frame.pack(side="right")
        status_online_font = get_scaled_font("Segoe UI", 11, "bold", self.screen_width, self.screen_height)
        self.lbl_cam_status = ctk.CTkLabel(self.cam_status_frame, text="● ONLINE", font=status_online_font, text_color="white")
        self.lbl_cam_status.pack(padx=get_scaled_size(15, self.screen_width, self.screen_height), pady=get_scaled_size(5, self.screen_width, self.screen_height))

        self.camera_container = ctk.CTkFrame(camera_card, fg_color="#0a0a0a", corner_radius=get_scaled_size(12, self.screen_width, self.screen_height))
        self.camera_container.pack(fill="both", expand=True, padx=get_scaled_size(20, self.screen_width, self.screen_height), pady=(0, get_scaled_size(20, self.screen_width, self.screen_height)))

        self.camera_canvas = tk.Canvas(self.camera_container, bg="#1a1a1a", highlightthickness=get_scaled_size(2, self.screen_width, self.screen_height), highlightbackground=STATUS_COLORS["accent"])
        self.camera_canvas.pack(fill="both", expand=True, padx=get_scaled_size(3, self.screen_width, self.screen_height), pady=get_scaled_size(3, self.screen_width, self.screen_height))
        self.camera_canvas.bind("<Configure>", self.draw_camera_placeholder)

    def draw_camera_placeholder(self, event=None):
        self.camera_canvas.delete("all")
        width = self.camera_canvas.winfo_width()
        height = self.camera_canvas.winfo_height()
        if width > 1 and height > 1:
            center_font = get_scaled_font("Segoe UI", 16, "normal", self.screen_width, self.screen_height)
            self.camera_canvas.create_text(width / 2, height / 2, text="📷\n\nMenunggu Koneksi Kamera...", font=center_font, fill="#7F8C8D", justify="center")

    def build_control_card(self, parent):
        control_height = get_scaled_size(140, self.screen_width, self.screen_height)
        control_card = ctk.CTkFrame(parent, corner_radius=get_scaled_size(15, self.screen_width, self.screen_height), fg_color=STATUS_COLORS["bg_card"], height=control_height)
        control_card.pack(fill="x")
        control_card.pack_propagate(False)

        # Status Label di tengah atas kontrol
        status_frame_height = get_scaled_size(40, self.screen_width, self.screen_height)
        status_frame = ctk.CTkFrame(control_card, fg_color="#1e1e1e", corner_radius=get_scaled_size(10, self.screen_width, self.screen_height), height=status_frame_height)
        status_frame.pack(fill="x", padx=get_scaled_size(20, self.screen_width, self.screen_height), pady=(get_scaled_size(15, self.screen_width, self.screen_height), get_scaled_size(10, self.screen_width, self.screen_height)))
        status_frame.pack_propagate(False)

        status_label_font = get_scaled_font("Segoe UI", 12, "normal", self.screen_width, self.screen_height)
        ctk.CTkLabel(status_frame, text="Status Sistem:", font=status_label_font, anchor="w").pack(side="left", padx=get_scaled_size(15, self.screen_width, self.screen_height))

        status_value_font = get_scaled_font("Segoe UI", 13, "bold", self.screen_width, self.screen_height)
        self.status_label = ctk.CTkLabel(status_frame, text=self.current_status, font=status_value_font, text_color=STATUS_COLORS["pass"], fg_color="#1e1e1e")
        self.status_label.pack(side="right", padx=get_scaled_size(15, self.screen_width, self.screen_height))

        # Kontainer Tombol (Perubahan Sesuai Permintaan)
        button_container = ctk.CTkFrame(control_card, fg_color="transparent")
        button_container.pack(fill="both", expand=True, padx=get_scaled_size(20, self.screen_width, self.screen_height), pady=(0, get_scaled_size(15, self.screen_width, self.screen_height)))

        btn_height = get_scaled_size(50, self.screen_width, self.screen_height)
        btn_font = get_scaled_font("Segoe UI", 13, "bold", self.screen_width, self.screen_height)
        corner_radius_btn = get_scaled_size(10, self.screen_width, self.screen_height)

        # 1. Tombol Mulai Inspeksi (Dulu: Ambil Gambar)
        self.btn_inspect = ctk.CTkButton(button_container, text="MULAI INSPEKSI", height=btn_height, font=btn_font, fg_color=STATUS_COLORS["accent"], hover_color="#2980B9", corner_radius=corner_radius_btn)
        self.btn_inspect.pack(side="left", fill="both", expand=True, padx=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

        # 2. Tombol Simpan Hasil (Dulu: Mulai Inspeksi)
        self.btn_save = ctk.CTkButton(button_container, text="SIMPAN HASIL", height=btn_height, font=btn_font, fg_color=STATUS_COLORS["pass"], hover_color="#27AE60", corner_radius=corner_radius_btn)
        self.btn_save.pack(side="left", fill="both", expand=True, padx=(0, get_scaled_size(5, self.screen_width, self.screen_height)))

        # 3. Tombol Reset
        self.btn_reset = ctk.CTkButton(button_container, text="RESET", height=btn_height, font=btn_font, fg_color="#7F8C8D", hover_color="#5D6D7E", corner_radius=corner_radius_btn, width=get_scaled_size(120, self.screen_width, self.screen_height))
        self.btn_reset.pack(side="left", padx=(0, 0))

    def build_footer(self):
        footer_height = get_scaled_size(40, self.screen_width, self.screen_height)
        footer_frame = ctk.CTkFrame(self.root, height=footer_height, corner_radius=0, fg_color=STATUS_COLORS["bg_dark"])
        footer_frame.pack(fill="x", side="bottom")
        footer_frame.pack_propagate(False)

        footer_left_font = get_scaled_font("Segoe UI", 10, "normal", self.screen_width, self.screen_height)
        ctk.CTkLabel(footer_frame, text="© 2026 FTMM UNAIR | Version 1.2.8", font=footer_left_font, text_color="#7F8C8D").pack(side="left", padx=get_scaled_size(30, self.screen_width, self.screen_height))

        exit_btn_height = get_scaled_size(28, self.screen_width, self.screen_height)
        exit_btn_width = get_scaled_size(100, self.screen_width, self.screen_height)
        exit_btn_font = get_scaled_font("Segoe UI", 11, "bold", self.screen_width, self.screen_height)
        self.exit_btn = ctk.CTkButton(
            footer_frame, text="KELUAR", command=self.exit_program,
            height=exit_btn_height, width=exit_btn_width, font=exit_btn_font,
            fg_color=STATUS_COLORS["fail"], hover_color="#C0392B",
            corner_radius=get_scaled_size(6, self.screen_width, self.screen_height)
        )
        self.exit_btn.pack(side="right", padx=get_scaled_size(30, self.screen_width, self.screen_height), pady=get_scaled_size(5, self.screen_width, self.screen_height))

    # ==========================================
    # METHOD BARU UNTUK MAIN SYSTEM
    # ==========================================
    def set_operator_info(self, nik, nama):
        """Method untuk dipanggil dari main_system saat membaca JSON operator"""
        self.lbl_nik.configure(text=nik)
        
        # Jika nama terlalu panjang, kita potong agar balok tidak pecah
        if len(nama) > 12:
            nama_singkat = nama[:10] + ".."
        else:
            nama_singkat = nama
            
        self.lbl_nama.configure(text=nama_singkat)

    def set_guide_image(self, image_path):
        """Method untuk menetapkan gambar panduan dengan Smart Auto-Retry"""
        self.current_guide_path = image_path
        
        # Ambil dimensi frame saat ini
        self.root.update_idletasks() # Paksa Tkinter update geometri secara sinkron
        w = self.guide_canvas_frame.winfo_width()
        h = self.guide_canvas_frame.winfo_height()
        
        # Jika frame masih terlalu kecil (berarti layout pack/grid belum selesai merentang)
        # Kita panggil ulang fungsi ini 50ms kemudian
        if w < 100 or h < 100:
            self.root.after(50, lambda: self.set_guide_image(image_path))
            return
            
        # Jika ukurannya sudah wajar (sudah merentang), baru kita resize gambarnya!
        self.resize_guide_image()

    def resize_guide_image(self, event=None):
        """Method dinamis untuk menyesuaikan gambar dengan ukuran frame"""
        if not hasattr(self, 'current_guide_path') or not self.current_guide_path:
            return

        try:
            # Gunakan ukuran dari event jika dipicu oleh <Configure>, 
            # atau gunakan winfo jika dipicu manual
            if event and hasattr(event, 'width') and event.width > 10:
                w, h = event.width, event.height
            else:
                w = self.guide_canvas_frame.winfo_width()
                h = self.guide_canvas_frame.winfo_height()

            if w > 100 and h > 100: # Pastikan ukurannya benar-benar solid
                img = Image.open(self.current_guide_path)
                
                # Resize image to fit box maintaining aspect ratio
                img.thumbnail((w - 10, h - 10), Image.Resampling.LANCZOS)
                
                # Simpan ke self agar tidak dihapus oleh Garbage Collector
                self.guide_ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
                
                self.lbl_guide_img.configure(image=self.guide_ctk_img, text="")
        except Exception as e:
            print(f"[UI WARN] Gagal meload/resize gambar panduan: {e}")
            self.lbl_guide_img.configure(text="Gambar tidak\nditemukan", image="")

    def update_status(self, text, color):
        try:
            self.status_label.configure(text=text, text_color=color)
            self.status_label.update() 
        except Exception as e:
            print(f"Gagal update status: {e}")

    def exit_program(self):
        show_taskbar_windows()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = InspectionUI(root)
    # Testing fungsi baru
    app.set_operator_info("1012304", "Juhen FW")
    root.mainloop()