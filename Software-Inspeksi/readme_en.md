*Baca dalam bahasa lain: [Bahasa Indonesia](readme.md)*

# Automated Visual Inspection Software (Pianika QC)

This directory contains the operational software suite for the **Automated Visual Inspection System** on the Pianika production line. The software integrates image acquisition from industrial cameras, real-time AI inference using EfficientDet, and production data management.

This system is divided into four main modules functioning for hardware calibration, spatial logic configuration, and inspection operations across two different workstations.

---

## Software Modules

### 1. Camera Calibration Tool ([Calibrate_Camera.py](./Calibrate_Camera.py))
This instrument is used to adjust the optical parameters of the MindVision industrial camera to obtain optimal image quality for the AI model.
* **Key Features**: Exposure time, gamma, contrast, analog gain, and White Balance settings (Manual RGB, Auto, or D65 Preset).
* **Output**: Saves configurations to the `configs/` folder as a reference for the main system.

### 2. Zone Configuration Tool ([Calibrate_Zone.py](./Calibrate_Zone.py))
A tool to establish the "Ideal Zone" or *Region of Interest* (ROI) for each component to be detected so the system can verify object positions accurately.
* **Key Features**: Automated AI inference to detect reference object positions and save them as spatial coordinate boundaries.
* **Function**: Ensures components are not only present but also correctly positioned according to QC standards.

### 3. Station 1 Inspection System ([main_system_station1.py](./main_system_station1.py))
Operational software for the **Accessories Station**.
* **Detection Targets**: Label, hose, mouthpiece, leaflet, and manual book.
* **Features**: Operator management via Tag No. scanning, daily production log recording, and a smart queuing system (IPC).

### 4. Station 2 Inspection System ([main_system_station2.py](./main_system_station2.py))
Operational software for the **Main Unit Station**.
* **Detection Targets**: Pianika Unit (Blue/Pink) and Bag/Case (Blue/Pink).
* **Features**: Queue data synchronization from Station 1 and verification of main unit completeness before final packaging.

---

## Supporting Architecture

* **[GUI_v5.py](./GUI_v5.py)**: The core user interface library based on *CustomTkinter*, providing a responsive layout and production status visualization (OK/Not Good (NG)).
* **[mvsdk.py](./mvsdk.py)**: Python SDK driver for direct integration with MindVision industrial cameras.

---

## Operational Directory Structure

The system expects the following folder structure to run stably:

```text
.
├── configs/                 # Calibration result files (.json)
├── models/                  # AI weight files (best_loss_d1.pth)
├── assets/                  # Operator guide images (.jpg)
├── hasil_inspeksi/          # Production image record database
├── projects/                # Object class definitions (pianika.yml)
├── Training-EfficientDet/   # Access to the AI training module
└── Software-Inspeksi/       # This operational directory
```

---

## User Guide

### Step 1: Hardware Calibration
Ensure camera lighting and focus are optimal. Run:
```bash
python Calibrate_Camera.py
```
Save the configuration for each installed camera index.

### Step 2: Spatial Logic Configuration
Set the detection zone for each pianika model (P32E/P32EP) at every station:
```bash
python Calibrate_Zone.py
```

### Step 3: Production Operations
Run the main system according to the workstation placement:
* **Station 1**: `python main_system_station1.py`
* **Station 2**: `python main_system_station2.py`

---

## Data Integration
Every inspection result will be automatically recorded into:
1.  **Daily Statistics**: `statistik_station_X.json`.
2.  **Production Log**: `log_produksi_station_X.csv` which records time, status, missing components, system latency, and operator identity.

---
*This project is part of a final year research thesis by Juhen FW - FTMM Universitas Airlangga.*