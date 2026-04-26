# auto_label_per_file.py - Script untuk auto-labeling (1 JSON per gambar)
import torch
import cv2
import numpy as np
import yaml
import os
import json
from glob import glob
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

# --- KONFIGURASI ---
PROJECT_NAME = 'pianika_1'
COMPOUND_COEF = 1
WEIGHTS_PATH = r'D:\VSCODE\Skripsi_EfficientDet\Software\models\best_loss_d1_batch8.pth'
THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
USE_CUDA = True

# Folder input/output
INPUT_FOLDER = r'C:\Users\JuhenFW\Documents\HuaTengVision\P32EP'
OUTPUT_FOLDER = r'C:\Users\JuhenFW\Documents\HuaTengVision\P32EP'
# -------------------

def convert_bbox_to_center_format(x1, y1, x2, y2):
    """
    Konversi dari format (x1, y1, x2, y2) ke (center_x, center_y, width, height)
    Menggunakan float untuk menjaga presisi agar lebih 'ngepas'.
    """
    width = float(x2 - x1)
    height = float(y2 - y1)
    center_x = float(x1) + (width / 2.0)
    center_y = float(y1) + (height / 2.0)
    return center_x, center_y, width, height

def main():
    # 1. Load Parameter Project
    project_yml = f'projects/{PROJECT_NAME}.yml'
    try:
        with open(project_yml, 'r') as f:
            params = yaml.safe_load(f)
            obj_list = params['obj_list']
            print(f"[Info] Class names loaded: {obj_list}")
    except FileNotFoundError:
        print(f"[Error] File config tidak ditemukan: {project_yml}")
        return

    # 2. Setup Device
    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        cudnn.benchmark = True
    print(f"[Info] Running inference on: {device}")

    # 3. Load Model
    print(f"[Info] Loading model EfficientDet-D{COMPOUND_COEF}...")
    model = EfficientDetBackbone(compound_coef=COMPOUND_COEF, 
                                 num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), 
                                 scales=eval(params['anchors_scales']))
    
    if os.path.exists(WEIGHTS_PATH):
        try:
            # Gunakan map_location agar fleksibel antara CPU/GPU
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            print(f"[Info] Weights loaded successfully.")
        except Exception as e:
            print(f"[Error] Gagal load weights: {e}")
            return
    else:
        print(f"[Error] Weights tidak ditemukan di: {WEIGHTS_PATH}")
        return

    model.requires_grad_(False)
    model.eval()
    model.to(device)

    # 4. Create Output Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"[Info] Folder output dibuat: {OUTPUT_FOLDER}")

    # 5. Get Image Files
    image_extensions = ['*.jpg', ['*.jpeg'], '*.png', '*.bmp', '*.BMP', '*.webp']
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.BMP', '*.webp']:
        image_files.extend(glob(os.path.join(INPUT_FOLDER, ext)))
    
    if len(image_files) == 0:
        print(f"[Error] Tidak ada gambar ditemukan di folder: {INPUT_FOLDER}")
        return

    print(f"[Info] Ditemukan {len(image_files)} gambar untuk dilabeli")

    # 6. Input Size (D1 standar adalah 640)
    MAX_INPUT_SIZE = 640 

    # 7. Process All Images
    total_objects = 0
    images_with_objects = 0
    
    for idx, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        print(f"[{idx}/{len(image_files)}] {filename}...", end=" ")
        
        try:
            # Preprocess
            ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=MAX_INPUT_SIZE)
            x = torch.from_numpy(framed_imgs[0]).to(device).permute(2, 0, 1).unsqueeze(0).float()

            # Inference
            with torch.no_grad():
                features, regression, classification, anchors = model(x)
                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()
                out = postprocess(x, anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  THRESHOLD, IOU_THRESHOLD)

            # Invert Affine (Kembali ke koordinat gambar asli)
            out = invert_affine(framed_metas, out)
            
            # Create annotation entry
            image_annotation = {
                "image": filename,
                "verified": False,
                "annotations": []
            }
            
            if len(out[0]['rois']) > 0:
                for i in range(len(out[0]['rois'])):
                    # Ambil koordinat dengan presisi float asli
                    x1, y1, x2, y2 = out[0]['rois'][i].astype(float)
                    class_id = int(out[0]['class_ids'][i])
                    class_name = obj_list[class_id]
                    
                    # Convert ke format center
                    center_x, center_y, width, height = convert_bbox_to_center_format(x1, y1, x2, y2)
                    
                    annotation = {
                        "label": class_name,
                        "coordinates": {
                            "x": round(center_x, 3), # Menggunakan 3 desimal untuk presisi lebih baik
                            "y": round(center_y, 3),
                            "width": round(width, 3),
                            "height": round(height, 3)
                        }
                    }
                    image_annotation["annotations"].append(annotation)
                
                total_objects += len(out[0]['rois'])
                images_with_objects += 1
                print(f"✓ {len(out[0]['rois'])} objek", end=" ")
            else:
                print("⚠ Kosong", end=" ")
            
            # Save individual JSON file
            json_filename = f"{filename_no_ext}.json"
            json_path = os.path.join(OUTPUT_FOLDER, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([image_annotation], f, indent=2, ensure_ascii=False)
            
            print(f"→ Saved")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            # Entry kosong jika error
            image_annotation = {"image": filename, "verified": False, "annotations": []}
            json_path = os.path.join(OUTPUT_FOLDER, f"{filename_no_ext}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([image_annotation], f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}\n[Success] Auto-labeling selesai!\n{'='*70}")

if __name__ == '__main__':
    main()