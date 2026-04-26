# evaluate_metrics_combined.py - Script Evaluasi AP50 (PyCOCOTools) & Operational Recall (Manual)
import os
import json
import yaml
import torch
import copy
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

# ============================================================================
# KONFIGURASI FOLDER & MODEL
# ============================================================================
PROJECT_NAME = 'pianika_1'
COMPOUND_COEF = 0

# PENTING: Ganti path ini ke D0 atau D1 sesuai model yang sedang Anda uji!
WEIGHTS_PATH = r'D:\VSCODE\EfficientDet-Pytorch\logs\pianika_d0_batch9\best_loss_d0.pth'
# WEIGHTS_PATH = r'D:\VSCODE\EfficientDet-Pytorch\logs\pianika_d1_batch9\best_loss_d1.pth'

# --- PATH DATA TEST ---
IMG_DIR = r'D:\VSCODE\EfficientDet-Pytorch\datasets\pianika_1\valid'
COCO_JSON_PATH = r'D:\VSCODE\EfficientDet-Pytorch\datasets\pianika_1\annotations\instances_valid.json'

# --- PARAMETER EVALUASI ---
# Untuk AP50 (PyCOCOTools) agar mendapat kurva PR utuh
AP_CONFIDENCE_THRESHOLD = 0.01  
# Untuk Operational Recall (Manual) sesuai standar pabrik
OPERATIONAL_CONF_THRESHOLD = 0.50 
IOU_THRESHOLD = 0.5 

USE_CUDA = True
PREDICTIONS_JSON = 'predictions_temp.json'
# ============================================================================

def calculate_iou(box1, box2):
    """Menghitung Intersection over Union (IoU) format [x1, y1, x2, y2]"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter: return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0: return 0.0
    return inter_area / union_area

def load_coco_ground_truth(json_path):
    """Membaca GT untuk perhitungan Manual Recall"""
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    img_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}
    gt_dict = {img['file_name']: [] for img in coco_data['images']}
    
    for ann in coco_data['annotations']:
        img_name = img_id_to_name.get(ann['image_id'])
        if not img_name: continue
        label = category_map[ann['category_id']]
        x, y, w, h = ann['bbox']
        bbox = [x, y, x + w, y + h] # Konversi COCO ke VOC format
        gt_dict[img_name].append({'label': label, 'bbox': bbox, 'matched': False})
    return gt_dict

def main():
    print("="*90)
    print("EVALUATOR GABUNGAN: AP50 (PyCOCOTools) & OPERATIONAL RECALL (Manual)")
    print("="*90)

    # 1. Load Ground Truth (Untuk AP & Recall)
    print("[Info] Memuat Ground Truth...")
    cocoGt = COCO(COCO_JSON_PATH)
    img_name_to_id = {img_info['file_name']: img_id for img_id, img_info in cocoGt.imgs.items()}
    manual_ground_truths = load_coco_ground_truth(COCO_JSON_PATH)

    # 2. Setup Model
    device = torch.device('cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu')
    print(f"[Info] Menggunakan Device: {device}")
    
    project_yml = f'projects/{PROJECT_NAME}.yml'
    with open(project_yml, 'r') as f:
        params = yaml.safe_load(f)
        obj_list = params['obj_list']

    model = EfficientDetBackbone(compound_coef=COMPOUND_COEF, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    
    max_input_size = {0: 512, 1: 640, 2: 768, 3: 896, 4: 1024}.get(COMPOUND_COEF, 512)

    img_files = [f for f in os.listdir(IMG_DIR) if f in img_name_to_id]
    if not img_files: return print("[Error] Tidak ada gambar yang cocok.")

    print(f"\n[Info] Memulai Inferensi pada {len(img_files)} gambar...")
    
    # Variabel Penyimpanan
    coco_results = [] # Untuk AP50
    manual_metrics = {cls: {'TP': 0, 'FP': 0, 'FN': 0} for cls in obj_list} # Untuk Recall
    
    # 3. Looping Inferensi AI
    for idx, img_name in enumerate(img_files, 1):
        print(f"\rMemproses inferensi gambar {idx}/{len(img_files)}...", end="")
        img_path = os.path.join(IMG_DIR, img_name)
        img_id = img_name_to_id[img_name]

        # Inisialisasi FN Manual untuk gambar ini
        gt_objects = copy.deepcopy(manual_ground_truths[img_name])
        for gt in gt_objects:
            if gt['label'] in manual_metrics:
                manual_metrics[gt['label']]['FN'] += 1

        # Inferensi AI (Gunakan threshold rendah 0.01)
        ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=max_input_size)
        x = torch.from_numpy(framed_imgs[0]).to(device).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, AP_CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
        out = invert_affine(framed_metas, out)
        
        pred_objects_manual = []
        
        if len(out[0]['rois']) > 0:
            for i in range(len(out[0]['rois'])):
                class_id = int(out[0]['class_ids'][i])
                score = float(out[0]['scores'][i])
                bbox = out[0]['rois'][i] # [x1, y1, x2, y2]
                cat_name = obj_list[class_id]
                
                # --- CABANG 1: Simpan semua untuk PyCOCOTools (AP50) ---
                x_min, y_min, x_max, y_max = bbox
                w, h = x_max - x_min, y_max - y_min
                category_id = next((cat['id'] for cat in cocoGt.cats.values() if cat['name'] == cat_name), -1)
                
                coco_results.append({
                    "image_id": img_id, "category_id": category_id,
                    "bbox": [round(float(x_min), 2), round(float(y_min), 2), round(float(w), 2), round(float(h), 2)],
                    "score": round(score, 5)
                })
                
                # --- CABANG 2: Filter untuk Manual Recall (>= 0.50) ---
                if score >= OPERATIONAL_CONF_THRESHOLD:
                    pred_objects_manual.append({'label': cat_name, 'score': score, 'bbox': bbox})

        # Hitung Metrik Manual (Operational Recall)
        pred_objects_manual.sort(key=lambda x: x['score'], reverse=True)
        for pred in pred_objects_manual:
            best_iou, best_gt_idx = 0.0, -1
            for j, gt in enumerate(gt_objects):
                if gt['label'] == pred['label'] and not gt['matched']:
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou, best_gt_idx = iou, j
                        
            if best_iou >= IOU_THRESHOLD:
                gt_objects[best_gt_idx]['matched'] = True
                manual_metrics[pred['label']]['TP'] += 1
                manual_metrics[pred['label']]['FN'] -= 1
            else:
                manual_metrics[pred['label']]['FP'] += 1

    # 4. Evaluasi PyCOCOTools (Menghitung AP)
    print("\n\n[Info] Menyimpan JSON dan Menjalankan PyCOCOTools Evaluator...")
    with open(PREDICTIONS_JSON, 'w') as f:
        json.dump(coco_results, f)
        
    cocoDt = cocoGt.loadRes(PREDICTIONS_JSON)
    final_results = []
    
    print("\n" + "="*90)
    print(f"{'KELAS KOMPONEN':<18} | {'AP (IoU 0.50:0.95)':<18} | {'AP50 (IoU 0.50)':<18} | {'OPERATIONAL RECALL':<20}")
    print("-" * 90)
    
    import contextlib
    import io
    
    for catId in cocoGt.getCatIds():
        cat_name = cocoGt.cats[catId]['name']
        
        # Ekstrak AP dari PyCOCOTools
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.catIds = [catId]
        with contextlib.redirect_stdout(io.StringIO()):
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            
        ap_050_095 = cocoEval.stats[0] if cocoEval.stats[0] != -1 else 0.0
        ap_50 = cocoEval.stats[1] if cocoEval.stats[1] != -1 else 0.0
        
        # Ekstrak Manual Recall
        TP = manual_metrics[cat_name]['TP']
        FN = manual_metrics[cat_name]['FN']
        recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # Simpan ke list
        ap_050_095_pct = round(ap_050_095 * 100, 2)
        ap_50_pct = round(ap_50 * 100, 2)
        recall_pct = round(recall_val * 100, 2)
        
        final_results.append({
            'Kelas': cat_name,
            'AP (IoU 0.50:0.95)': ap_050_095_pct,
            'AP50 (IoU 0.50)': ap_50_pct,
            'Operational Recall': recall_pct
        })
        
        print(f"{cat_name:<18} | {ap_050_095_pct:<18} | {ap_50_pct:<18} | {recall_pct:<20}")

    print("-" * 90)
    
    # Simpan ke CSV
    df = pd.DataFrame(final_results)
    mean_ap = df['AP (IoU 0.50:0.95)'].mean()
    mean_ap50 = df['AP50 (IoU 0.50)'].mean()
    mean_recall = df['Operational Recall'].mean()
    print(f"{'RATA-RATA':<18} | {round(mean_ap, 2):<18} | {round(mean_ap50, 2):<18} | {round(mean_recall, 2):<20}")
    print("="*90)

    # Nama file dinamis tergantung model yang dites
    model_name = "d0" if "d0" in WEIGHTS_PATH.lower() else "d1"
    output_filename = f'FINAL_Evaluation_{model_name}.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n[Info] Selesai! Hasil berhasil disimpan ke '{output_filename}'")
    
    if os.path.exists(PREDICTIONS_JSON):
        os.remove(PREDICTIONS_JSON)

if __name__ == '__main__':
    main()