# eval_model_visualization.py - Evaluation dengan Visualisasi Comprehensive & COCO mAP 
import time  
import torch  
import cv2  
import numpy as np  
import yaml  
import os  
import json  
from pathlib import Path  
from tqdm import tqdm  
from collections import defaultdict  
from torch.backends import cudnn  
import matplotlib as plt  
import matplotlib.patches as patches  
import seaborn as sns  
from matplotlib.gridspec import GridSpec  

# --- TAMBAHAN UNTUK COCO mAP ---
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# -------------------------------

from backbone import EfficientDetBackbone  
from efficientdet.utils import BBoxTransform, ClipBoxes  
from utils.utils import preprocess, invert_affine, postprocess  

# --- KONFIGURASI MANUAL ---  
PROJECT_NAME = 'pianika_1' 
COMPOUND_COEF = 0 # Ganti ke 0 jika ingin mengevaluasi D0
WEIGHTS_PATH = r'D:\VSCODE\EfficientDet-Pytorch\logs\pianika_d0_batch9\best_loss_d0.pth' # Sesuaikan path
THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
USE_CUDA = True  

# Input sizes per compound coefficient  
INPUT_SIZES = {0: 512, 1: 640, 2: 768, 3: 896, 4: 1024}  
MAX_INPUT_SIZE = INPUT_SIZES.get(COMPOUND_COEF, 512)  

# Dataset paths  
DATASET_ROOT = 'D:/VSCODE/EfficientDet-Pytorch/datasets/pianika_1'  

# -------------------------------------------------------  

class COCODatasetLoader:  
    """Load COCO-format dataset"""  
    
    def __init__(self, ann_file, img_dir):  
        self.ann_file = ann_file  
        self.img_dir = img_dir  
        
        if not os.path.exists(self.img_dir):  
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")  
        if not os.path.exists(self.ann_file):  
            raise FileNotFoundError(f"Annotation file not found: {self.ann_file}")  
        
        print(f"[Info] Loading COCO dataset from: {self.ann_file}")  
        
        with open(self.ann_file, 'r') as f:  
            self.coco = json.load(f)  
        
        self.images = {img['id']: img for img in self.coco['images']}  
        self.categories = {cat['id']: cat['name'] for cat in self.coco['categories']}  
        
        self.image_annotations = defaultdict(list)  
        for ann in self.coco['annotations']:  
            self.image_annotations[ann['image_id']].append(ann)  
        
        self.valid_image_ids = list(self.image_annotations.keys())  
        
        print(f"[Info] Loaded {len(self.valid_image_ids)} images")  
        print(f"[Info] Categories: {list(self.categories.values())}")  
    
    def __len__(self):  
        return len(self.valid_image_ids)  
    
    def __getitem__(self, idx):  
        image_id = self.valid_image_ids[idx]  
        img_info = self.images[image_id]  
        image_name = img_info['file_name']  
        image_path = os.path.join(self.img_dir, image_name)  
        
        if not os.path.exists(image_path):  
            return None  
        
        annotations = []  
        for ann in self.image_annotations[image_id]:  
            x, y, w, h = ann['bbox'].values() if isinstance(ann['bbox'], dict) else ann['bbox']  
            x1 = x  
            y1 = y  
            x2 = x + w  
            y2 = y + h  
            
            category_id = ann['category_id']  
            class_name = self.categories.get(category_id, f'unknown_{category_id}')  
            
            annotations.append({  
                'class': class_name,  
                'category_id': category_id,  
                'bbox': [x1, y1, x2, y2]  
            })  
        
        return {  
            'image_path': image_path,  
            'image_name': image_name,  
            'annotations': annotations  
        }  


class ModelEvaluator:  
    def __init__(self, project_name, compound_coef, weights_path,   
                 threshold=0.5, iou_threshold=0.3, dataset_root='datasets/pianika'):  
        
        self.project_name = project_name  
        self.compound_coef = compound_coef  
        self.weights_path = weights_path  
        self.threshold = threshold  
        self.iou_threshold = iou_threshold  
        self.dataset_root = dataset_root  
        
        project_yml = f'projects/{project_name}.yml'  
        try:  
            with open(project_yml, 'r') as f:  
                self.params = yaml.safe_load(f)  
                self.obj_list = self.params['obj_list']  
                print(f"[Info] Class names: {self.obj_list}")  
        except FileNotFoundError:  
            print(f"[Error] Config file not found: {project_yml}")  
            raise  
        
        if USE_CUDA and torch.cuda.is_available():  
            self.device = torch.device('cuda')  
            cudnn.benchmark = True  
            print(f"[Info] Using CUDA (GPU)")  
        else:  
            self.device = torch.device('cpu')  
            print(f"[Info] Using CPU")  
        
        self._load_model()  
    
    def _load_model(self):  
        print(f"[Info] Loading EfficientDet-D{self.compound_coef}...")  
        
        self.model = EfficientDetBackbone(  
            compound_coef=self.compound_coef,  
            num_classes=len(self.obj_list),  
            ratios=eval(self.params['anchors_ratios']),  
            scales=eval(self.params['anchors_scales'])  
        )  
        
        if os.path.exists(self.weights_path):  
            try:  
                state_dict = torch.load(self.weights_path, map_location=self.device)  
                self.model.load_state_dict(state_dict)  
                print(f"[Info] Weights loaded: {self.weights_path}")  
            except Exception as e:  
                print(f"[Error] Failed to load weights: {e}")  
                raise  
        else:  
            print(f"[Error] Weights not found: {self.weights_path}")  
            raise FileNotFoundError(self.weights_path)  
        
        self.model.to(self.device)  
        self.model.eval()  
        self.model.requires_grad_(False)  
    
    def _calculate_iou(self, box1, box2):  
        x1_min, y1_min, x1_max, y1_max = box1  
        x2_min, y2_min, x2_max, y2_max = box2  
        
        inter_xmin = max(x1_min, x2_min)  
        inter_ymin = max(y1_min, y2_min)  
        inter_xmax = min(x1_max, x2_max)  
        inter_ymax = min(y1_max, y2_max)  
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:  
            return 0.0  
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)  
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)  
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)  
        union_area = box1_area + box2_area - inter_area  
        
        if union_area == 0:  
            return 0.0  
        
        return inter_area / union_area  
    
    def evaluate_image(self, image_path, gt_annotations):  
        if not os.path.exists(image_path):  
            return None  
        
        ori_img = cv2.imread(image_path)  
        if ori_img is None:  
            return None  
        
        ori_h, ori_w = ori_img.shape[:2]  
        
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=MAX_INPUT_SIZE)  
        x = torch.from_numpy(framed_imgs[0]).to(self.device).permute(2, 0, 1).unsqueeze(0).float()  
        
        with torch.no_grad():  
            features, regression, classification, anchors = self.model(x)  
            regressBoxes = BBoxTransform()  
            clipBoxes = ClipBoxes()  
            out = postprocess(x, anchors, regression, classification,  
                            regressBoxes, clipBoxes,  
                            self.threshold, self.iou_threshold)  
        
        out = invert_affine(framed_metas, out)  
        
        predictions = []  
        if len(out[0]['rois']) > 0:  
            for i in range(len(out[0]['rois'])):  
                x1, y1, x2, y2 = out[0]['rois'][i].astype(int)  
                score = float(out[0]['scores'][i])  
                class_id = int(out[0]['class_ids'][i])  
                class_name = self.obj_list[class_id]  
                
                predictions.append({  
                    'class': class_name,  
                    'class_id': class_id,  
                    'confidence': score,  
                    'bbox': [x1, y1, x2, y2]  
                })  
        
        matched_gt = set()  
        tp = 0  
        fp = 0  
        
        image_errors = {  
            'false_positives': [],  
            'false_negatives': [],  
            'low_confidence': [],  
            'misclassified': [],  
            'partial_detections': []  
        }  
        
        for pred_idx, pred in enumerate(predictions):  
            best_iou = 0  
            best_gt_idx = -1  
            
            for gt_idx, gt in enumerate(gt_annotations):  
                if gt_idx in matched_gt:  
                    continue  
                
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])  
                if iou > best_iou:  
                    best_iou = iou  
                    best_gt_idx = gt_idx  
            
            if best_iou >= 0.5:  
                if pred['class'] == gt_annotations[best_gt_idx]['class']:  
                    tp += 1  
                    matched_gt.add(best_gt_idx)  
                else:  
                    fp += 1  
                    image_errors['misclassified'].append({  
                        'predicted': pred['class'],  
                        'actual': gt_annotations[best_gt_idx]['class'],  
                        'confidence': pred['confidence'],  
                        'iou': best_iou  
                    })  
            elif best_iou > 0.1 and best_iou < 0.5:  
                fp += 1  
                image_errors['partial_detections'].append({  
                    'predicted': pred['class'],  
                    'actual': gt_annotations[best_gt_idx]['class'] if best_gt_idx >= 0 else 'unknown',  
                    'confidence': pred['confidence'],  
                    'iou': best_iou  
                })  
            elif pred['confidence'] < 0.7:  
                fp += 1  
                image_errors['low_confidence'].append({  
                    'class': pred['class'],  
                    'confidence': pred['confidence'],  
                    'has_match': best_gt_idx >= 0  
                })  
            else:  
                fp += 1  
                image_errors['false_positives'].append({  
                    'class': pred['class'],  
                    'confidence': pred['confidence']  
                })  
        
        fn = len(gt_annotations) - len(matched_gt)  
        for gt_idx, gt in enumerate(gt_annotations):  
            if gt_idx not in matched_gt:  
                image_errors['false_negatives'].append({  
                    'class': gt['class'],  
                    'bbox': gt['bbox']  
                })  
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  
        
        return {  
            'image_path': image_path,  
            'image_name': Path(image_path).name,  
            'image_size': f"{ori_w}×{ori_h}",  
            'gt_count': len(gt_annotations),  
            'pred_count': len(predictions),  
            'tp': tp,  
            'fp': fp,  
            'fn': fn,  
            'precision': precision,  
            'recall': recall,  
            'f1': f1,  
            'predictions': predictions,  
            'ground_truths': gt_annotations,  
            'errors': image_errors,  
            'ori_img': ori_img,  
            'ori_w': ori_w,  
            'ori_h': ori_h  
        }  
    
    def evaluate_dataset(self, split='valid'):  
        img_dir = os.path.join(self.dataset_root, split)  
        ann_file = os.path.join(self.dataset_root, 'annotations', f'instances_{split}.json')  
        
        try:  
            dataset = COCODatasetLoader(ann_file, img_dir)  
        except FileNotFoundError as e:  
            print(f"[Error] {e}")  
            return None  
        
        print(f"\n[Eval] Evaluating {len(dataset)} images dari split '{split}'...\n")  
        
        all_results = []  
        all_errors = defaultdict(list)  
        
        all_tp = 0  
        all_fp = 0  
        all_fn = 0  

        # --- TAMBAHAN UNTUK COCO mAP ---
        coco_predictions = []
        name_to_coco_id = {name: cat_id for cat_id, name in dataset.categories.items()}
        # -------------------------------
        
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):  
            try:  
                data = dataset[idx]  
                
                if data is None:  
                    continue  
                
                image_path = data['image_path']  
                gt_annotations = data['annotations']  
                image_id = dataset.valid_image_ids[idx] # Dapatkan ID gambar
                
                result = self.evaluate_image(image_path, gt_annotations)  
                
                if result:  
                    all_results.append(result)  
                    all_tp += result['tp']  
                    all_fp += result['fp']  
                    all_fn += result['fn']  
                    
                    for error_type, errors in result['errors'].items():  
                        all_errors[error_type].extend(errors)  

                    # --- TAMBAHAN UNTUK COCO mAP ---
                    for pred in result['predictions']:
                        # Konversi dari [x1, y1, x2, y2] ke [x, y, w, h] format COCO
                        x1, y1, x2, y2 = pred['bbox']
                        coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        cat_id = name_to_coco_id.get(pred['class'])
                        
                        if cat_id is not None:
                            coco_predictions.append({
                                'image_id': int(image_id),
                                'category_id': int(cat_id),
                                'bbox': coco_bbox,
                                'score': float(pred['confidence'])
                            })
                    # -------------------------------
            
            except Exception as e:  
                print(f"\n[Error] Failed to process image {idx}: {e}")  
                continue  
        
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0  
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0  
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0  

        # --- HITUNG COCO mAP ---
        mAP_50 = 0.0
        if coco_predictions:
            print("\n[Eval] Calculating COCO mAP. Please wait...")
            coco_gt = COCO(ann_file)
            coco_dt = coco_gt.loadRes(coco_predictions)
            
            # Jalankan evaluasi COCO
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Ambil nilai mAP @ IoU=0.50 (index ke-1 pada stats COCO)
            mAP_50 = float(coco_eval.stats[1])
        # -----------------------
        
        return {  
            'results': all_results,  
            'errors': dict(all_errors),  
            'overall_metrics': {  
                'total_images': len(all_results),  
                'tp': all_tp,  
                'fp': all_fp,  
                'fn': all_fn,  
                'precision': overall_precision,  
                'recall': overall_recall,  
                'f1': overall_f1,
                'mAP_50': mAP_50 # Masukkan mAP ke return dict
            }  
        }  
    
    def generate_report(self, evaluation_results, output_dir='eval_reports'):  
        """Generate comprehensive evaluation report dengan visualisasi"""  
        
        os.makedirs(output_dir, exist_ok=True)  
        
        overall = evaluation_results['overall_metrics']  
        errors = evaluation_results['errors']  
        results = evaluation_results['results']  
        
        # === 1. PRINT SUMMARY ===  
        print("\n" + "="*80)  
        print(f"{'EVALUATION REPORT':^80}")  
        print("="*80)  
        
        print(f"\n📊 OVERALL METRICS:")  
        print(f"  Total Images       : {overall['total_images']}")  
        print(f"  True Positives     : {overall['tp']}")  
        print(f"  False Positives    : {overall['fp']}")  
        print(f"  False Negatives    : {overall['fn']}")  
        print(f"  Precision          : {overall['precision']:.4f}")  
        print(f"  Recall             : {overall['recall']:.4f}")  
        print(f"  F1-Score           : {overall['f1']:.4f}")  
        print(f"  mAP@0.50           : {overall['mAP_50']:.4f}") # Cetak mAP
        
        print(f"\n❌ ERROR ANALYSIS:")  
        print(f"  False Positives    : {len(errors.get('false_positives', []))} detections")  
        print(f"  False Negatives    : {len(errors.get('false_negatives', []))} missed objects")  
        print(f"  Misclassified      : {len(errors.get('misclassified', []))} wrong class")  
        print(f"  Low Confidence     : {len(errors.get('low_confidence', []))} detections < 0.7")  
        print(f"  Partial Detections : {len(errors.get('partial_detections', []))} weak matches")  
        
        print("\n" + "="*80)  
        
        # === 2. SAVE VISUALIZATIONS ===  
        import matplotlib.pyplot as plt # Fix import issue dynamically
        
        self._save_metrics_visualization(overall, output_dir)  
        self._save_error_visualization(errors, output_dir)  
        self._save_per_class_metrics(results, output_dir)  
        self._save_sample_predictions(results, output_dir)  
        self._save_detailed_reports(results, errors, overall, output_dir) # Pass 'overall' kesini
    
    def _save_metrics_visualization(self, overall, output_dir):  
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
        fig.suptitle('Overall Evaluation Metrics', fontsize=16, fontweight='bold')  
        
        # 1. Confusion Matrix-like visualization  
        ax = axes[0, 0]  
        metrics = ['TP', 'FP', 'FN']  
        values = [overall['tp'], overall['fp'], overall['fn']]  
        colors = ['#2ecc71', '#e74c3c', '#f39c12']  
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)  
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')  
        ax.set_title('Detection Results', fontsize=12, fontweight='bold')  
        ax.grid(axis='y', alpha=0.3)  
        
        for bar, val in zip(bars, values):  
            height = bar.get_height()  
            ax.text(bar.get_x() + bar.get_width()/2., height,  
                   f'{int(val)}',  
                   ha='center', va='bottom', fontweight='bold')  
        
        # 2. Precision, Recall, F1  
        ax = axes[0, 1]  
        metrics = ['Precision', 'Recall', 'F1-Score']  
        values = [overall['precision'], overall['recall'], overall['f1']]  
        colors = ['#3498db', '#9b59b6', '#1abc9c']  
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)  
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')  
        ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')  
        ax.set_ylim([0, 1])  
        ax.grid(axis='y', alpha=0.3)  
        
        for bar, val in zip(bars, values):  
            height = bar.get_height()  
            ax.text(bar.get_x() + bar.get_width()/2., height,  
                   f'{val:.4f}',  
                   ha='center', va='bottom', fontweight='bold')  
        
        # 3. Summary Statistics  
        ax = axes[1, 0]  
        ax.axis('off')  
        
        summary_text = f"""  
        EVALUATION SUMMARY  
        
        Total Images Evaluated: {overall['total_images']}  
        
        True Positives (TP):    {overall['tp']}  
        False Positives (FP):   {overall['fp']}  
        False Negatives (FN):   {overall['fn']}  
        
        Precision:  {overall['precision']:.4f}  
        Recall:     {overall['recall']:.4f}  
        F1-Score:   {overall['f1']:.4f}  
        mAP@0.50:   {overall['mAP_50']:.4f}
        """  
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',  
               verticalalignment='center', bbox=dict(boxstyle='round',   
               facecolor='wheat', alpha=0.5))  
        
        # 4. Accuracy Rate  
        ax = axes[1, 1]  
        accuracy = overall['tp'] / (overall['tp'] + overall['fp'] + overall['fn']) \
                   if (overall['tp'] + overall['fp'] + overall['fn']) > 0 else 0  
        
        ax.text(0.5, 0.7, f'{accuracy:.2%}', fontsize=48, ha='center',   
               va='center', fontweight='bold', color='green')  
        ax.text(0.5, 0.3, 'Overall Accuracy', fontsize=14, ha='center',   
               va='center', fontweight='bold')  
        ax.set_xlim([0, 1])  
        ax.set_ylim([0, 1])  
        ax.axis('off')  
        
        plt.tight_layout()  
        metrics_path = os.path.join(output_dir, '01_overall_metrics.png')  
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')  
        plt.close()  
        print(f"✅ Metrics visualization saved: {metrics_path}")  
    
    def _save_error_visualization(self, errors, output_dir):  
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))  
        fig.suptitle('Error Analysis', fontsize=16, fontweight='bold')  
        
        # 1. Error types distribution  
        ax = axes[0, 0]  
        error_counts = {  
            'False Positives': len(errors.get('false_positives', [])),  
            'False Negatives': len(errors.get('false_negatives', [])),  
            'Misclassified': len(errors.get('misclassified', [])),  
            'Low Confidence': len(errors.get('low_confidence', [])),  
            'Partial Detections': len(errors.get('partial_detections', []))  
        }  
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#1abc9c']  
        bars = ax.barh(list(error_counts.keys()), list(error_counts.values()),   
                      color=colors, alpha=0.7, edgecolor='black', linewidth=2)  
        ax.set_xlabel('Count', fontweight='bold')  
        ax.set_title('Error Types Distribution', fontweight='bold')  
        ax.grid(axis='x', alpha=0.3)  
        
        for bar, val in zip(bars, error_counts.values()):  
            width = bar.get_width()  
            ax.text(width, bar.get_y() + bar.get_height()/2.,  
                   f'{int(val)}',  
                   ha='left', va='center', fontweight='bold', fontsize=10)  
        
        # 2. False Positives by Class  
        ax = axes[0, 1]  
        if errors.get('false_positives'):  
            fp_by_class = defaultdict(list)  
            for fp in errors['false_positives']:  
                fp_by_class[fp['class']].append(fp['confidence'])  
            
            classes = list(fp_by_class.keys())[:10]  # Top 10  
            counts = [len(fp_by_class[c]) for c in classes]  
            
            ax.barh(classes, counts, color='#e74c3c', alpha=0.7, edgecolor='black')  
            ax.set_xlabel('Count', fontweight='bold')  
            ax.set_title('False Positives by Class (Top 10)', fontweight='bold')  
            ax.grid(axis='x', alpha=0.3)  
        else:  
            ax.text(0.5, 0.5, 'No False Positives', ha='center', va='center',  
                   fontsize=14, transform=ax.transAxes)  
            ax.axis('off')  
        
        # 3. False Negatives by Class  
        ax = axes[1, 0]  
        if errors.get('false_negatives'):  
            fn_by_class = defaultdict(int)  
            for fn in errors['false_negatives']:  
                fn_by_class[fn['class']] += 1  
            
            classes = list(fn_by_class.keys())[:10]  # Top 10  
            counts = [fn_by_class[c] for c in classes]  
            
            ax.barh(classes, counts, color='#f39c12', alpha=0.7, edgecolor='black')  
            ax.set_xlabel('Count', fontweight='bold')  
            ax.set_title('False Negatives by Class (Top 10)', fontweight='bold')  
            ax.grid(axis='x', alpha=0.3)  
        else:  
            ax.text(0.5, 0.5, 'No False Negatives', ha='center', va='center',  
                   fontsize=14, transform=ax.transAxes)  
            ax.axis('off')  
        
        # 4. Misclassification Matrix  
        ax = axes[1, 1]  
        if errors.get('misclassified'):  
            misclass = defaultdict(lambda: defaultdict(int))  
            for m in errors['misclassified']:  
                misclass[m['actual']][m['predicted']] += 1  
            
            ax.text(0.5, 0.5, f"Total Misclassified: {len(errors['misclassified'])}",   
                   ha='center', va='center', fontsize=14, fontweight='bold',  
                   transform=ax.transAxes)  
            ax.axis('off')  
        else:  
            ax.text(0.5, 0.5, 'No Misclassifications', ha='center', va='center',  
                   fontsize=14, transform=ax.transAxes)  
            ax.axis('off')  
        
        plt.tight_layout()  
        error_path = os.path.join(output_dir, '02_error_analysis.png')  
        plt.savefig(error_path, dpi=300, bbox_inches='tight')  
        plt.close()  
        print(f"✅ Error visualization saved: {error_path}")  
    
    def _save_per_class_metrics(self, results, output_dir):  
        import matplotlib.pyplot as plt
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})  
        
        for result in results:  
            for pred in result['predictions']:  
                class_name = pred['class']  
                class_metrics[class_name]['tp'] += 1  
            
            for error_type in ['false_positives']:  
                for error in result['errors'].get(error_type, []):  
                    class_name = error.get('class')  
                    if class_name:  
                        class_metrics[class_name]['fp'] += 1  
            
            for error in result['errors'].get('false_negatives', []):  
                class_name = error['class']  
                class_metrics[class_name]['fn'] += 1  
        
        # Calculate metrics per class  
        class_perf = {}  
        for class_name, metrics in class_metrics.items():  
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']  
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0  
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  
            
            class_perf[class_name] = {  
                'tp': tp, 'fp': fp, 'fn': fn,  
                'precision': precision,  
                'recall': recall,  
                'f1': f1  
            }  
        
        # Visualization  
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  
        fig.suptitle('Per-Class Performance Metrics', fontsize=16, fontweight='bold')  
        
        # F1-Score per class  
        ax = axes[0]  
        classes = sorted(class_perf.keys())  
        f1_scores = [class_perf[c]['f1'] for c in classes]  
        
        bars = ax.barh(classes, f1_scores, color='#2ecc71', alpha=0.7, edgecolor='black')  
        ax.set_xlabel('F1-Score', fontweight='bold')  
        ax.set_title('F1-Score per Class', fontweight='bold')  
        ax.set_xlim([0, 1])  
        ax.grid(axis='x', alpha=0.3)  
        
        for bar, val in zip(bars, f1_scores):  
            width = bar.get_width()  
            ax.text(width, bar.get_y() + bar.get_height()/2.,  
                   f'{val:.3f}',  
                   ha='left', va='center', fontweight='bold', fontsize=9)  
        
        # Precision & Recall per class  
        ax = axes[1]  
        x = np.arange(len(classes))  
        width = 0.35  
        
        precisions = [class_perf[c]['precision'] for c in classes]  
        recalls = [class_perf[c]['recall'] for c in classes]  
        
        ax.bar(x - width/2, precisions, width, label='Precision', color='#3498db', alpha=0.7, edgecolor='black')  
        ax.bar(x + width/2, recalls, width, label='Recall', color='#9b59b6', alpha=0.7, edgecolor='black')  
        
        ax.set_ylabel('Score', fontweight='bold')  
        ax.set_title('Precision & Recall per Class', fontweight='bold')  
        ax.set_xticks(x)  
        ax.set_xticklabels(classes, rotation=45, ha='right')  
        ax.set_ylim([0, 1])  
        ax.legend()  
        ax.grid(axis='y', alpha=0.3)  
        
        plt.tight_layout()  
        per_class_path = os.path.join(output_dir, '03_per_class_metrics.png')  
        plt.savefig(per_class_path, dpi=300, bbox_inches='tight')  
        plt.close()  
        print(f"✅ Per-class metrics saved: {per_class_path}")  
        
        # Save per-class detailed report
        report_path = os.path.join(output_dir, 'per_class_detailed_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PER-CLASS PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")
            
            for class_name in sorted(class_perf.keys()):
                perf = class_perf[class_name]
                f.write(f"\n📦 Class: {class_name}\n")
                f.write(f"   TP (True Positive)      : {perf['tp']:3d}\n")
                f.write(f"   FP (False Positive)     : {perf['fp']:3d}\n")
                f.write(f"   FN (False Negative)     : {perf['fn']:3d}\n")
                f.write(f"   Precision               : {perf['precision']:.4f}\n")
                f.write(f"   Recall                  : {perf['recall']:.4f}\n")
                f.write(f"   F1-Score                : {perf['f1']:.4f}\n")
                f.write("-"*70 + "\n")
        
        print(f"✅ Per-class detailed report saved: {report_path}")
    
    def _save_sample_predictions(self, results, output_dir):
        sample_dir = os.path.join(output_dir, 'sample_predictions')
        os.makedirs(sample_dir, exist_ok=True)
        
        sorted_by_f1 = sorted(results, key=lambda x: x['f1'])
        
        # Best 3 images
        best_images = sorted_by_f1[-3:]
        for idx, result in enumerate(best_images):
            self._draw_detection_result(result, os.path.join(sample_dir, f'00_best_{idx+1}_{result["image_name"]}'))
        
        # Worst 3 images
        worst_images = sorted_by_f1[:3]
        for idx, result in enumerate(worst_images):
            self._draw_detection_result(result, os.path.join(sample_dir, f'01_worst_{idx+1}_{result["image_name"]}'))
        
        print(f"✅ Sample predictions saved to: {sample_dir}")
    
    def _draw_detection_result(self, result, save_path):
        ori_img = result['ori_img'].copy()
        ori_img_rgb = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        
        # Draw ground truth (green box)
        for gt in result['ground_truths']:
            x1, y1, x2, y2 = [int(x) for x in gt['bbox']]
            cv2.rectangle(ori_img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(ori_img_rgb, f"GT: {gt['class']}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw predictions (blue box for correct, red for incorrect)
        for pred in result['predictions']:
            x1, y1, x2, y2 = [int(x) for x in pred['bbox']]
            
            is_correct = False
            best_iou = 0
            for gt in result['ground_truths']:
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= 0.5 and pred['class'] == gt['class']:
                    is_correct = True
                    best_iou = iou
            
            color = (0, 0, 255) if not is_correct else (0, 165, 255)  
            cv2.rectangle(ori_img_rgb, (x1, y1), (x2, y2), color, 2)
            
            text = f"{pred['class']}: {pred['confidence']:.2f}"
            cv2.putText(ori_img_rgb, text, (x1, y1-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        stats_text = f"F1: {result['f1']:.3f} | TP:{result['tp']} FP:{result['fp']} FN:{result['fn']}"
        cv2.putText(ori_img_rgb, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ori_img_bgr = cv2.cvtColor(ori_img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + '.jpg', ori_img_bgr)
    
    # --- TERIMA ARGUMEN 'overall' DISINI ---
    def _save_detailed_reports(self, results, errors, overall, output_dir):
        
        # === REPORT 1: Summary Report ===
        summary_path = os.path.join(output_dir, '00_summary_report.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION SUMMARY REPORT\n".center(80))
            f.write("="*80 + "\n\n")
            
            f.write("BASIC STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Images Evaluated: {len(results)}\n")
            f.write(f"Total Detections: {sum(r['pred_count'] for r in results)}\n")
            f.write(f"Total Ground Truth Objects: {sum(r['gt_count'] for r in results)}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            
            # Gunakan metrics dari 'overall'
            f.write(f"True Positives (TP):  {overall['tp']}\n")
            f.write(f"False Positives (FP): {overall['fp']}\n")
            f.write(f"False Negatives (FN): {overall['fn']}\n\n")
            
            f.write(f"Precision: {overall['precision']:.4f}\n")
            f.write(f"Recall:    {overall['recall']:.4f}\n")
            f.write(f"F1-Score:  {overall['f1']:.4f}\n")
            f.write(f"mAP@0.50:  {overall['mAP_50']:.4f}\n\n") # --- CETAK mAP KE DALAM FILE ---
            
            f.write("BEST PERFORMING IMAGES (Top 5)\n")
            f.write("-"*80 + "\n")
            sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)[:5]
            for idx, res in enumerate(sorted_results, 1):
                f.write(f"{idx}. {res['image_name']:50s} | F1: {res['f1']:.4f}\n")
            
            f.write("\nWORST PERFORMING IMAGES (Bottom 5)\n")
            f.write("-"*80 + "\n")
            sorted_results = sorted(results, key=lambda x: x['f1'])[:5]
            for idx, res in enumerate(sorted_results, 1):
                f.write(f"{idx}. {res['image_name']:50s} | F1: {res['f1']:.4f}\n")
        
        print(f"✅ Summary report saved: {summary_path}")
        
        # === REPORT 2: Detailed Error Analysis ===
        error_path = os.path.join(output_dir, '01_detailed_error_analysis.txt')
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED ERROR ANALYSIS\n".center(80))
            f.write("="*80 + "\n\n")
            
            f.write("FALSE POSITIVES ANALYSIS\n")
            f.write("-"*80 + "\n")
            if errors.get('false_positives'):
                fp_by_class = defaultdict(list)
                for fp in errors['false_positives']:
                    fp_by_class[fp['class']].append(fp['confidence'])
                
                for class_name in sorted(fp_by_class.keys()):
                    confs = fp_by_class[class_name]
                    f.write(f"\n  {class_name}:\n")
                    f.write(f"    Count: {len(confs)}\n")
                    f.write(f"    Avg Confidence: {np.mean(confs):.4f}\n")
                    f.write(f"    Min Confidence: {np.min(confs):.4f}\n")
                    f.write(f"    Max Confidence: {np.max(confs):.4f}\n")
            else:
                f.write("No false positives detected!\n")
            
            f.write("\n\nFALSE NEGATIVES ANALYSIS (MISSED OBJECTS)\n")
            f.write("-"*80 + "\n")
            if errors.get('false_negatives'):
                fn_by_class = defaultdict(int)
                for fn in errors['false_negatives']:
                    fn_by_class[fn['class']] += 1
                
                for class_name in sorted(fn_by_class.keys()):
                    count = fn_by_class[class_name]
                    f.write(f"  {class_name:30s}: {count:3d} missed objects\n")
            else:
                f.write("No false negatives detected!\n")
            
            f.write("\n\nMISCLASSIFICATION ANALYSIS\n")
            f.write("-"*80 + "\n")
            if errors.get('misclassified'):
                f.write(f"Total misclassified: {len(errors['misclassified'])}\n\n")
                
                misclass_matrix = defaultdict(lambda: defaultdict(int))
                for m in errors['misclassified']:
                    misclass_matrix[m['actual']][m['predicted']] += 1
                
                for actual_class in sorted(misclass_matrix.keys()):
                    f.write(f"  {actual_class}:\n")
                    for pred_class, count in misclass_matrix[actual_class].items():
                        f.write(f"    Predicted as {pred_class}: {count}\n")
            else:
                f.write("No misclassifications detected!\n")
            
            f.write("\n\nLOW CONFIDENCE DETECTIONS (<0.7)\n")
            f.write("-"*80 + "\n")
            if errors.get('low_confidence'):
                f.write(f"Total: {len(errors['low_confidence'])}\n")
                lc_by_class = defaultdict(list)
                for lc in errors['low_confidence']:
                    lc_by_class[lc['class']].append(lc['confidence'])
                
                for class_name in sorted(lc_by_class.keys()):
                    confs = lc_by_class[class_name]
                    f.write(f"  {class_name:30s}: {len(confs):3d} | Avg: {np.mean(confs):.4f}\n")
            else:
                f.write("No low confidence detections!\n")
            
            f.write("\n\nPARTIAL DETECTIONS (IoU 0.1-0.5)\n")
            f.write("-"*80 + "\n")
            if errors.get('partial_detections'):
                f.write(f"Total: {len(errors['partial_detections'])}\n\n")
                
                pd_by_class = defaultdict(list)
                for pd in errors['partial_detections']:
                    pd_by_class[pd['predicted']].append(pd['iou'])
                
                for class_name in sorted(pd_by_class.keys()):
                    ious = pd_by_class[class_name]
                    f.write(f"  {class_name:30s}: {len(ious):3d} | Avg IoU: {np.mean(ious):.4f}\n")
            else:
                f.write("No partial detections!\n")
        
        print(f"✅ Detailed error analysis saved: {error_path}")
        
        # === REPORT 3: Per-Image Detailed Results ===
        detailed_path = os.path.join(output_dir, 'per_image_results.csv')
        import csv
        
        with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Image', 'Size', 'GT_Count', 'Pred_Count', 'TP', 'FP', 'FN',
                'Precision', 'Recall', 'F1', 'Accuracy'
            ])
            
            for result in sorted(results, key=lambda x: x['f1'], reverse=True):
                accuracy = result['tp'] / (result['tp'] + result['fp'] + result['fn']) \
                          if (result['tp'] + result['fp'] + result['fn']) > 0 else 0
                
                writer.writerow([
                    result['image_name'],
                    result['image_size'],
                    result['gt_count'],
                    result['pred_count'],
                    result['tp'],
                    result['fp'],
                    result['fn'],
                    f"{result['precision']:.4f}",
                    f"{result['recall']:.4f}",
                    f"{result['f1']:.4f}",
                    f"{accuracy:.4f}"
                ])
        
        print(f"✅ Per-image results CSV saved: {detailed_path}")
    
    def _save_confusion_matrix_visualization(self, results, output_dir):
        import matplotlib.pyplot as plt
        all_predictions = []
        all_ground_truths = []
        
        for result in results:
            for pred in result['predictions']:
                is_correct = False
                best_iou = 0
                gt_class = None
                
                for gt in result['ground_truths']:
                    iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        gt_class = gt['class']
                        if iou >= 0.5 and pred['class'] == gt['class']:
                            is_correct = True
                
                if best_iou >= 0.5:
                    all_predictions.append(pred['class'])
                    all_ground_truths.append(gt_class)
            
            for gt in result['ground_truths']:
                found = False
                for pred in result['predictions']:
                    iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                    if iou >= 0.5 and pred['class'] == gt['class']:
                        found = True
                        break
                
                if not found:
                    all_ground_truths.append(gt['class'])
                    all_predictions.append('not_detected')
        
        if not all_predictions or not all_ground_truths:
            return
        
        from sklearn.metrics import confusion_matrix
        
        classes = sorted(set(all_predictions + all_ground_truths))
        cm = confusion_matrix(all_ground_truths, all_predictions, labels=classes)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        cm_path = os.path.join(output_dir, '04_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Confusion matrix saved: {cm_path}")


def main():
    print(f"🔍 EfficientDet Model Evaluator dengan Visualisasi & COCO mAP")
    print(f"Model: {WEIGHTS_PATH}\n")
    
    evaluator = ModelEvaluator(
        project_name=PROJECT_NAME,
        compound_coef=COMPOUND_COEF,
        weights_path=WEIGHTS_PATH,
        threshold=THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        dataset_root=DATASET_ROOT
    )
    
    print(f"[Info] Evaluating on validation set...\n")
    eval_results = evaluator.evaluate_dataset(split='valid')
    
    if eval_results:
        evaluator.generate_report(eval_results, output_dir='eval_reports')
        
        try:
            evaluator._save_confusion_matrix_visualization(
                eval_results['results'], 
                'eval_reports'
            )
        except ImportError:
            print("[Warning] scikit-learn not installed, skipping confusion matrix")
        
        print(f"\n{'='*80}")
        print(f"✅ EVALUATION COMPLETE!".center(80))
        print(f"{'='*80}")
        print(f"\n📁 Output saved to: eval_reports/")
        print(f"\n📊 Generated files:")
        print(f"   - 00_summary_report.txt (Kini memuat skor mAP@0.50)")
        print(f"   - 01_detailed_error_analysis.txt")
        print(f"   - 01_overall_metrics.png")
        print(f"   - 02_error_analysis.png")
        print(f"   - 03_per_class_metrics.png")
        print(f"   - 04_confusion_matrix.png")
        print(f"   - per_class_detailed_report.txt")
        print(f"   - per_image_results.csv")
        print(f"   - sample_predictions/")
    else:
        print(f"\n❌ Evaluation failed!")


if __name__ == '__main__':
    main()