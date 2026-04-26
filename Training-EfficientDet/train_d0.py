# train_optimized.py - Memory-Efficient Training with Mixup
"""
Optimized training script dengan:
- Memory optimization (gradient accumulation, cache clearing)
- Mixup augmentation untuk better generalization
- Clean code structure
- Better error handling
- Complete metrics (mAP, F1, Precision, Recall)
"""

import argparse
import datetime
import time
import os
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from torch.amp import autocast, GradScaler

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.utils import init_weights, boolean_string
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess
from pycocotools.cocoeval import COCOeval

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================
class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


# ============================================================================
# MIXUP AUGMENTATION
# ============================================================================
def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation untuk better generalization
    alpha: mixup strength (0.2 recommended)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculate mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# MODEL WITH LOSS
# ============================================================================
class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations):
        _, regression, classification, anchors = self.model(imgs)
        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss

# ============================================================================
# METRICS TRACKER CLASS (TAMBAH INI - LETAKKAN SETELAH class Params)
# ============================================================================
class MetricsTracker:
    """Track all training metrics"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'mAP': [],
            'F1': [],
            'precision': [],
            'recall': [],
            'TP': [],
            'FP': [],
            'FN': []
        }
        
        self.confusion_matrix = None
        
    def update(self, epoch, train_loss=None, val_loss=None, lr=None,
               map_score=None, f1=None, precision=None, recall=None,
               tp=None, fp=None, fn=None):
        """Update metrics for current epoch"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss if train_loss is not None else np.nan)
        self.history['val_loss'].append(val_loss if val_loss is not None else np.nan)
        self.history['learning_rate'].append(lr if lr is not None else np.nan)
        self.history['mAP'].append(map_score if map_score is not None else np.nan)
        self.history['F1'].append(f1 if f1 is not None else np.nan)
        self.history['precision'].append(precision if precision is not None else np.nan)
        self.history['recall'].append(recall if recall is not None else np.nan)
        self.history['TP'].append(tp if tp is not None else np.nan)
        self.history['FP'].append(fp if fp is not None else np.nan)
        self.history['FN'].append(fn if fn is not None else np.nan)
    
    def save_csv(self):
        """Save all metrics to CSV"""
        df = pd.DataFrame(self.history)
        csv_path = os.path.join(self.save_dir, 'training_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"✅ Metrics saved to: {csv_path}")
        return csv_path
    
    def plot_all(self):
        """Generate all plots"""
        print("\n[Info] Generating training plots...")
        self._plot_losses()
        self._plot_lr()
        self._plot_performance()
        self._plot_precision_recall()
        self._plot_detection_metrics()
        if self.confusion_matrix is not None:
            self._plot_confusion_matrix()
        print(f"✅ All plots saved to: {self.save_dir}")
    
    def _plot_losses(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = self.history['epoch']
        train_loss = [x for x in self.history['train_loss'] if not np.isnan(x)]
        val_loss = [x for x in self.history['val_loss'] if not np.isnan(x)]
        
        if train_loss:
            ax.plot(epochs[:len(train_loss)], train_loss, 'b-', label='Train Loss', linewidth=2)
        if val_loss:
            val_epochs = [e for e, v in zip(epochs, self.history['val_loss']) if not np.isnan(v)]
            val_values = [v for v in self.history['val_loss'] if not np.isnan(v)]
            ax.plot(val_epochs, val_values, 'r-', label='Val Loss', linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lr(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = self.history['epoch']
        lr = [x for x in self.history['learning_rate'] if not np.isnan(x)]
        
        if lr:
            ax.plot(epochs[:len(lr)], lr, 'g-', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_performance(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = self.history['epoch']
        map_scores = [x for x in self.history['mAP'] if not np.isnan(x)]
        f1_scores = [x for x in self.history['F1'] if not np.isnan(x)]
        
        if map_scores:
            map_epochs = [e for e, v in zip(epochs, self.history['mAP']) if not np.isnan(v)]
            ax.plot(map_epochs, map_scores, 'b-', label='mAP', linewidth=2, marker='s')
        if f1_scores:
            f1_epochs = [e for e, v in zip(epochs, self.history['F1']) if not np.isnan(v)]
            ax.plot(f1_epochs, f1_scores, 'r-', label='F1-Score', linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Metrics (mAP & F1-Score)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = self.history['epoch']
        precision = [x for x in self.history['precision'] if not np.isnan(x)]
        recall = [x for x in self.history['recall'] if not np.isnan(x)]
        
        if precision:
            prec_epochs = [e for e, v in zip(epochs, self.history['precision']) if not np.isnan(v)]
            ax.plot(prec_epochs, precision, 'b-', label='Precision', linewidth=2, marker='s')
        if recall:
            rec_epochs = [e for e, v in zip(epochs, self.history['recall']) if not np.isnan(v)]
            ax.plot(rec_epochs, recall, 'r-', label='Recall', linewidth=2, marker='o')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision and Recall', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detection_metrics(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = self.history['epoch']
        tp = [x for x in self.history['TP'] if not np.isnan(x)]
        fp = [x for x in self.history['FP'] if not np.isnan(x)]
        fn = [x for x in self.history['FN'] if not np.isnan(x)]
        
        if tp:
            tp_epochs = [e for e, v in zip(epochs, self.history['TP']) if not np.isnan(v)]
            ax.plot(tp_epochs, tp, 'g-', label='True Positives', linewidth=2, marker='o')
        if fp:
            fp_epochs = [e for e, v in zip(epochs, self.history['FP']) if not np.isnan(v)]
            ax.plot(fp_epochs, fp, 'r-', label='False Positives', linewidth=2, marker='s')
        if fn:
            fn_epochs = [e for e, v in zip(epochs, self.history['FN']) if not np.isnan(v)]
            ax.plot(fn_epochs, fn, 'orange', label='False Negatives', linewidth=2, marker='^')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Detection Metrics (TP, FP, FN)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'detection_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                    ax=ax, cbar_kws={'label': 'Count'}, 
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_xlabel('Predicted Class', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Class', fontsize=13, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def set_confusion_matrix(self, cm):
        """Set confusion matrix"""
        self.confusion_matrix = cm

# ============================================================================
# ENHANCED EVALUATION (GANTI FUNCTION evaluate_coco YANG LAMA)
# ============================================================================
def evaluate_coco_with_metrics(dataset, model, threshold=0.05, iou_threshold=0.5, device='cuda'):
    """
    Enhanced evaluation with TP/FP/FN metrics
    Returns: dict with mAP, F1, precision, recall, TP, FP, FN
    """
    model.eval()
    results = []
    
    # Track GT and predictions for TP/FP/FN calculation
    total_gt = 0
    total_pred = 0
    
    print("\n[Eval] Calculating comprehensive metrics...")
    
    # ... (COPY label mapping code dari function lama) ...
    coco_cats = dataset.coco.loadCats(dataset.coco.getCatIds())
    name_to_coco_id = {cat['name']: cat['id'] for cat in coco_cats}
    
    classes_raw = dataset.classes
    idx_to_coco_id = {}
    
    if isinstance(classes_raw, list):
        for idx, name in enumerate(classes_raw):
            idx_to_coco_id[idx] = name_to_coco_id.get(name, idx + 1)
    elif isinstance(classes_raw, dict):
        first_key = next(iter(classes_raw))
        if isinstance(first_key, int):
            for idx, name in classes_raw.items():
                idx_to_coco_id[idx] = name_to_coco_id.get(name, idx + 1)
        else:
            for name, idx in classes_raw.items():
                idx = int(idx)
                idx_to_coco_id[idx] = name_to_coco_id.get(name, idx + 1)
    
    # Count GT annotations
    for idx in range(len(dataset)):
        image_id = dataset.image_ids[idx]
        ann_ids = dataset.coco.getAnnIds(imgIds=[image_id])
        total_gt += len(ann_ids)
    
    # ... (COPY inference code dari function lama) ...
    with torch.no_grad():
        for index in tqdm(range(len(dataset)), desc="Evaluating", leave=False):
            data = dataset[index]
            imgs = data['img'].permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            _, regression, classification, anchors = model(imgs)
            
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            
            out = postprocess(imgs, anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold=threshold, iou_threshold=iou_threshold)
            
            if 'meta' in data:
                out = invert_affine([data['meta']], out)
            else:
                scale = data.get('scale', 1.0)
                if len(out[0]['rois']) > 0:
                    out[0]['rois'] /= scale
            
            total_pred += len(out[0]['rois'])
            
            for i in range(len(out[0]['rois'])):
                x1, y1, x2, y2 = out[0]['rois'][i]
                score = float(out[0]['scores'][i])
                label = int(out[0]['class_ids'][i])
                
                bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                final_cat_id = idx_to_coco_id.get(label, label + 1)
                image_id = dataset.image_ids[index]
                
                results.append({
                    'image_id': int(image_id),
                    'category_id': int(final_cat_id),
                    'bbox': bbox,
                    'score': score
                })
            
            # if index % 100 == 0:
            #     torch.cuda.empty_cache()
    
    if not results:
        return {
            'mAP': 0.0, 'F1': 0.0, 'precision': 0.0, 'recall': 0.0,
            'TP': 0, 'FP': 0, 'FN': 0
        }
    
    # COCO evaluation
    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    map_score = coco_eval.stats[0]

    # --- TAMBAHAN BARU: EKSTRAK METRIK PER KELAS ---
    per_class_data = {}
    for i, catId in enumerate(coco_gt.getCatIds()):
        cat_name = coco_gt.loadCats(catId)[0]['name']
        
        # 1. Ekstrak AP@0.5 (mAP dengan IoU=0.50)
        p_ap50 = coco_eval.eval['precision'][0, :, i, 0, 2] # T=0 (0.50), A=0 (all), M=2 (maxDets=100)
        valid_ap50 = p_ap50[p_ap50 > -1]
        ap50 = float(np.mean(valid_ap50)) if len(valid_ap50) > 0 else 0.0
        
        # 2. Ekstrak Recall
        # recall matrix shape: [T, K, A, M] -> T=10, K=classes, A=0, M=2
        r_rec = coco_eval.eval['recall'][:, i, 0, 2]
        valid_rec = r_rec[r_rec > -1]
        recall_val = float(np.mean(valid_rec)) if len(valid_rec) > 0 else 0.0
        
        # 3. Ekstrak Precision @0.50
        precision_val = float(np.mean(valid_ap50)) if len(valid_ap50) > 0 else 0.0
        
        per_class_data[cat_name] = {
            'mAP_50': ap50,
            'Precision': precision_val,
            'Recall': recall_val
        }
    # -----------------------------------------------
    
    # ← TAMBAHAN: Calculate TP, FP, FN (simplified estimation)
    # Dari COCO eval precision matrix
    precision_matrix = coco_eval.eval['precision'][0, :, :, 0, 2]
    valid_precisions = precision_matrix[precision_matrix > -1]
    mean_precision = np.mean(valid_precisions) if len(valid_precisions) > 0 else 0.0
    mean_recall = coco_eval.stats[8]
    
    # f1_score = 0.0
    # if (mean_precision + mean_recall) > 0:
    #     f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)

    mean_precision = max(0.0, mean_precision)
    mean_recall = max(0.0, coco_eval.stats[8])

    f1_score = 0.0
    if (mean_precision + mean_recall) > 0:
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
    
    # Estimate TP, FP, FN
    TP = int(total_pred * mean_precision)
    FP = total_pred - TP
    FN = total_gt - TP
    
    print(f"📊 mAP: {map_score:.4f} | F1: {f1_score:.4f} | Prec: {mean_precision:.4f} | Rec: {mean_recall:.4f}")
    print(f"📊 TP: {TP} | FP: {FP} | FN: {FN}")
    
    return {
        'mAP': float(map_score),
        'F1': float(f1_score),
        'precision': float(mean_precision),
        'recall': float(mean_recall),
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'per_class': per_class_data
    }


# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train(opt):
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Info] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load config
    params = Params(f'projects/{opt.project}.yml')

    # ✨ Determine output folder name
    if opt.output_name:
        output_folder_name = opt.output_name
    else:
        output_folder_name = params.project_name

    # Add timestamp if requested
    if opt.add_timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_folder_name = f"{output_folder_name}_{timestamp}"

    # Set paths with custom folder name
    opt.saved_path = f'{opt.saved_path}/{output_folder_name}/'
    opt.log_path = f'{opt.log_path}/{output_folder_name}/tensorboard/'
    metrics_dir = f'{opt.saved_path}/metrics/'

    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Initialize tracker
    tracker = MetricsTracker(metrics_dir)

    # DataLoaders
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[opt.compound_coef]

    print(f"\n[Info] Loading datasets...")
    training_set = CocoDataset(
        root_dir=os.path.join(opt.data_path, opt.dataset_name), 
        set=params.train_set,
        transform=transforms.Compose([
            Normalizer(mean=params.mean, std=params.std),
            Augmenter(),
            Resizer(input_size)
        ])
    )

    val_set = CocoDataset(
        root_dir=os.path.join(opt.data_path, opt.dataset_name), 
        set=params.val_set,
        transform=transforms.Compose([
            Normalizer(mean=params.mean, std=params.std),
            Resizer(input_size)
        ])
    )

    training_generator = DataLoader(
        training_set, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=opt.num_workers, 
        collate_fn=collater, 
        pin_memory=True,
        drop_last=True  # Avoid batch size mismatch
    )

    val_generator = DataLoader(
        val_set, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=opt.num_workers, 
        collate_fn=collater, 
        pin_memory=True
    )

    print(f"[Info] Train: {len(training_set)} | Val: {len(val_set)} | Batch: {opt.batch_size}")

    # Model
    print(f"[Info] Building EfficientDet-D{opt.compound_coef}...")
    model = EfficientDetBackbone(
        num_classes=len(params.obj_list), 
        compound_coef=opt.compound_coef,
        ratios=eval(params.anchors_ratios), 
        scales=eval(params.anchors_scales)
    )

    # Load pretrained weights
    if opt.load_weights and os.path.exists(opt.load_weights):
        try:
            state_dict = torch.load(opt.load_weights, map_location='cpu')
            # Filter classifier/regressor for transfer learning
            filtered = {k: v for k, v in state_dict.items() 
                       if 'classifier' not in k and 'regressor' not in k}
            model.load_state_dict(filtered, strict=False)
            print(f"[Info] Pretrained weights loaded from {opt.load_weights}")
        except Exception as e:
            print(f"[Warning] Failed to load weights: {e}")
            init_weights(model)
    else:
        print("[Info] Initializing weights from scratch")
        init_weights(model)

    # Model with loss
    model_loss = ModelWithLoss(model, debug=opt.debug).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model_loss.parameters(), 
        lr=opt.lr, 
        weight_decay=opt.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3, 
        verbose=True
    )

    # Mixed precision scaler
    scaler = GradScaler() if opt.use_amp else None

    # TensorBoard
    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # Training state
    best_loss = float('inf')
    best_f1 = 0.0
    best_map = 0.0
    step = 0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Project: {params.project_name}")
    print(f"Output Folder: {output_folder_name}")
    print(f"Model: EfficientDet-D{opt.compound_coef}")
    print(f"Input Size: {input_size}×{input_size}")
    print(f"Batch Size: {opt.batch_size}")
    print(f"Gradient Accumulation: {opt.grad_accumulation_steps}")
    print(f"Learning Rate: {opt.lr}")
    print(f"Epochs: {opt.num_epochs}")
    print(f"Mixup: {'Enabled' if opt.use_mixup else 'Disabled'} (alpha={opt.mixup_alpha})")
    print(f"AMP: {'Enabled' if opt.use_amp else 'Disabled'}")
    print(f"Models Path: {opt.saved_path}")  # ← TAMBAH INI
    print(f"TensorBoard: {opt.log_path}")   # ← TAMBAH INI
    print(f"{'='*70}\n")

    # Mencatat Waktu Start
    training_start_time = time.time()

    try:
        for epoch in range(opt.num_epochs):
            model_loss.train()
            epoch_loss = []
            optimizer.zero_grad()

            pbar = tqdm(training_generator, desc=f'Epoch {epoch+1}/{opt.num_epochs}')

            for iter_num, data in enumerate(pbar):
                try:
                    imgs = data['img'].to(device, non_blocking=True)
                    annot = data['annot'].to(device, non_blocking=True)

                    # Mixup augmentation (optional)
                    if opt.use_mixup and np.random.rand() < 0.5:
                        imgs, annot_a, annot_b, lam = mixup_data(imgs, annot, opt.mixup_alpha)

                        if scaler:
                            with torch.amp.autocast('cuda'):
                                cls_loss_a, reg_loss_a = model_loss(imgs, annot_a)
                                cls_loss_b, reg_loss_b = model_loss(imgs, annot_b)
                                cls_loss = lam * cls_loss_a + (1 - lam) * cls_loss_b
                                reg_loss = lam * reg_loss_a + (1 - lam) * reg_loss_b
                                loss = (cls_loss.mean() + reg_loss.mean()) / opt.grad_accumulation_steps
                            scaler.scale(loss).backward()

                            
                        else:
                            cls_loss_a, reg_loss_a = model_loss(imgs, annot_a)
                            cls_loss_b, reg_loss_b = model_loss(imgs, annot_b)
                            cls_loss = lam * cls_loss_a + (1 - lam) * cls_loss_b
                            reg_loss = lam * reg_loss_a + (1 - lam) * reg_loss_b
                            loss = (cls_loss.mean() + reg_loss.mean()) / opt.grad_accumulation_steps
                            loss.backward()
                    else:
                        # Standard training
                        if scaler:
                            with torch.amp.autocast('cuda'):
                                cls_loss, reg_loss = model_loss(imgs, annot)
                                loss = (cls_loss.mean() + reg_loss.mean()) / opt.grad_accumulation_steps
                            scaler.scale(loss).backward()
                        else:
                            cls_loss, reg_loss = model_loss(imgs, annot)
                            loss = (cls_loss.mean() + reg_loss.mean()) / opt.grad_accumulation_steps
                            loss.backward()

                    # Gradient accumulation
                    if (iter_num + 1) % opt.grad_accumulation_steps == 0:
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss.append(loss.item() * opt.grad_accumulation_steps)
                    pbar.set_postfix({'loss': loss.item() * opt.grad_accumulation_steps})

                    writer.add_scalar('Loss/train_step', loss.item() * opt.grad_accumulation_steps, step)
                    step += 1

                    # Memory cleanup
                    # if iter_num % 50 == 0:
                    #     torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n[Warning] Batch error: {e}")
                    # BERSIHKAN sisa gradien yang tercemar agar tidak merusak batch selanjutnya!
                    optimizer.zero_grad() 
                    continue

            # Jika jumlah total batch tidak habis dibagi grad_accumulation_steps, 
            # eksekusi sisa gradien terakhir agar tidak terbuang.
            if len(training_generator) % opt.grad_accumulation_steps != 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Epoch summary
            avg_loss = np.mean(epoch_loss) if epoch_loss else 0

            # --- FITUR BARU: AUTO-STOP JIKA LOSS NaN ---
            if np.isnan(avg_loss) or np.isinf(avg_loss):
                print(f"\n🚨 [FATAL ERROR] Loss meledak menjadi NaN di Epoch {epoch}!")
                print("🚨 Menghentikan training otomatis untuk mencegah kerusakan model...")
                break # Menghentikan loop training saat itu juga
            # -------------------------------------------

            print(f'\nEpoch {epoch + 1} - Train Loss: {avg_loss:.4f}')
            writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

            val_loss = None
            metrics = None
            current_lr = optimizer.param_groups[0]['lr']

            # Validation
            if epoch % opt.val_interval == 0:
                model_loss.eval()
                val_losses = []

                with torch.no_grad():
                    for data in tqdm(val_generator, desc="Validating", leave=False):
                        imgs = data['img'].to(device)
                        annot = data['annot'].to(device)
                        cls, reg = model_loss(imgs, annot)
                        loss = cls.mean() + reg.mean()
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                print(f'Val Loss: {val_loss:.4f}')
                writer.add_scalar('Loss/val', val_loss, epoch)

                # Update scheduler
                scheduler.step(val_loss)

                # Save best loss model
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'{opt.saved_path}/best_loss_d{opt.compound_coef}.pth')
                    print(f"✅ Best Loss Model Saved: {best_loss:.4f}")
                else:
                    # Menambah kesabaran sesuai dengan lompatan epoch, bukan siklus validasi
                    patience_counter += opt.val_interval

                # Comprehensive evaluation (every N epochs)
                if epoch % opt.eval_interval == 0 or epoch == opt.num_epochs - 1:
                    metrics = evaluate_coco_with_metrics(val_set, model, threshold=0.05, device=device)

                    writer.add_scalar('Metrics/mAP', metrics['mAP'], epoch)
                    writer.add_scalar('Metrics/F1', metrics['F1'], epoch)
                    writer.add_scalar('Metrics/Precision', metrics['precision'], epoch)
                    writer.add_scalar('Metrics/Recall', metrics['recall'], epoch)
                    writer.add_scalar('Metrics/TP', metrics['TP'], epoch)
                    writer.add_scalar('Metrics/FP', metrics['FP'], epoch)
                    writer.add_scalar('Metrics/FN', metrics['FN'], epoch)

                    # Save best F1 model
                    if metrics['F1'] > best_f1:
                        best_f1 = metrics['F1']
                        torch.save(model.state_dict(), f'{opt.saved_path}/best_f1_d{opt.compound_coef}.pth')
                        print(f"✅ Best F1 Model Saved: {best_f1:.4f}")

                    # Save best mAP model
                    if metrics['mAP'] > best_map:
                        best_map = metrics['mAP']
                        torch.save(model.state_dict(), f'{opt.saved_path}/best_map_d{opt.compound_coef}.pth')
                        print(f"✅ Best mAP Model Saved: {best_map:.4f}")

                # Early stopping
                if patience_counter >= opt.es_patience:
                    print(f"\n[Info] Early stopping triggered at epoch {epoch+1}")
                    break
            
            # ✅ BENAR: Tracker update DIPINDAHKAN KE LUAR validation block!
            # Update metrics tracker (every epoch)
            tracker.update(
                epoch=epoch + 1,
                train_loss=avg_loss,
                val_loss=val_loss,  # Will be None if no validation this epoch
                lr=current_lr,
                map_score=metrics['mAP'] if metrics else None,
                f1=metrics['F1'] if metrics else None,
                precision=metrics['precision'] if metrics else None,
                recall=metrics['recall'] if metrics else None,
                tp=metrics['TP'] if metrics else None,
                fp=metrics['FP'] if metrics else None,
                fn=metrics['FN'] if metrics else None
            )

            # --- SIMPAN CSV PER KELAS ---
            # Kita gunakan "if metrics is not None" agar tidak error pada epoch yang tidak ada evaluasinya
            if metrics is not None and 'per_class' in metrics:
                per_class_dict = metrics['per_class']
                
                # Ubah dictionary ke pandas DataFrame
                df_per_class = pd.DataFrame.from_dict(per_class_dict, orient='index')
                df_per_class.reset_index(inplace=True)
                df_per_class.rename(columns={'index': 'Class'}, inplace=True)
                df_per_class['Epoch'] = epoch + 1  # Sesuaikan dengan format epoch tracker
                
                # Simpan ke CSV di folder metrics
                per_class_csv_path = os.path.join(tracker.save_dir, f'per_class_metrics_epoch_{epoch+1}.csv')
                df_per_class.to_csv(per_class_csv_path, index=False)
                print(f"✅ Per-class metrics saved to: {per_class_csv_path}")
            # -----------------------------------------------

            # Checkpoint every N epochs
            if (epoch + 1) % opt.save_interval == 0:
                torch.save(model.state_dict(), f'{opt.saved_path}/checkpoint_d{opt.compound_coef}_epoch{epoch+1}.pth')

            # Memory cleanup
            # torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print('\n[Info] Training interrupted. Saving checkpoint...')
        torch.save(model.state_dict(), f'{opt.saved_path}/interrupted_d{opt.compound_coef}.pth')
    
    finally:
        print("\n" + "="*70)
        print("FINALIZING TRAINING RESULTS")
        print("="*70)

        # 1. Fallback untuk kalkulasi waktu
        training_end_time = time.time()
        # Cek apakah training_start_time sudah terdefinisi (aman jika crash sebelum deklarasi)
        if 'training_start_time' in locals():
            elapsed_seconds = int(training_end_time - training_start_time)
            formatted_duration = str(datetime.timedelta(seconds=elapsed_seconds))
        else:
            formatted_duration = "0:00:00 (Interrupted early)"

        # 2. Fallback untuk epoch
        # Cek apakah variabel 'epoch' sudah ada
        final_epoch = epoch + 1 if 'epoch' in locals() else 0

        # Save CSV and plots (dibungkus try-except agar jika plot gagal, JSON tetap tersimpan)
        try:
            tracker.save_csv()
            tracker.plot_all()
        except Exception as e:
            print(f"[Warning] Failed to save CSV/Plots in finally block: {e}")
        
        # Save summary JSON
        summary = {
            'project': params.project_name,
            'model': f'EfficientDet-D{opt.compound_coef}',
            'total_epochs': final_epoch, # Menggunakan variabel fallback
            'best_val_loss': float(best_loss),
            'best_mAP': float(best_map),
            'best_F1': float(best_f1),
            'final_lr': float(optimizer.param_groups[0]['lr']),
            'end_timestamp': str(datetime.datetime.now()), 
            'total_training_duration': formatted_duration # Menggunakan variabel fallback
        }
        
        try:
            import json
            with open(os.path.join(metrics_dir, 'training_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
            
            print(f"\n✅ All results saved to: {metrics_dir}")
            print(f"  - training_metrics.csv")
            print(f"  - loss_curves.png")
            print(f"  - learning_rate.png")
            print(f"  - performance_metrics.png")
            print(f"  - precision_recall.png")
            print(f"  - detection_metrics.png")
            print(f"  - training_summary.json")
        except Exception as e:
            print(f"[Error] Failed to save summary JSON: {e}")
            
        if 'writer' in locals():
            writer.close()
            
        print(f'\n[Done] Best Loss: {best_loss:.4f} | Best F1: {best_f1:.4f} | Best mAP: {best_map:.4f}')

# ============================================================================
# ARGUMENT PARSER
# ============================================================================
def get_args():
    parser = argparse.ArgumentParser('EfficientDet Optimized Training')

    # Project
    parser.add_argument('-p', '--project', type=str, default='pianika_1')
    parser.add_argument('-c', '--compound_coef', type=int, default=0)

    # Output customization
    parser.add_argument('--output_name', type=str, default='pianika_d0_batch9',
                    help='Custom output folder name (default: project_name)')
    parser.add_argument('--add_timestamp', type=boolean_string, default=False,
                    help='Add timestamp to output folder name')

    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=4e-5)

    # Memory optimization
    parser.add_argument('--grad_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation (effective batch = batch_size * this)')
    parser.add_argument('--use_amp', type=boolean_string, default=True,
                       help='Use mixed precision training')

    # Augmentation
    parser.add_argument('--use_mixup', type=boolean_string, default=False,
                       help='Use Mixup augmentation')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                       help='Mixup alpha parameter')

    # Validation
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='Full evaluation interval (mAP, F1)')

    # Saving
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--es_patience', type=int, default=15,
                       help='Early stopping patience')

    # Paths
    parser.add_argument('--dataset_name', type=str, default='pianika_1', help='Name of the dataset folder')
    parser.add_argument('--data_path', type=str, default='datasets/')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, 
                       default='weights/efficientdet-d0.pth')

    # Others
    parser.add_argument('-n', '--num_workers', type=int, default=4)
    parser.add_argument('--debug', type=boolean_string, default=False)

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    opt = get_args()
    train(opt)