*Baca dalam bahasa lain: [Bahasa Indonesia](readme.md)*

# Yet Another EfficientDet PyTorch (Optimized Implementation)

This PyTorch re-implementation of EfficientDet was developed to accurately reproduce Google AutoML's official algorithm. This project has been specifically adopted and optimized to support the [Automated Visual Inspection System](../Software-Inspeksi/), aiming to perform real-time component detection on a musical instrument production line (case study: Pianika).

---

## Performance and Pretrained Weights

Evaluation results on the COCO dataset demonstrate competitive precision levels. The available pretrained weights can be downloaded via the following links:

| Coefficient | Download Link | Img Size | GPU Memory (MB) | FPS | mAP 0.5:0.95 (Official) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 512 | 1049 | 36.20 | 33.8 |
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 640 | 1159 | 29.69 | 39.6 |
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 768 | 1321 | 26.50 | 43.0 |
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 896 | 1647 | 22.73 | 45.8 |

---

## Training Optimization Features

This repository includes dedicated training scripts ([train_d0.py](./train_d0.py)) and ([train_d1.py](./train_d1.py)) featuring the following key enhancements:

* **Memory Management**:
    * **Gradient Accumulation**: Supports large effective batch sizes through gradient accumulation prior to weight updates.
    * **Automatic Mixed Precision (AMP)**: Optimizes speed and memory usage via mixed-precision operations using `torch.amp`.
* **Model Generalization**: Integrates **Mixup Augmentation** to enhance model generalization on novel data variations.
* **Comprehensive Detection Metrics**: Tracks mAP, F1-Score, Precision, Recall, alongside TP, FP, and FN statistics.
* **Automated Visualization**: Automatically generates loss trend, learning rate, and detection performance charts (`.png`).
* **Operational Safety**: Includes an **Auto-stop** mechanism triggered upon detecting `NaN` loss values.

---

## System Specifications (Hardware Reference)

The training and evaluation processes in this repository were conducted using the following hardware specifications:
* **GPU**: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM).
* **Note**: AMP and Gradient Accumulation features are enabled in these scripts to ensure training stability within the 8GB VRAM capacity.

---

## Installation

Follow these steps to set up the development environment:

1. Ensure you are using Python 3.7 or a newer version.
2. Install all dependencies via the [requirements.txt](./requirements.txt) file:
   ```bash
   pip install -r requirements.txt
   ```

---

## Application Context: Automated Visual Inspection

This implementation is targeted at standardizing visual inspection on the production line. The model is configured to recognize various pianika components to verify completeness or detect production defects:

* **Detection Objects**: label, pianika_biru, hose, mouthpiece, case_biru, leaflet, buku_manual, case_pink, pianika_pink.
* **Integration**: This Computer Vision system directly integrates with the supporting software in the [Software-Inspeksi](../Software-Inspeksi/) directory.

---

## Dataset Directory Structure

Ensure your dataset is structured relatively so the scripts can automatically detect file paths:

```text
.
├── datasets/
│   └── pianika_1/
│       ├── train/          # Training images (.jpg)
│       ├── valid/          # Validation images (.jpg)
│       └── annotations/    # COCO format .json files
├── projects/
│   └── pianika.yml         # Project parameter configuration
├── train_d0.py             # EfficientDet-D0 training script
└── train_d1.py             # EfficientDet-D1 training script
```

---

## Training Guide

Use the script corresponding to the model coefficient you wish to train:

* **EfficientDet-D0**: Run [train_d0.py](./train_d0.py)
    ```bash
    python train_d0.py -p pianika --batch_size 2 --grad_accumulation_steps 4 --use_amp True
    ```
* **EfficientDet-D1**: Run [train_d1.py](./train_d1.py)
    ```bash
    python train_d1.py -p pianika --batch_size 2 --grad_accumulation_steps 4 --use_amp True
    ```

The system will automatically save metric reports and performance plots in the `logs/` directory upon completion.

---

## Technical Comparative Analysis

This project addresses several algorithmic discrepancies to align with the original TensorFlow architecture:
1. **Batch Normalization**: System momentum adjustment.
2. **Depthwise-Separable Conv2D**: Accurate BiasAdd placement.
3. **BiFPN Integration**: Precise feature paths and connection weights.
4. **Padding**: Implementation of Static Same Padding.

---

## References
- [google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [signatrix/efficientdet](https://github.com/signatrix/efficientdet)