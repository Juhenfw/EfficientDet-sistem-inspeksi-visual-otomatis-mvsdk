*Baca dalam bahasa lain: [Bahasa Indonesia](README.md)*

# Automated Visual Inspection System Based on EfficientDet

This repository contains the implementation of the Undergraduate Thesis titled **"Implementation of EfficientDet for Component Detection and Completeness Classification of Melodica Musical Instruments in the Inspection System at PT. XYZ"**.

This system is designed to automate quality control during the final packaging stage of melodica products to minimize the risk of identification errors due to human fatigue.

## Project Summary
This research develops an automated visual inspection system based on dual cameras using the EfficientDet Deep Learning architecture to detect nine component categories and classify package completeness in real-time.

### Detection Object Categories:
* **Accessories**: Hose, Mouthpiece, Label, Manual Book, and Leaflet.
* **Main Unit**: Melodica (Blue/Pink) and Case (Blue/Pink).

## Main System Features
* **Dual-Station Integration**: The system is designed in accordance with industrial SOPs, comprising Station 1 (accessory verification) and Station 2 (main unit and case verification).
* **Spatial Logic (IoU)**: Besides object detection, the system applies a spatial logic filter based on Intersection over Union (IoU) with a threshold of 0.3 to ensure component layout precision meets company standards.
* **Data-Centric Approach**: Utilizes an object isolation strategy and hard negative samples (empty containers) to improve model robustness against light reflections and visual disturbances on the factory floor.
* **High-Speed Performance**: Average system response time (latency) is around 0.55 to 0.56 seconds, meeting the industrial productivity target of under 2 seconds per package.

## Performance and Experimental Results
Based on internal evaluation of 432 validation image samples:
* **mAP@0.50**: 0.9932.
* **F1-Score**: 0.9927.

Based on evaluation of 2,441 operational image samples obtained directly from the field:
* **Best Model Variant**: EfficientDet-D1.
* **Field Accuracy**: 99.4% (Accessory Station) and 97.8% (Main Unit Station).

## Repository Structure
This project is divided into two main functional modules:

1.  **[Training-EfficientDet](./Training-EfficientDet/)**
    Contains the AI model development infrastructure, including adaptive data acquisition strategies, optimized training scripts with Automatic Mixed Precision (AMP), and Gradient Accumulation.
2.  **[Software-Inspeksi](./Software-Inspeksi/)**
    Operational software package that includes industrial camera calibration tools, region of interest (ROI) configuration, and the main graphical user interface (GUI) based on *CustomTkinter*.

## System Specifications (Hardware Reference)
* **Cameras**: 2 Industrial Cameras HT-SUA501GC-TIV-C (2/3" CMOS Sensor, 5MP, 40 FPS).
* **Processing Unit**: Laptop/Mini PC (Intel Core i7, 16GB RAM, NVIDIA RTX 4060 8GB GPU).

## Citation
If you use the code or research results from this repository, please provide attribution in the following format:

> **Bahasa Indonesia:**
> Wildan, J. F. (2026). Implementasi EfficientDet untuk Deteksi Komponen dan Klasifikasi Kelengkapan Alat Musik Pianika pada Sistem Inspeksi di PT. XYZ. Skripsi. Surabaya: Universitas Airlangga.
> 
> **English:**
> Wildan, J. F. (2026). Implementation of EfficientDet for Component Detection and Completeness Classification of Melodica Musical Instruments in the Inspection System at PT. XYZ. Undergraduate Thesis. Surabaya: Universitas Airlangga.

## Author
**Juhen Fashikha Wildan**<br>
Bachelor Program in Robotics and Artificial Intelligence Engineering<br>
Faculty of Advanced Technology and Multidiscipline<br>
**Universitas Airlangga**

---
*This research was supported by PT. XYZ as part of efforts to improve the efficiency of the packaging process and product quality control.*