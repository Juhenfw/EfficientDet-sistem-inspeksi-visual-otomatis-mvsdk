import os
import json
import glob
import cv2  # pip install opencv-python

def custom_json_to_coco(json_files, output_json_path, image_folder_path):
    # 1. Setup Output
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 2. Class Mapping (Sesuai request Anda)
    class_mapping = {
        "label": 1,
        "pianika_biru": 2,
        "hose": 3,
        "mouthpiece": 4,
        "case_biru": 5,
        "leaflet": 6,
        "buku_manual": 7,
        "case_pink": 8,
        "pianika_pink": 9
    }
    
    for class_name, class_id in class_mapping.items():
        coco_output["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "none"
        })

    annotation_id = 1
    image_id = 1

    print(f"🔄 Memproses {len(json_files)} file JSON...")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Data adalah LIST, ambil item pertama
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
            else:
                print(f"⚠️ Format salah/kosong: {json_file}")
                continue

            # --- A. Info Gambar ---
            # Nama file di json: "HT...BMP". Kita cari file aslinya di folder.
            original_filename = item.get("image", "")
            
            # Cek apakah file aslinya JPG atau BMP di folder
            # Kita coba cari file dengan nama base yang sama
            base_name_no_ext = os.path.splitext(original_filename)[0]
            
            # Cari file gambar di folder (bisa jpg, jpeg, png, bmp)
            image_path = None
            final_filename = original_filename
            
            for ext in [".jpg", ".JPG", ".jpeg", ".bmp", ".BMP", ".png"]:
                potential_path = os.path.join(image_folder_path, base_name_no_ext + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    final_filename = base_name_no_ext + ext
                    break
            
            # Wajib baca gambar untuk dapat width/height karena JSON tidak punya info itu
            height, width = 0, 0
            if image_path:
                img = cv2.imread(image_path)
                if img is not None:
                    height, width, _ = img.shape
                else:
                    print(f"⚠️ Gagal baca gambar opencv: {image_path}")
            else:
                # Jika gambar tidak ketemu, skip file ini (karena COCO wajib punya width/height)
                print(f"⚠️ Gambar tidak ditemukan untuk JSON: {json_file}")
                continue

            image_info = {
                "id": image_id,
                "file_name": final_filename, # Gunakan nama file yang benar-benar ada
                "width": width,
                "height": height
            }
            coco_output["images"].append(image_info)

            # --- B. Annotations ---
            annotations = item.get("annotations", [])
            for ann in annotations:
                label = ann.get("label")
                coords = ann.get("coordinates", {})
                
                if label not in class_mapping:
                    continue

                # Ambil coord: CreateML biasanya formatnya CENTER (x,y,width,height)
                # COCO butuh format TOP-LEFT (xmin, ymin, width, height)
                c_x = coords.get("x", 0)
                c_y = coords.get("y", 0)
                w_box = coords.get("width", 0)
                h_box = coords.get("height", 0)
                
                # Konversi Center -> TopLeft
                # Rumus: xmin = cx - (w/2)
                xmin = c_x - (w_box / 2)
                ymin = c_y - (h_box / 2)
                
                area = w_box * h_box

                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_mapping[label],
                    "bbox": [xmin, ymin, w_box, h_box],
                    "area": area,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(annotation_info)
                annotation_id += 1
            
            image_id += 1

        except Exception as e:
            print(f"❌ Error {os.path.basename(json_file)}: {e}")

    with open(output_json_path, "w") as f:
        json.dump(coco_output, f, indent=4)
    print(f"✅ Selesai! Disimpan ke: {output_json_path}\n")

# --- KONFIGURASI ---
base_path = r"D:\VSCODE\EfficientDet-Pytorch\datasets\pianika_1"

# Folder gambar (Train & Valid)
train_dir = os.path.join(base_path, "train")
val_dir = os.path.join(base_path, "valid")

# List JSON Files
train_jsons = glob.glob(os.path.join(train_dir, "*.json"))
val_jsons = glob.glob(os.path.join(val_dir, "*.json"))

# Run
if train_jsons:
    custom_json_to_coco(train_jsons, os.path.join(base_path, "annotations", "instances_train.json"), train_dir)

if val_jsons:
    custom_json_to_coco(val_jsons, os.path.join(base_path, "annotations", "instances_valid.json"), val_dir)