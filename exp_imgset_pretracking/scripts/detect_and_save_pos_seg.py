# Este script permite elegir entre YOLO pose y YOLO segmentación para recortar personas de imágenes.
# Uso: python detect_and_save_pos_seg.py --use_segmentation True/False

import os
import cv2
from ultralytics import YOLO
import numpy as np
import argparse

# Parsear argumentos
parser = argparse.ArgumentParser(description='Detect and save persons with pose or segmentation.')
parser.add_argument('--use_segmentation', action='store_true', help='Use segmentation instead of pose')
args = parser.parse_args()

USE_SEGMENTATION = args.use_segmentation

# Modelos
YOLO_POSE_PATH = r'C:\Users\Carlo\Desktop\CRIARTE\exp_imgset_pretracking\yoloModels\yolo26x-pose.pt'
YOLO_SEG_PATH = r'C:\Users\Carlo\Desktop\CRIARTE\exp_imgset_pretracking\yoloModels\yolo26x-seg.pt'

# Cargar el modelo adecuado
if USE_SEGMENTATION:
    model = YOLO(YOLO_SEG_PATH)
else:
    model = YOLO(YOLO_POSE_PATH)

# Directorio de imágenes
images_dir = r'C:\Users\Carlo\Desktop\CRIARTE\exp_imgset_pretracking\images'

for folder in os.listdir(images_dir):
    if folder.startswith('persona') and folder[7:].isdigit() and not 'Yolo' in folder and os.path.isdir(os.path.join(images_dir, folder)):
        num = folder.replace('persona', '')
        new_folder = f'personaYolo{num}'
        new_folder_path = os.path.join(images_dir, new_folder)
        os.makedirs(new_folder_path, exist_ok=True)
        cut_folder = f'personaYoloCut{num}'
        cut_folder_path = os.path.join(images_dir, cut_folder)
        os.makedirs(cut_folder_path, exist_ok=True)

        for img_file in os.listdir(os.path.join(images_dir, folder)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, folder, img_file)
                # Leer imagen ignorando warnings de color
                original_img = cv2.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
                if original_img is None:
                    print(f"No se pudo cargar {img_path}")
                    continue
                results = model.predict(img_path, conf=0.1, classes=[0])  # Solo detectar personas (clase 0)
                # Usar plot() de YOLO para dibujar automáticamente bounding boxes, keypoints y confianza
                annotated_img = results[0].plot()
                # Guardar la imagen anotada
                save_path = os.path.join(new_folder_path, img_file)
                cv2.imwrite(save_path, annotated_img)
                boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
                if len(boxes) > 0:
                    scores = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else [1.0]*len(boxes)
                    best_idx = int(scores.argmax()) if hasattr(scores, 'argmax') else 0
                    x1, y1, x2, y2 = map(int, boxes[best_idx])
                    if USE_SEGMENTATION and hasattr(results[0], 'masks') and results[0].masks is not None:
                        # Usar la máscara de segmentación en tamaño completo y alinear con el recorte
                        mask_full = results[0].masks.data[best_idx].cpu().numpy().astype(np.uint8) * 255  # (H, W)
                        # Asegurarse que la máscara tenga el mismo tamaño que la imagen original
                        if mask_full.shape != original_img.shape[:2]:
                            mask_full = cv2.resize(mask_full, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                        # Recortar la máscara y la imagen en la misma región
                        crop = original_img[y1:y2, x1:x2]
                        crop_mask = mask_full[y1:y2, x1:x2]
                        crop_masked = cv2.bitwise_and(crop, crop, mask=crop_mask)
                        crop_save_path = os.path.join(cut_folder_path, img_file)
                        cv2.imwrite(crop_save_path, crop_masked)
                    else:
                        crop = original_img[y1:y2, x1:x2]
                        crop_save_path = os.path.join(cut_folder_path, img_file)
                        cv2.imwrite(crop_save_path, crop)

print("Procesamiento completado.")
