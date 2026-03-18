import os
import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv26 pose desde la nueva ruta
model = YOLO(r'C:\Users\Carlo\Desktop\CRIARTE\exp_imgset_pretracking\yoloModels\yolo26n-pose.pt')

# Directorio de imágenes
images_dir = r'C:\Users\Carlo\Desktop\CRIARTE\exp_imgset_pretracking\images'


# Iterar sobre las carpetas persona1, persona2, etc.

for folder in os.listdir(images_dir):
    if folder.startswith('persona') and folder[7:].isdigit() and not 'Yolo' in folder and os.path.isdir(os.path.join(images_dir, folder)):
        # Extraer el número de la carpeta
        num = folder.replace('persona', '')
        # Crear la nueva carpeta personaYolo1, personaYolo2, etc. en la misma carpeta images
        new_folder = f'personaYolo{num}'
        new_folder_path = os.path.join(images_dir, new_folder)
        os.makedirs(new_folder_path, exist_ok=True)
        # Crear la carpeta para los recortes en la misma carpeta images
        cut_folder = f'personaYoloCut{num}'
        cut_folder_path = os.path.join(images_dir, cut_folder)
        os.makedirs(cut_folder_path, exist_ok=True)

        # Iterar sobre las imágenes en la carpeta
        for img_file in os.listdir(os.path.join(images_dir, folder)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, folder, img_file)

                # Realizar la predicción con YOLO
                results = model.predict(img_path, conf=0.1)  # Umbral de confianza

                # Usar plot() de YOLO para dibujar automáticamente bounding boxes, keypoints y confianza
                annotated_img = results[0].plot()

                # Guardar la imagen anotada
                save_path = os.path.join(new_folder_path, img_file)
                cv2.imwrite(save_path, annotated_img)

                # Leer la imagen original para recortar
                original_img = cv2.imread(img_path)
                boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
                if len(boxes) > 0:
                    # Si hay varias personas, tomar la de mayor confianza
                    scores = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else [1.0]*len(boxes)
                    best_idx = int(scores.argmax()) if hasattr(scores, 'argmax') else 0
                    x1, y1, x2, y2 = map(int, boxes[best_idx])
                    crop = original_img[y1:y2, x1:x2]
                    crop_save_path = os.path.join(cut_folder_path, img_file)
                    cv2.imwrite(crop_save_path, crop)

print("Procesamiento completado.")