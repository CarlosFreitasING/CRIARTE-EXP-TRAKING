import os
import cv2
import numpy as np
import torch

import torchreid
from torchvision import transforms
import pathlib

# Ruta local de modelos OSNet
osnet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeModels'))
os.makedirs(osnet_dir, exist_ok=True)

# Modelos OSNet a usar
osnet_models = {
    "osnet_x0_25": None,
    "osnet_x0_5": None,
    "osnet_x1_0": None
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelos

for name in osnet_models:
    local_pth = os.path.join(osnet_dir, f"{name}.pth")
    if os.path.exists(local_pth):
        print(f"Cargando modelo local {local_pth}...")
        model = torchreid.models.build_model(name=name, num_classes=1000)
        state_dict = torch.load(local_pth, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        osnet_models[name] = model
        print(f"Modelo {name} cargado desde archivo local.")
    else:
        print(f"Modelo local {local_pth} no encontrado. Descargando con torchreid...")
        model = torchreid.models.build_model(name=name, num_classes=1000, pretrained=True)
        model.eval().to(device)
        osnet_models[name] = model
        print(f"Modelo {name} descargado y cargado.")
print("Modelos cargados.")

# Preprocesamiento estándar para OSNet
def preprocess(img_bgr, size=256):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    img_res = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    img_sq = cv2.copyMakeBorder(img_res, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img_sq).unsqueeze(0)

# Diccionario para guardar los embeddings
embeddings = {model_name: {} for model_name in osnet_models}

images_dir = r"C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/images"
for folder in os.listdir(images_dir):
    if folder.startswith("personaYoloCut") and os.path.isdir(os.path.join(images_dir, folder)):
        folder_path = os.path.join(images_dir, folder)
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                input_tensor = preprocess(img)
                for model_name, model in osnet_models.items():
                    with torch.no_grad():
                        feat = model(input_tensor.to(device))
                        vec = feat.squeeze().cpu().numpy().astype(np.float32)
                        n = np.linalg.norm(vec)
                        if n > 1e-6:
                            vec = vec / n
                        # Guardar: clave = carpeta/imagen
                        key = f"{folder}/{img_file}"
                        embeddings[model_name][key] = vec

# Guardar los embeddings en archivos npy en la carpeta embeddings
embeddings_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeddings'))
os.makedirs(embeddings_dir, exist_ok=True)
for model_name, model_dict in embeddings.items():
    out_path = os.path.join(embeddings_dir, f"embeddings_{model_name}.npy")
    np.save(out_path, model_dict)
    print(f"Embeddings guardados en {out_path}")

print("¡Embeddings generados y guardados!")
