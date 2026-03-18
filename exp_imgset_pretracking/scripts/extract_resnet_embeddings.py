import os
import cv2
import numpy as np
import torch

import torchreid
from torchvision import transforms
import pathlib

# Ruta local de modelos ResNet/SE-ResNet
embe_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../embeModels'))
os.makedirs(embe_dir, exist_ok=True)

# Modelos ResNet a usar
resnet_models = {
    "resnet50": None,
    "se_resnet50": None
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelos

for name in resnet_models:
    local_pth = os.path.join(embe_dir, f"{name}_reid.pth")
    if os.path.exists(local_pth):
        print(f"Cargando modelo local {local_pth}...")
        model = torchreid.models.build_model(name=name, num_classes=1000)
        state_dict = torch.load(local_pth, map_location=device)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        resnet_models[name] = model
        print(f"Modelo {name} cargado desde archivo local.")
    else:
        print(f"Modelo local {local_pth} no encontrado. Descargando con torchreid...")
        model = torchreid.models.build_model(name=name, num_classes=1000, pretrained=True)
        model.eval().to(device)
        resnet_models[name] = model
        print(f"Modelo {name} descargado y cargado.")
print("Modelos cargados.")

# Preprocesamiento para ResNet/SE-ResNet (256x128, no cuadrado)
def preprocess(img_bgr, height=256, width=128):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(img_resized).unsqueeze(0)

# Diccionario para guardar los embeddings
embeddings = {model_name: {} for model_name in resnet_models}

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
                for model_name, model in resnet_models.items():
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
