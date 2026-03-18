import numpy as np



EMBEDDINGS_FILES = [
    ("osnet_x0_25", r"C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_osnet_x0_25.npy"),
    ("osnet_x0_5", r"C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_osnet_x0_5.npy"),
    ("osnet_x1_0", r"C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_osnet_x1_0.npy"),
    ("resnet50", r"C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_resnet50.npy"),
    ("se_resnet50", r"C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_se_resnet50.npy")
]

for model_name, file_path in EMBEDDINGS_FILES:
    print(f"\n--- {model_name} ---")
    try:
        embeddings = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"No se pudo cargar {file_path}: {e}")
        continue
    print(f"Total de imágenes: {len(embeddings)}\n")
    for i, (img_id, vec) in enumerate(embeddings.items()):
        print(f"ID: {img_id}")
        print(f"Shape: {vec.shape}")
        print(f"Primeros 5 valores: {vec[:5]}")
        print()
        if i >= 4:
            break
