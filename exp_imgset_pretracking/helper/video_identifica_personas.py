import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

# Rutas
VIDEO_IN = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/video/IAALL.mp4'
VIDEO_OUT = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/videoProcesado/IAALL_procesado.mp4'
YOLO_MODEL = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/yoloModels/yolo26n-pose.pt'
OSNET_MODEL = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/osnetembeModelst_x1_0.pth'
EMBEDDINGS_FILE = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_osnet_x1_0.npy'

# Cargar modelo YOLO pose
model = YOLO(YOLO_MODEL)

osnet = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000)
osnet.load_state_dict(torch.load(OSNET_MODEL, map_location='cpu'))
osnet.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
osnet.to(device)

# Preprocesamiento igual que en extract_osnet_embeddings.py
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

# Cargar embeddings
embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
emb_keys = list(embeddings.keys())
emb_vecs = np.stack([embeddings[k] for k in emb_keys])

# Crear carpeta de salida si no existe
os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)

# Abrir video
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))


while True:

    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.7, verbose=False)
    annotated_img = results[0].plot()
    boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
    scores = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else []
    # ====== ASIGNACIÓN ÓPTIMA DE IDS PRINCIPALES ======
    # 1. Extraer embeddings de cada persona detectada
    det_embeddings = []
    det_boxes = []
    det_confs = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = scores[i] if len(scores) > i else 0
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        # Preprocesar y extraer embedding
        input_tensor = preprocess(crop).to(device)
        with torch.no_grad():
            feat = osnet(input_tensor)
            vec = feat.squeeze().cpu().numpy().astype(np.float32)
            n = np.linalg.norm(vec)
            if n > 1e-6:
                vec = vec / n
        det_embeddings.append(vec)
        det_boxes.append((x1, y1, x2, y2))
        det_confs.append(conf)

    # 2. Extraer ids principales únicos de los embeddings base
    id_principales = list(sorted(set([k.split('/')[0] for k in emb_keys])))
    # 3. Para cada persona detectada y cada id principal, obtener la mejor similitud
    sim_matrix = np.zeros((len(det_embeddings), len(id_principales)))
    sim_display = []
    for i, vec in enumerate(det_embeddings):
        sim_list = []
        for j, idp in enumerate(id_principales):
            # Buscar todos los embeddings de ese id principal
            idxs = [k for k, ek in enumerate(emb_keys) if ek.startswith(idp+'/')]
            if idxs:
                sims = [1 - cosine(vec, emb_vecs[k]) for k in idxs]
                best_sim = max(sims)
            else:
                best_sim = -1
            sim_matrix[i, j] = best_sim
            sim_list.append((idp, best_sim*100))
        sim_display.append(sorted(sim_list, key=lambda x: -x[1]))


    # 4. Asignación óptima usando Hungarian
    if sim_matrix.size > 0:
        cost_matrix = -sim_matrix  # Maximizar similitud = minimizar -similitud
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned = [None]*len(det_embeddings)
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] > 0.8:  # Umbral de similitud
                assigned[r] = (id_principales[c], sim_matrix[r, c])
            else:
                assigned[r] = ('unknown', None)
    else:
        assigned = []

    # 5. Dibujar resultados
    for i, (box, conf) in enumerate(zip(det_boxes, det_confs)):
        x1, y1, x2, y2 = box
        if i < len(assigned):
            person_id, sim_val = assigned[i]
            if person_id != 'unknown' and sim_val is not None:
                label = f"{person_id} ({sim_val*100:.1f}%)"
            else:
                label = "unknown"
        else:
            label = "unknown"
        cv2.putText(annotated_img, label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # Mostrar la lista de similitudes en la esquina superior izquierda
    y_offset = 30
    for idx, sim_list in enumerate(sim_display):
        x_disp = 10 + idx*350  # Separar si hay varias personas
        y_disp = y_offset
        cv2.putText(annotated_img, f"Persona {idx+1} similitud:", (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y_disp += 25
        for emb_id, pct in sim_list[:18]:
            cv2.putText(annotated_img, f"{emb_id}: {pct:.1f}%", (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_disp += 18

    out.write(annotated_img)
    # Mostrar ventana en tiempo real
    cv2.imshow('Video Identificación', annotated_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video procesado guardado en: {VIDEO_OUT}")
