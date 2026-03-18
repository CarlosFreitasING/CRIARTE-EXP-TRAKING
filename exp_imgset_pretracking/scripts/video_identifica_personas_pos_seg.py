import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import torchreid
from torchvision import transforms
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import argparse

# Parsear argumentos
parser = argparse.ArgumentParser(description='Video person identification with pose or segmentation.')
parser.add_argument('--model_type', choices=['osnet_x1_0', 'resnet50', 'se_resnet50'], default='osnet_x1_0', help='Embedding model type')
parser.add_argument('--use_segmentation', action='store_true', help='Use segmentation instead of pose')
parser.add_argument('--show_crops', action='store_true', help='Show crop windows')
parser.add_argument('--video_name', type=str, default='IA2.mp4', help='Nombre del archivo de video a procesar')
args = parser.parse_args()

MODEL_TYPE = args.model_type
USE_SEGMENTATION = args.use_segmentation
SHOW_CROPS = args.show_crops
VIDEO_NAME = args.video_name

seg_or_pose = 'seg' if USE_SEGMENTATION else 'pose'
# Usar el video seleccionado
VIDEO_IN = os.path.join('C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/video', VIDEO_NAME)
# Generar nombre de salida dinámico
nombre_base = os.path.splitext(VIDEO_NAME)[0]
VIDEO_OUT = os.path.join('C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/videoProcesado', f'{nombre_base}_procesado_{seg_or_pose}_{MODEL_TYPE}.mp4')
YOLO_POSE_MODEL = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/yoloModels/yolo26n-pose.pt'
YOLO_SEG_MODEL = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/yoloModels/yolo26n-seg.pt'

if MODEL_TYPE == 'osnet_x1_0':
    EMBEDDING_MODEL_PATH = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeModels/osnet_x1_0.pth'
    EMBEDDINGS_FILE = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_osnet_x1_0.npy'
    MODEL_NAME = 'osnet_x1_0'
elif MODEL_TYPE == 'resnet50':
    EMBEDDING_MODEL_PATH = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeModels/resnet50_reid.pth'
    EMBEDDINGS_FILE = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_resnet50.npy'
    MODEL_NAME = 'resnet50'
elif MODEL_TYPE == 'se_resnet50':
    EMBEDDING_MODEL_PATH = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeModels/se_resnet50_reid.pth'
    EMBEDDINGS_FILE = r'C:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/embeddings/embeddings_se_resnet50.npy'
    MODEL_NAME = 'se_resnet50'

# Cargar modelo YOLO
if USE_SEGMENTATION:
    model = YOLO(YOLO_SEG_MODEL)
else:
    model = YOLO(YOLO_POSE_MODEL)

embedding_model = torchreid.models.build_model(name=MODEL_NAME, num_classes=1000)
embedding_model.load_state_dict(torch.load(EMBEDDING_MODEL_PATH, map_location='cpu'))
embedding_model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_model.to(device)

# Preprocesamiento
if MODEL_TYPE == 'osnet_x1_0':
    # Para OSNet: 256x256 cuadrado con padding
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
else:
    # Para ResNet y SE-ResNet: 256x128 rectangular
    def preprocess(img_bgr, height=256, width=128):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return tfm(img_resized).unsqueeze(0)

embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
emb_keys = list(embeddings.keys())
emb_vecs = np.stack([embeddings[k] for k in emb_keys])

os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

person_colors = [
    (0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255),
    (255,255,0), (128,0,255), (0,128,255), (255,128,0), (128,255,0),
]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame, conf=0.8, verbose=False)
    annotated_img = results[0].plot()
    boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else []
    scores = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else []
    det_embeddings = []
    det_boxes = []
    det_confs = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = scores[i] if len(scores) > i else 0
        if USE_SEGMENTATION and hasattr(results[0], 'masks') and results[0].masks is not None:
            mask_full = results[0].masks.data[i].cpu().numpy().astype(np.uint8) * 255
            if mask_full.shape != frame.shape[:2]:
                mask_full = cv2.resize(mask_full, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            crop = frame[y1:y2, x1:x2]
            crop_mask = mask_full[y1:y2, x1:x2]
            crop_masked = cv2.bitwise_and(crop, crop, mask=crop_mask)
            if SHOW_CROPS:
                cv2.imshow(f'Crop Persona {i+1}', crop_masked)
            input_tensor = preprocess(crop_masked).to(device)
        else:
            crop = frame[y1:y2, x1:x2]
            if SHOW_CROPS:
                cv2.imshow(f'Crop Persona {i+1}', crop)
            input_tensor = preprocess(crop).to(device)
        if crop.size == 0:
            continue
        with torch.no_grad():
            feat = embedding_model(input_tensor)
            vec = feat.squeeze().cpu().numpy().astype(np.float32)
            n = np.linalg.norm(vec)
            if n > 1e-6:
                vec = vec / n
        det_embeddings.append(vec)
        det_boxes.append((x1, y1, x2, y2))
        det_confs.append(conf)
    id_principales = list(sorted(set([k.split('/')[0] for k in emb_keys])))
    sim_matrix = np.zeros((len(det_embeddings), len(id_principales)))
    sim_display = []
    for i, vec in enumerate(det_embeddings):
        sim_list = []
        for j, idp in enumerate(id_principales):
            idxs = [k for k, ek in enumerate(emb_keys) if ek.startswith(idp+'/')]
            if idxs:
                sims = [1 - cosine(vec, emb_vecs[k]) for k in idxs]
                best_sim = max(sims)
            else:
                best_sim = -1
            sim_matrix[i, j] = best_sim
            sim_list.append((idp, best_sim*100))
        sim_display.append(sorted(sim_list, key=lambda x: -x[1]))
    if sim_matrix.size > 0:
        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned = [None]*len(det_embeddings)
        for r, c in zip(row_ind, col_ind):
            if sim_matrix[r, c] > 0.7: # Umbral de similitud
                assigned[r] = (id_principales[c], sim_matrix[r, c])
            else:
                assigned[r] = ('unknown', None)
    else:
        assigned = []
    for i, (box, conf) in enumerate(zip(det_boxes, det_confs)):
        x1, y1, x2, y2 = box
        color = person_colors[i % len(person_colors)]
        if i < len(assigned):
            person_id, sim_val = assigned[i]
            if person_id != 'unknown' and sim_val is not None:
                label = f"{person_id} ({sim_val*100:.1f}%)"
            else:
                label = "unknown"
        else:
            label = "unknown"
        cv2.putText(annotated_img, label, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y_offset = 30
    for idx, sim_list in enumerate(sim_display):
        x_disp = 10 + idx*350
        y_disp = y_offset
        color = person_colors[idx % len(person_colors)]
        cv2.putText(annotated_img, f"Persona {idx+1} similitud:", (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_disp += 25
        for emb_id, pct in sim_list[:18]:
            cv2.putText(annotated_img, f"{emb_id}: {pct:.1f}%", (x_disp, y_disp), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y_disp += 18
    out.write(annotated_img)
    cv2.imshow('Video Identificación', annotated_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Cerrar ventanas de crops para el siguiente frame
    if SHOW_CROPS:
        for i in range(len(boxes)):
            cv2.destroyWindow(f'Crop Persona {i+1}')
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video procesado guardado en: {VIDEO_OUT}")
