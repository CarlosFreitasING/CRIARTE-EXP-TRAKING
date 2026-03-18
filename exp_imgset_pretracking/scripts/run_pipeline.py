import subprocess
import sys

# Configuración del pipeline
USE_SEGMENTATION = True  # Usar segmentación en detect_and_save_pos_seg.py y video_identifica_personas_pos_seg.py
MODEL_TYPE = 'se_resnet50'  # Modelo de embeddings: 'osnet_x1_0', 'resnet50', 'se_resnet50'
SHOW_CROPS = False  # Mostrar ventanas de crops en video_identifica_personas_pos_seg.py

# Rutas de scripts
SCRIPTS = [
    'detect_and_save_pos_seg.py',
    'extract_osnet_embeddings.py',
    'extract_resnet_embeddings.py',
    'ver_embeddings.py',
    'video_identifica_personas_pos_seg.py',
]

# Carpeta de scripts
SCRIPTS_DIR = 'c:/Users/Carlo/Desktop/CRIARTE/exp_imgset_pretracking/scripts/'

for script in SCRIPTS:
    script_path = SCRIPTS_DIR + script
    print(f"Ejecutando: {script_path}")
    if script == 'detect_and_save_pos_seg.py':
        if USE_SEGMENTATION:
            result = subprocess.run([sys.executable, script_path, '--use_segmentation'])
        else:
            result = subprocess.run([sys.executable, script_path])
    elif script == 'video_identifica_personas_pos_seg.py':
        cmd = [sys.executable, script_path]
        if USE_SEGMENTATION:
            cmd.append('--use_segmentation')
        cmd.extend(['--model_type', MODEL_TYPE])
        if SHOW_CROPS:
            cmd.append('--show_crops')
        result = subprocess.run(cmd)
    else:
        result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error ejecutando {script}")
        sys.exit(result.returncode)

print("Pipeline completado.")
