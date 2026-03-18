import streamlit as st
import subprocess
import sys
import os
from PIL import Image

# Configuración
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'images')

SCRIPTS = [
    'detect_and_save_pos_seg.py',
    'extract_osnet_embeddings.py',
    'extract_resnet_embeddings.py',
    'ver_embeddings.py',
    'video_identifica_personas_pos_seg.py',
]

def show_images_from_folder(folder_path, title, max_images=9):
    if not os.path.exists(folder_path):
        st.write(f"Carpeta {title} no encontrada.")
        return
    
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        st.write(f"No hay imágenes en {title}.")
        return
    
    st.subheader(title)
    cols = st.columns(3)
    # For any personaYoloCut folder, find the largest image size
    resize_to = None
    if os.path.basename(folder_path).startswith('personaYoloCut'):
        max_width, max_height = 0, 0
        for img_file in images[:max_images]:
            img_path = os.path.join(folder_path, img_file)
            try:
                img = Image.open(img_path)
                w, h = img.size
                if w > max_width:
                    max_width = w
                if h > max_height:
                    max_height = h
            except Exception:
                pass
        resize_to = (max_width, max_height)

    for i, img_file in enumerate(images[:max_images]):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path)
            # Resize personaYoloCut images to largest found size
            if resize_to:
                img = img.resize(resize_to)
                cols[i % 3].image(img, caption=img_file)
            else:
                cols[i % 3].image(img, caption=img_file)
        except Exception as e:
            cols[i % 3].write(f"Error cargando {img_file}: {e}")

# Crear pestañas
tab1, tab2 = st.tabs(["Ejecutar Pipeline", "Ver Imágenes"])

with tab1:
    st.title("Pipeline de Procesamiento de Video")

    # Opciones
    use_segmentation = st.radio("Tipo de Detección", ["Pose", "Segmentación"]) == "Segmentación"
    model_type = st.selectbox("Modelo de Embeddings", ['osnet_x1_0', 'resnet50', 'se_resnet50'])
    show_crops = st.checkbox("Mostrar Ventanas de Crops")

    # Selector de video
    video_dir = os.path.join(os.path.dirname(__file__), '..', 'video')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    selected_video = st.selectbox("Selecciona el video a procesar", video_files, index=0 if video_files else None)

    if st.button("Ejecutar Pipeline"):
        st.write("Ejecutando pipeline...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        output_area = st.empty()

        output = ""

        for i, script in enumerate(SCRIPTS):
            script_path = os.path.join(SCRIPTS_DIR, script)
            status_text.text(f"Ejecutando: {script}")
            progress_bar.progress((i) / len(SCRIPTS))

            cmd = ['C:\\Python310\\python.exe', script_path]
            if script == 'detect_and_save_pos_seg.py':
                if use_segmentation:
                    cmd.append('--use_segmentation')
            elif script == 'video_identifica_personas_pos_seg.py':
                if use_segmentation:
                    cmd.append('--use_segmentation')
                cmd.extend(['--model_type', model_type])
                if show_crops:
                    cmd.append('--show_crops')
                # Siempre pasar el nombre del video seleccionado
                cmd.extend(['--video_name', selected_video])

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(SCRIPTS_DIR))
                output += f"Ejecutando: {script}\n"
                output += result.stdout
                if result.stderr:
                    output += f"Errores:\n{result.stderr}\n"
                output += "\n"
                if result.returncode != 0:
                    output += f"Error ejecutando {script}\n"
                    break
            except Exception as e:
                output += f"Excepción ejecutando {script}: {str(e)}\n"
                break

        progress_bar.progress(1.0)
        status_text.text("Pipeline completado.")
        output_area.text_area("Salida del Pipeline", output, height=400)

with tab2:
    st.title("Visualizar Imágenes")

    # Un único selector de persona
    persona_indices = [str(i) for i in range(1, 10)]  # Ajusta el rango según el número de personas
    persona_folders = [f"persona{i}" for i in persona_indices if os.path.isdir(os.path.join(IMAGES_DIR, f"persona{i}"))]
    persona_yolo_folders = [f"personaYolo{i}" for i in persona_indices if os.path.isdir(os.path.join(IMAGES_DIR, f"personaYolo{i}"))]
    persona_cut_folders = [f"personaYoloCut{i}" for i in persona_indices if os.path.isdir(os.path.join(IMAGES_DIR, f"personaYoloCut{i}"))]

    # Solo mostrar personas que existen en las tres carpetas
    personas_validas = [i for i in persona_indices if os.path.isdir(os.path.join(IMAGES_DIR, f"persona{i}")) and os.path.isdir(os.path.join(IMAGES_DIR, f"personaYolo{i}")) and os.path.isdir(os.path.join(IMAGES_DIR, f"personaYoloCut{i}"))]
    if not personas_validas:
        st.write("No hay personas con todas las carpetas disponibles.")
    else:
        persona_labels = [f"persona {i}" for i in personas_validas]
        selected_label = st.selectbox("Selecciona persona", persona_labels, index=0)
        selected_persona = selected_label.split()[1]
        show_images_from_folder(os.path.join(IMAGES_DIR, f"persona{selected_persona}"), f"Imágenes Originales (persona{selected_persona})")
        show_images_from_folder(os.path.join(IMAGES_DIR, f"personaYolo{selected_persona}"), f"Imágenes Procesadas con YOLO (personaYolo{selected_persona})")
        show_images_from_folder(os.path.join(IMAGES_DIR, f"personaYoloCut{selected_persona}"), f"Imágenes Recortadas (personaYoloCut{selected_persona})")