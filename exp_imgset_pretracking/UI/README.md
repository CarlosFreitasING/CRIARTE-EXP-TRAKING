# UI para Pipeline de Procesamiento de Video

Esta interfaz gráfica permite controlar el pipeline de procesamiento de video usando Streamlit.

## Instalación

1. Asegúrate de tener Python instalado.
2. Crea un entorno virtual (si no existe):
   ```
   python -m venv venv
   ```
3. Activa el entorno virtual:
   ```
   venv\Scripts\activate
   ```
4. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Uso

1. Activa el entorno virtual (si no está activado):
   ```
   venv\Scripts\activate
   ```
2. Ejecuta la aplicación:
   ```
   python -m streamlit run app.py
   ```

- Selecciona el tipo de detección: Pose o Segmentación.
- Elige el modelo de embeddings: osnet_x1_0, resnet50, o se_resnet50.
- Marca la casilla si quieres mostrar las ventanas de crops.
- Haz clic en "Ejecutar Pipeline" para correr los scripts con la configuración seleccionada.
- La salida del pipeline se mostrará en el área de texto.