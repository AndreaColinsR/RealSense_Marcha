# Configuración del entorno y requerimientos

## Requerimientos
1. Python 3.9/3.10/3.11
2. `librealsense2` (Módulo para controlar las cámaras de Intel. Repositorio fuente [aquí](https://github.com/realsenseai/librealsense))
3. `cv2`
4. `numpy`
5. `mediapipe`
6. `matplotlib`
7. `scipy`
   
   Las versiones de Python son un requisito dado por librealsense2 y mediapipe para ser instaladas usando pip.
   Si se usa una versión más reciente de Python, librealsense2 debe ser construido desde el repositorio fuente. Mediapipe solo puede ser usado hasta Python 3.12

---

## Configuración recomendada (Windows)

Lo siguiente no es estrictamente necesario, pero es altamente recomendable si tienes varias versiones de Python instaladas.

### 1. Instalar Python

Descargar la versión escogida de Python (e.j. 3.10) desde la página oficial:  
https://www.python.org/downloads/release/python-3100/

Durante la instalación:
- Marcar la opción **"Add Python to PATH"**

---

### 2. Verificar instalación

Abrir una terminal (CMD o PowerShell):

```bash
python --version
```

Debería mostrar algo como

```bash
Python 3.10.x
```
### 3. Crear un ambiente virtual donde se utilice sólo la versión escogida de Python

En el siguiente ejemplo se crea el entorno virtual Marcha_RealSense que usa Python 3.10

```bash
py -3.10 -m venv Marcha_RealSense
```
Activamos el ambiente virtual 

```bash
.\Marcha_RealSense\Scripts\Activate.ps1
```

### 4. Instalar todas las dependencias en el ambiente virtual nombradas arriba, más Jupyter Notebook

```bash
pip install numpy opencv-python matplotlib notebook pyrealsense2 scipy
```

Para instalar mediapipe, instalar la siguiente versión
```bash
pip install mediapipe==0.10.9
```

3. Si se usa un ambiente virtual, entonces antes de ejecutar el código se debe activar el ambiente virtual

# Stream desde dos cámaras simultáneamente
 
1. Comenzar Jupyter Notebook
2. Navegar hasta la carpeta donde están los notebooks descargados
3. Conectar ambas cámaras usando el conector USBC-USBA. Este paso es crucial: Las cámaras deben estar conectadas a un puerto USB 3.0 o superior y el conector debe tener una capacidad de transferencia de datos mayor a un GB. Actualmente usamos conectores de velocidad máxima de transferencia de 5 GB.
4. Abrir el Notebook RealSense_Stream2Cameras.ipynb
5. Presionar :fast_forward: en la barra superior
6. El código detectará las cámaras conectadas y mostrará las imágenes por 60 segundos o hasta que el usuario presione la tecla **q** en el teclado

# Grabar desde dos cámaras simultáneamente

1. Seguir los pasos 1 hasta 3 de la sección anterior
2. Abrir el Notebook RealSense_RecordVideo.ipynb
3. Editar el nombre del video y opcionalmente otros parámetros
4. Presionar :fast_forward: en la barra superior
5. El código detectará las cámaras conectadas y grabará un video por 5 segundos o hasta que el usuario presione la tecla **q** en el teclado.

# Calibrar dos cámaras para reconstrucción 3D
1. Posicionar un tablero de ajedrez de 9x7 cuadrados en la región donde caminarán los pacientes. El tablero completo debe ser visible desde las dos cámaras, como en las figuras de ejemplo.
   
<img width="50%" alt="Screenshot 2026-04-21 125415" src="https://github.com/user-attachments/assets/63ed56d9-c3f3-4d52-8002-86bfbb536d85" />
<img width="50%" alt="Screenshot 2026-04-21 125436" src="https://github.com/user-attachments/assets/abee5e0b-19e8-4ed4-a172-c7c81c0dde4b" />


2. Grabar un video desde las dos cámaras simultáneamente. Ver la sección anterior de instrucciones.
3. Abrir el Notebook Calibrate_cameras.ipynb
4. Presionar :fast_forward: en la barra superior
5. Al terminar, el notebook mostrará las esquinas del tablero detectadas en la última imagen y reconstruidas de la calibración de las cámaras. Es de vital importancia verificar que exista una correspondencia entre los puntos detectados en ambas cámaras.
6. Finalmente, la última celda del notebook mostrará la reconstrucción en 3D de los puntos del tablero. Es de vital importancia verificar que los puntos correspondan a la figura en 3 dimensiones. Por ejemplo, los puntos deben yacer en un plano formando líneas paralelas que marcan las filas del tablero.

#  Detectar los puntos del esqueleto y reconstruir en 3D

1. Abrir una ventana de command prompt ( por ej., escribiendo cmd en la barra de búsqueda de Windows).
2. Activar el ambiente virtual

```bash
.\Marcha_RealSense\Scripts\Activate.ps1
```
3. Navegar hasta la carpeta donde está el código y las grabaciones.

```bash
cd .\Documentos\Marcha
```
4. Verificar que el archivo bodypose3d.py contenga el nombre del video que se quiere analizar y el archivo de calibración que se desea ocupar para este video.
5. Ejecutar el archivo bodypose3d.py
    
```bash
   python bodypose3d.py
```

6. Una vez finalizado sin errores, se mostrará por pantalla el mensaje "All done". Por favor note que este código asume que existe una subcarpeta Tracking donde se guardarán los resultados.

# Mostrar resultados de reconstrucción en 3D.
1. Verificar que el archivo show_3d_pose.py contenga el nombre del video que se quiere mostrar
2. En la misma ventana de command prompt (con el ambiente virtual activo) ejecutar el archivo.

```bash
   python show_3d_pose.py
```
