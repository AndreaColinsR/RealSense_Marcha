# Configuración del entorno y requerimientos

## Requerimientos
1. Python 3.9/3.10/3.11
2. `librealsense2` (Módulo para controlar las cámaras de Intel. Repositorio fuente [aquí](https://github.com/realsenseai/librealsense)
3. `cv2`
4. `numpy`
5. `mediapipe`
6. `matplotlib`
   
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
pip install numpy opencv-python mediapipe matplotlib notebook pyrealsense2
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
2. Editar el nombre del video y opcionalmente otros parámetros
3. Presionar :fast_forward: en la barra superior
4. El código detectará las cámaras conectadas y grabará un video por 5 segundos o hasta que el usuario presione la tecla **q** en el teclado.
