# TFG_PIPELINE2

Explicación del uso de los diferentes módulos y scripts en general del pipeline

## Estructura
- `config.py`: Aqui se ubican todas las direcciones a archivos que necesitan los diferentes códigos. Tanto rutas de entradas como de salidas, rutas a carpetas, números identificadores  de keypoints, los parámetros con los que entrena el modelo de YOLO y los parámetros de la camara de Blender.

- `main.py`: Código que ejecuta los diferentes modulos. Tiene booleanas para que se pueda poner True o False según si se quiere utilizar o no los diferentes modulos. Importante: si se quiere activar isBeamformingPred o isBeamformingGT tambien se tiene que activar (True) isBeamformingRequired. Se encuentra entre las lineas de código 35-47 en el apartado CONTROL y tiene una breve explicación de lo que hace cada uno.
Además, esta dividido por pasos los diferentes modulos.

- `modules/`: Módulos del pipeline. Scripts donde se desarrolla el código que determina el comportamiento de cada etapa del pipeline.

## Requisitos
- Python 3.10+
- Instalar dependencias desde el entorno virtual.

## Uso
1. Activar el entorno virtual.
2. Ejecutar `python main.py`.

## entrenamiento.py
Aquí es donde se entrena el modelo de YOLO, sería el primer paso a realizar si se quiere comenzar de nuevo con otro CFD.

## prediccion_videos.py
Genera los videos del GT y la predicción del modelo YOLO y las compara.

## camara.py 
Es el script que se ejecuta dentro de Blender para extraer las matrices extrinsecas e intrinsecas

## pixel_a_metros.py
Es el script donde están las matrices extrinsecas e intrinsecas para pasar de pixeles (u,v) a coordenadas en metro.

## lectura_gt.py
Script que busca los keypoints seleccionados en el csv alturanodos.csv donde se extrae la elevacion de los nodos directamente del SEAFEM (groundtruth).

## fft_analisis.py 
Script donde se calcula la FFT 1D + 3D (FINUFFT)

## beamforming.py 
Hace el calculo de el desfase de la fase para encontrar la dirección de las olas, es el último método que he probado al ver que FINUFFT tampoco me daba bien.

## energia_gt.py
Script donde se calcula y grafica las gráficas de energia S vs frecuencia (w) y E vs A del groundtruth.

## energiapred.py
Script donde se calcula y grafica las gráficas de energia S vs frecuencia (w) y E vs A a través de la predicción del modelo una vez pasado a coordenadas en metros gracias a camara.py. El csv se llama z_metros.csv.

## espectro_polar.py 
Script que genera el espectro polar tanto del GT como de la predicción. Se debe de especificar en main.py en la parte de CONTROL.



