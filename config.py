"""
config.py
Configuración central del pipeline TFG: rutas, constantes y parámetros
compartidos por todos los módulos.
"""

import os

# ─────────────────────────────────────────────────────────────────
# RUTAS BASE
# ─────────────────────────────────────────────────────────────────
RUTA_DATASET   = r"C:/ensayo correcto TFG/datasetcorrecto1000frames"
RUTA_DATASET_NUEVO = r"C:\ensayo correcto TFG\datasetcorrecto1000frames\dataset_CFD_mejorado_2"
RUTA_CSVS      = os.path.join(RUTA_DATASET, "CSV MATRIZ X y T")
RUTA_BASE_RUNS = r"C:\TFG_pipeline2\runs\pose"


RUTA_MODELO         = os.path.join(RUTA_BASE_RUNS, "train-7", "weights", "best.pt") #train 2 no es habra q cambiarlo
RUTA_MATRICES_CAM   = os.path.join(RUTA_DATASET, "camara_matrices.csv")
RUTA_PRED_CSV       = os.path.join(RUTA_DATASET_NUEVO, "elevacion_CFD_metros.csv") #es la antigua, ahora se llama elevacion_CFD_metros.csv
RUTA_SPECTRUM       = r"C:/seafemTFG.gid/Spectrum.out.dat"
RUTA_GT_ALTURA_CSV  = r"C:/Users/rocio/Repositorio TFG Visual Code/tfe-tfg-rocio-fernandez-ortega/Carpeta Rocio/SCRIPT/alturanodos.csv"
RUTA_DATASET_YAML = os.path.join(RUTA_DATASET_NUEVO, "dataset.yaml")

DIR_SALIDA_FFT_PRED       = os.path.join(RUTA_DATASET, "FFT3D")
DIR_SALIDA_FFT_GT         = os.path.join(RUTA_DATASET, "FFT3D_GT", "NUFFT3D1")
DIR_SALIDA_ENERGIA_GT     = os.path.join(RUTA_DATASET, "energiavsamplitud_GT")
DIR_SALIDA_VIDEOS         = os.path.join(RUTA_DATASET_NUEVO, "VIDEOS")
DIR_SALIDA_METRICAS       = os.path.join(RUTA_DATASET_NUEVO, "METRICAS")
DIR_SALIDA_BEAMFORMING_PRED = os.path.join(RUTA_DATASET,"FFT3D", "BEAMFORMING")
DIR_SALIDA_BEAMFORMING_GT   = os.path.join(RUTA_DATASET,"FFT3D_GT", "BEAMFORMING")
DIR_SALIDA_ENERGIAPRED   = os.path.join(RUTA_DATASET, "ENERGIAVSAMPLITUD")



CARPETAS_IMAGENES_DIVIDIDAS = {
    "train": os.path.join(RUTA_DATASET_NUEVO, "train", "images"),
    "val":   os.path.join(RUTA_DATASET_NUEVO, "val",   "images"),
    "test":  os.path.join(RUTA_DATASET_NUEVO, "test",  "images"),
}



CARPETAS_LABELS_CFD_DIVIDIDAS = {
    "train": os.path.join(RUTA_DATASET_NUEVO, "train", "labels"),
    "val":  os.path.join(RUTA_DATASET_NUEVO, "val",   "labels"),
    "test": os.path.join(RUTA_DATASET_NUEVO, "test",  "labels"),
}

RUTA_Z_METROS_PRED = os.path.join(RUTA_DATASET_NUEVO, "Z_METROS_CFD_MEJORADO_2.csv")# Revisar porque esta mal ahora que he cambiado la ruta del dataset a _2

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS DE IMAGEN / DATASET
# ─────────────────────────────────────────────────────────────────
IMG_W, IMG_H   = 640, 640
NUM_KEYPOINTS  = 24
DELTA_T        = 0.48          # intervalo entre frames (s)
GRID_FILAS     = 3
GRID_COLS      = 8

# Orden exacto de los 24 nodos en la rejilla 3x8 (fila a fila)
LISTA_KEYPOINTS_ORDENADOS = [
    "25607", "25746", "25129", "24440", "23686", "22992", "22719", "21628",
    "27969", "26986", "26284", "25779", "25105", "24168", "23455", "22963",
    "28074", "27486", "26835", "26322", "25671", "25202", "24945", "24036"
]

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS FFT / ESPECTRO
# ─────────────────────────────────────────────────────────────────
FILTRO_FREC_MAX_HZ = 0.33   # = 1/3s, shortest period de SeaFEM
N_BINS_DIR         = 36
N_BINS_T           = 20
T_BIN_MIN, T_BIN_MAX = 3.0, 12.0
UMBRAL_DENOMINADOR = 5.0    # para descartar denominadores casi singulares en pixel->metro

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS DE ENTRENAMIENTO (solo si isTrainingRequired = True)
# ─────────────────────────────────────────────────────────────────
MODELO_BASE   = "yolo26s-pose.pt"
EPOCHS        = 1000
BATCH_SIZE    = 8
DEVICE        = 0
PATIENCE      = 0

# ─────────────────────────────────────────────────────────────────
# PARÁMETROS DE VIDEO
# ─────────────────────────────────────────────────────────────────
VIDEO_FPS    = 10
VIDEO_ESCALA = 1.0
COLOR_GT     = (255, 0, 0)     # azul BGR
COLOR_PRED   = (0, 0, 255)     # rojo BGR
GROSOR_LINEA = 1
RADIO_NODO   = 4
