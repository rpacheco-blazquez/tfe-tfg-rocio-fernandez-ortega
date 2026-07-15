"""
main.py
Orquestador del pipeline completo del TFG. Activa o desactiva cada paso
cambiando las variables booleanas de la sección de CONTROL.

Orden lógico del pipeline:
  1. Entrenamiento del modelo YOLO-Pose             (entrenamiento.py)
  2. Predicción sobre train/val/test + vídeos        (prediccion_videos.py)
  3. Extracción de matrices de cámara (Blender)       (camara.py)        [manual, fuera de aquí]
  4. Conversión píxel -> metros (predicción)          (pixel_a_metros.py)
  5. FFT 1D + FFT 3D + espectro polar de PREDICCIÓN   (fft_analisis.py, espectro_polar.py)
  6. FFT 1D + FFT 3D + espectro polar de GROUNDTRUTH  (lectura_gt.py, fft_analisis.py, espectro_polar.py)
  7. Gráficas de energía E vs A y S(w) vs w del GT     (energia_gt.py)
  8. Gráficas de energía E vs A y S(w) vs w de la PRED  (energia_pred.py)
"""

import os
import numpy as np

import config
import modules
from modules import (
    energia_gt_alturanodos,
    energiapred,
    entrenamiento,
    prediccion_videos,
    pixel_a_metros,
    fft_analisis,
    espectro_polar,
    energia_gt,
    lectura_gt,
)

# ═════════════════════════════════════════════════════════════════
# CONTROL — activa / desactiva cada paso del pipeline
# ═════════════════════════════════════════════════════════════════
# MODO SEGUR0: por defecto todo está desactivado para que ejecutar main.py no genere
# imágenes ni graficas automáticamente.
RUN_SAFE_MODE = True

isTrainingRequired         = False   # entrena el modelo desde cero (tarda horas)
isPredictionRequired       = True   # corre YOLO sobre train/val/test
isVideoRequired             = False   # genera vídeos superpuesto + subplot
isPixelToMetrosRequired    = True   # convierte labels predichos -> z_metros.csv
isFFTPredRequired          = False  # FFT 1D + 3D de la PREDICCIÓN
isFFTGTRequired            = False   # FFT 1D + 3D del GROUNDTRUTH
isBeamformingRequired      = False  # activar beamforming
isBeamformingPred          = False  # aplicar sobre prediccion
isBeamformingGT            = False # aplicar sobre GT (alturanodos.csv)
isComparacionGTvsPredRequired = False  # genera la figura apilada GT vs Predicción
isEnergiaGTRequired        = False   # gráficas E vs A y S(w) vs w del GT
isEnergia_GT_alturanodos   = False # gráficas E vs A y S(w) vs w del GT (alturanodos.csv)
isComparacionGTSpecVsGTNodos = False #comparacion D.Polar de GT(spectrum) y GT (alturanodos)
isenergiapred              = False # gráficas E vs A y S(w) vs w de la PREDICCIÓN

if RUN_SAFE_MODE:
    print("MODO SEGUR0 activado: ningún paso de generación de imágenes se ejecutará al lanzar main.py.")

# ═════════════════════════════════════════════════════════════════
# PASO 1 — ENTRENAMIENTO
# ═════════════════════════════════════════════════════════════════
if isTrainingRequired:
    modules.entrenamiento.entrenar_modelo()
else:
    print("PASO 1 (entrenamiento) omitido.")

# ═════════════════════════════════════════════════════════════════
# PASO 2 — PREDICCIÓN + VÍDEOS
# ═════════════════════════════════════════════════════════════════
if isPredictionRequired:
    modules.prediccion_videos.predecir_con_modelo()
else:
    print("PASO 2 (predicción) omitido — usando labels ya existentes.")

if isVideoRequired:
    modules.prediccion_videos.generar_videos()
else:
    print("PASO 2b (vídeos) omitido.")

# ═════════════════════════════════════════════════════════════════
# PASO 3 — CONVERSIÓN PÍXEL -> METROS (predicción)
# ═════════════════════════════════════════════════════════════════
if isPixelToMetrosRequired:
    pixel_a_metros.convertir_pixeles_a_metros()
else:
    print("PASO 3 (pixel->metros) omitido — usando z_metros.csv ya existente.")

# ═════════════════════════════════════════════════════════════════
# PASO 4 — FFT DE LA PREDICCIÓN
# ═════════════════════════════════════════════════════════════════
G_pred, T_pred, A_pred, dir_bins = None, None, None, None

if isFFTPredRequired:
    import pandas as pd

    df_z = pd.read_csv(config.RUTA_Z_METROS_PRED, sep=';')
    cols_kp = [c for c in df_z.columns if c.startswith('kp_')]
    z_all_pred = df_z[cols_kp].values
    n_frames = z_all_pred.shape[0]
    tiempo_vector = np.arange(n_frames) * config.DELTA_T + 30.0

    z_centrada_pred = modules.fft_analisis.fft_1d_por_keypoint(
        z_all_pred, tiempo_vector, config.DIR_SALIDA_FFT_PRED)

    dir_flat, T_flat, amp_flat = modules.fft_analisis.fft_3d_direccional(z_centrada_pred)
    G_pred, T_pred, A_pred, dir_bins, _ = modules.fft_analisis.agrupar_en_bins(dir_flat, T_flat, amp_flat)

    modules.espectro_polar.guardar_csv_bins(G_pred, T_pred, A_pred,
                                     config.DIR_SALIDA_FFT_PRED, "espectro_3D_bins_pred.csv")
    modules.espectro_polar.generar_polar_individual(
        G_pred, T_pred, A_pred, dir_bins, config.DIR_SALIDA_FFT_PRED,
        "espectro_polar_Pred_FFT3D.png",
        "Espectro direccional Predicción IA (FFT 3D)\nÁngulo=Dirección · Radio=Periodo · Color=Amplitud")
else:
    print("PASO 4 (FFT predicción) omitido.")

# ═════════════════════════════════════════════════════════════════
# PASO 5 — FFT DEL GROUNDTRUTH
# ═════════════════════════════════════════════════════════════════
G_gt_3d, T_gt_3d, A_gt_3d, dir_bins_gt = None, None, None, None

if isFFTGTRequired:
    z_all_gt, tiempo_vector_gt = modules.lectura_gt.cargar_z_gt()
    etiquetas = modules.lectura_gt.etiquetas_gt()

    z_centrada_gt = modules.fft_analisis.fft_1d_por_keypoint(
        z_all_gt, tiempo_vector_gt, config.DIR_SALIDA_FFT_GT, etiquetas_kp=etiquetas)

    dir_flat_gt, T_flat_gt, amp_flat_gt = modules.fft_analisis.fft_3d_direccional(z_centrada_gt)
    G_gt_3d, T_gt_3d, A_gt_3d, dir_bins_gt, _ = modules.fft_analisis.agrupar_en_bins(
        dir_flat_gt, T_flat_gt, amp_flat_gt)

    modules.espectro_polar.guardar_csv_bins(G_gt_3d, T_gt_3d, A_gt_3d,
                                     config.DIR_SALIDA_FFT_GT, "espectro_3D_bins_GT.csv")
    modules.espectro_polar.generar_polar_individual(
        G_gt_3d, T_gt_3d, A_gt_3d, dir_bins_gt, config.DIR_SALIDA_FFT_GT,
        "espectro_polar_GT_FFT3D.png",
        "Espectro direccional Groundtruth (FFT 3D sobre nodos)\nÁngulo=Dirección · Radio=Periodo · Color=Amplitud")
else:
    print("PASO 5 (FFT groundtruth) omitido.")

# ═════════════════════════════════════════════════════════════════
# PASO 6 — COMPARACIÓN GT (Spectrum.out.dat) vs PREDICCIÓN
# ═════════════════════════════════════════════════════════════════
if isComparacionGTvsPredRequired:
    if G_pred is None or dir_bins is None:
        print("⚠️  Necesitas isFFTPredRequired=True para poder comparar.")
    else:
        T_gt_spec, A_gt_spec, G_gt_spec = modules.espectro_polar.cargar_gt_spectrum()
        modules.espectro_polar.generar_comparacion_gt_vs_pred(
            G_gt_spec, T_gt_spec, A_gt_spec,
            G_pred, T_pred, A_pred,
            dir_bins, config.DIR_SALIDA_FFT_PRED,
            "espectro_polar_GT_vs_Pred_FFT3D.png")
else:
    print("PASO 6 (comparación GT vs predicción) omitido.")
# ═════════════════════════════════════════════════════════════════
# PASO 6b — COMPARACIÓN GT (Spectrum.out.dat) vs FFT3D de alturanodos.csv
#           Aísla si el ensanchamiento del espectro viene de la rejilla
#           3x8 (aparecería aquí también) o del modelo YOLO (solo en pred)
# ═════════════════════════════════════════════════════════════════
if isComparacionGTSpecVsGTNodos:
    if G_gt_3d is None or dir_bins_gt is None:
        print("⚠️  Necesitas isFFTGTRequired=True para poder comparar.")
    else:
        T_gt_spec, A_gt_spec, G_gt_spec = modules.espectro_polar.cargar_gt_spectrum()
        modules.espectro_polar.generar_comparacion_gt_vs_pred(
            G_gt_spec, T_gt_spec, A_gt_spec,
            G_gt_3d, T_gt_3d, A_gt_3d,
            dir_bins_gt, config.DIR_SALIDA_FFT_GT,
            "espectro_polar_GT_vs_GT3D_alturanodos.png")
else:
    print("PASO 6b (comparación GT spectrum vs GT alturanodos) omitido.")
# ═════════════════════════════════════════════════════════════════
# PASO 7 — GRÁFICAS DE ENERGÍA DEL GROUNDTRUTH
# ═════════════════════════════════════════════════════════════════
if isEnergiaGTRequired:
    energia_gt.generar_graficas_energia_gt()
else:
    print("PASO 7 (energía GT) omitido.")

# ═════════════════════════════════════════════════════════════════
# PASO 7a — GRÁFICAS DE ENERGÍA DEL GROUNDTRUTH alturanodos.csv
# ═════════════════════════════════════════════════════════════════
if isEnergia_GT_alturanodos:
    energia_gt_alturanodos.generar_graficas_energia_gt_alturanodos()
else:
    print("PASO 7a (energía GT alturanodos) omitido.")
    
# ═════════════════════════════════════════════════════════════════
# PASO 7b — GRÁFICAS DE ENERGÍA DE LA PREDICCIÓN
# ═════════════════════════════════════════════════════════════════
if isenergiapred:
    energiapred.generar_graficas_energiapred()
else:
    print("PASO 7b (energía predicción) omitido.")



# ═════════════════════════════════════════════════════════════════
# PASO 8 — BEAMFORMING TEMPORAL (estimacion direccional)
# ═════════════════════════════════════════════════════════════════
if isBeamformingRequired:
    from modules import beamforming

    if isBeamformingPred:
        thetas, freqs, A_f_theta = beamforming.calcular_beamforming(
        fuente="pred",
        dir_salida=config.DIR_SALIDA_BEAMFORMING_PRED
    )
    beamforming.generar_espectro_energiaomnidireccional(
        thetas, freqs, A_f_theta,
        dir_salida=config.DIR_SALIDA_BEAMFORMING_PRED,
        fuente="pred",
        Hs_ref=6.0,
        Tm_ref=9.0
    )

    if isBeamformingGT:
        thetas, freqs, A_f_theta = beamforming.calcular_beamforming(
            fuente="gt",
            dir_salida=config.DIR_SALIDA_BEAMFORMING_GT
        )
        beamforming.generar_espectro_energiaomnidireccional(
            thetas, freqs, A_f_theta,
            dir_salida=config.DIR_SALIDA_BEAMFORMING_GT,
            fuente="gt",
            Hs_ref=6.0,
            Tm_ref=9.0
        )

    if not isBeamformingPred and not isBeamformingGT:
        print("⚠️  Activa isBeamformingPred y/o isBeamformingGT.")
else:
    print("PASO 8 (beamforming) omitido.")
    
print("\n✅ Pipeline finalizado.")