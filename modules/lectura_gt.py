"""
modules/lectura_gt.py
Carga la elevacion real (groundtruth) de los 24 nodos seleccionados de
la malla desde alturanodos.csv, generado por SeaFEM.
"""

import numpy as np
import pandas as pd

import config


def cargar_z_gt():
    """
    Lee alturanodos.csv y extrae las columnas correspondientes a los
    24 nodos (en el orden exacto de la rejilla 3x8).
    Devuelve (z_all, tiempo_vector) donde z_all tiene shape (N_frames, 24).
    """
    print("Cargando datos del GT desde alturanodos.csv...")

    # index_col=0 convierte 'time' en indice, columnas = nodos como integers
    df_z = pd.read_csv(config.RUTA_GT_ALTURA_CSV, sep=',', header=0, index_col=0)

    # Tiempo desde el indice
    tiempo_vector = df_z.index.to_numpy()

    print(f"Total columnas (nodos): {len(df_z.columns)}")
    print(f"Tipo de columnas: {df_z.columns.dtype}")
    print(f"Primeras 5 columnas: {list(df_z.columns[:5])}")

    # Convertir LISTA_KEYPOINTS_ORDENADOS al mismo tipo que las columnas del CSV
    if df_z.columns.dtype == 'int64':
        nodos = [int(n) for n in config.LISTA_KEYPOINTS_ORDENADOS]
    else:
        nodos = [str(n) for n in config.LISTA_KEYPOINTS_ORDENADOS]

    # Verificar que todos los nodos existen
    nodos_faltantes = [n for n in nodos if n not in df_z.columns]
    if nodos_faltantes:
        print(f"  WARNING: estos nodos no estan en el CSV: {nodos_faltantes}")
    else:
        print(f"  OK: los 24 nodos encontrados en el CSV")

    # Construir z_all (N_frames x 24)
    z_all = df_z[nodos].values
    
    # Verificar cuantos nodos tienen datos reales
    nodos_con_datos = 0
    for i, nodo in enumerate(nodos):
        n_nonzero = np.sum(z_all[:, i] != 0)
        if n_nonzero == 0:
            print(f"  WARNING: nodo {nodo} (KP{i:02d}) tiene todos los valores a cero")
        else:
            nodos_con_datos += 1

    print(f"Frames leidos: {z_all.shape[0]} | Keypoints con datos reales: {nodos_con_datos}/24")
    print(f"z_all rango: {z_all.min():.3f}m -- {z_all.max():.3f}m")

    return z_all, tiempo_vector


def etiquetas_gt():
    """Etiquetas tipo KP_00_Nodo_25606 para los graficos individuales del GT."""
    return [f"KP_{i:02d}_Nodo_{nodo}" for i, nodo in enumerate(config.LISTA_KEYPOINTS_ORDENADOS)]