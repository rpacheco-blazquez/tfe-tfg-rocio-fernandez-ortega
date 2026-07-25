"""
modules/pixel_a_metros.py
Convierte las posiciones (u, v) en píxeles predichas por el modelo a
altura z en metros, usando la ecuación z_v despejada de la proyección
de cámara K[R|T] (la única numéricamente estable, ver denom_v >> denom_u).
"""

import os
import numpy as np
import pandas as pd

import config


def convertir_pixeles_a_metros():
    """Lee los labels predichos, aplica la fórmula z_v y guarda z_metros.csv."""
    print("Cargando parámetros de cámara...")
    df_cam = pd.read_csv(config.RUTA_MATRICES_CAM)
    params = dict(zip(df_cam['parametro'], df_cam['valor']))

    f_y, c_y = params['f_y'], params['c_y']
    r_21, r_22, r_23 = params['r_21'], params['r_22'], params['r_23']
    r_31, r_32, r_33 = params['r_31'], params['r_32'], params['r_33']
    t_2, t_3 = params['t_2'], params['t_3']

    print(f"  f_y={f_y:.4f}  c_y={c_y:.4f}  t_2={t_2:.4f}  t_3={t_3:.4f}")

    # Coordenadas X, Y fijas de cada keypoint
    df_x = pd.read_csv(os.path.join(config.RUTA_CSVS, "X.csv"))
    nodos = df_x['nodo'].astype(int).tolist()
    coords_x_kp = np.array([df_x.loc[df_x['nodo'] == n, 'x'].values[0] for n in nodos])
    coords_y_kp = np.array([df_x.loc[df_x['nodo'] == n, 'y'].values[0] for n in nodos])
    print(f"  Keypoints cargados: {len(nodos)}")

    # Términos constantes de la ecuación z_v (despejada vía MATLAB)
    # z_v = (c_y*t_3 + f_y*t_2 - v*(t_3 + r_31*x + r_32*y)
    #        + x*(c_y*r_31 + f_y*r_21) + y*(c_y*r_32 + f_y*r_22))
    #       / (c_y*r_33 + f_y*r_23 - r_33*v)
    num_const_v = (c_y * t_3 + f_y * t_2
                   + coords_x_kp * (c_y * r_31 + f_y * r_21)
                   + coords_y_kp * (c_y * r_32 + f_y * r_22))
    coef_v_kp = -(t_3 + r_31 * coords_x_kp + r_32 * coords_y_kp)
    denom_const_v = c_y * r_33 + f_y * r_23

    print(f"  denom_const_v = {denom_const_v:.4f}")

    # Verificación con KP00
    v_kp00 = 287.43
    den_v_test = denom_const_v - r_33 * v_kp00
    num_v_test = num_const_v[0] + coef_v_kp[0] * v_kp00
    z_v_test = num_v_test / den_v_test
    print(f"\n  Verificación KP00: z_v={z_v_test:.4f}m (real=0.2231m)")

    # Cargar labels predichos (generados por predecir_con_modelo() en prediccion_videos.py)
    print("\nCargando labels de predicción...")
    todos_los_txt = []
    for split in ["train", "val", "test"]:
        carpeta = os.path.join(config.RUTA_BASE_RUNS, f"pred_{split}", "labels")
        if os.path.exists(carpeta):
            for archivo in os.listdir(carpeta):
                if archivo.endswith('.txt'):
                    todos_los_txt.append(os.path.join(carpeta, archivo))
        else:
            print(f"  ⚠️  No encontrada: {carpeta}")
            
    def _extraer_numero(ruta):
        digitos = ''.join(filter(str.isdigit, os.path.basename(ruta)))
        return int(digitos) if digitos else 0

    todos_los_txt.sort(key=_extraer_numero)
    n_frames = len(todos_los_txt)
    print(f"  Total frames: {n_frames}")

    # Calcular z(t) con la ecuación v para todos los frames y keypoints
    print("\nCalculando z(t) con ecuación v frame a frame...")
    z_resultado = np.zeros((n_frames, config.NUM_KEYPOINTS))

    for f_idx, ruta_txt in enumerate(todos_los_txt):
        with open(ruta_txt, 'r') as f:
            lineas = f.readlines()
        if not lineas:
            continue

        datos_kp = lineas[0].strip().split()[5:]

        for kp_id in range(config.NUM_KEYPOINTS):
            idx_v = kp_id * 3 + 1
            if idx_v >= len(datos_kp):
                continue

            v = float(datos_kp[idx_v]) * config.IMG_H
            den_v = denom_const_v - r_33 * v

            if abs(den_v) > config.UMBRAL_DENOMINADOR:
                z_resultado[f_idx, kp_id] = (num_const_v[kp_id] + coef_v_kp[kp_id] * v) / den_v
            else:
                z_resultado[f_idx, kp_id] = 0.0

        if f_idx % 200 == 0:
            print(f"  Procesados {f_idx}/{n_frames} frames...")

    print(f"  ✅ {n_frames} frames procesados")
    print(f"\n  z_resultado — rango: {z_resultado.min():.3f}m — {z_resultado.max():.3f}m")
    print(f"  z_resultado — media: {z_resultado.mean():.3f}m")

    # Guardar CSV
    cols = ['frame'] + [f'kp_{i:02d}' for i in range(config.NUM_KEYPOINTS)]
    datos = np.column_stack([np.arange(1, n_frames + 1), z_resultado])
    df_salida = pd.DataFrame(datos, columns=cols)
    df_salida['frame'] = df_salida['frame'].astype(int)
    df_salida.to_csv(config.RUTA_Z_METROS_PRED, index=False, sep=';')

    print(f"\n✅ Guardado en: {config.RUTA_Z_METROS_PRED}")
    return config.RUTA_Z_METROS_PRED
