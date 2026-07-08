"""
modules/prediccion_videos.py
Ejecuta la predicción del modelo YOLO-Pose sobre los splits train/val/test
y genera vídeos comparativos GT vs predicción (superpuesto y subplot).
"""

import os
import csv
import numpy as np
import cv2
import pandas as pd

import config


def _cargar_conectividad():
    """Carga la conectividad de triángulos (T.csv) y el mapeo nodo->índice (X.csv)."""
    T = []
    with open(os.path.join(config.RUTA_CSVS, "T.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            T.append([int(row["nodo1"]), int(row["nodo2"]), int(row["nodo3"])])

    df = pd.read_csv(os.path.join(config.RUTA_CSVS, "X.csv"))
    nodos = df['nodo'].astype(int).tolist()
    nodo_a_idx = {n: i for i, n in enumerate(nodos)}
    return T, nodo_a_idx


def _leer_keypoints_txt(ruta_txt):
    with open(ruta_txt, "r") as f:
        valores = list(map(float, f.read().split()))
    kp_raw = valores[5:]
    kps = []
    for i in range(config.NUM_KEYPOINTS):
        u = kp_raw[i * 3] * config.IMG_W
        v = kp_raw[i * 3 + 1] * config.IMG_H
        kps.append([u, v])
    return np.array(kps)


def _dibujar_malla_cv2(img, kps, color, T, nodo_a_idx):
    for tri in T:
        i0, i1, i2 = (nodo_a_idx[tri[0]], nodo_a_idx[tri[1]], nodo_a_idx[tri[2]])
        for a, b in [(i0, i1), (i1, i2), (i2, i0)]:
            p1 = (int(kps[a][0]), int(kps[a][1]))
            p2 = (int(kps[b][0]), int(kps[b][1]))
            cv2.line(img, p1, p2, color, config.GROSOR_LINEA, cv2.LINE_AA)
    for u, v in kps:
        cv2.circle(img, (int(u), int(v)), config.RADIO_NODO, color, -1, cv2.LINE_AA)


def predecir_con_modelo():
    """Ejecuta YOLO-Pose sobre train/val/test y guarda los labels predichos."""
    from ultralytics import YOLO

    print("=" * 55)
    print("  PREDICCIÓN CON best.pt")
    print("=" * 55)

    modelo = YOLO(config.RUTA_MODELO)

    for split in ["train", "val", "test"]:
        ruta_imagenes = config.CARPETAS_IMAGENES_DIVIDIDAS[split]
        n_imgs = len([f for f in os.listdir(ruta_imagenes) if f.endswith(".png")])
        print(f"\n  Prediciendo {split.upper()} ({n_imgs} imágenes)...")

        modelo.predict(
            source=ruta_imagenes,
            save=False,
            save_txt=True,
            save_conf=False,
            project=config.RUTA_BASE_RUNS,
            name=f"pred_{split}",
            exist_ok=True,
            verbose=False,
        )

        ruta_labels = os.path.join(config.RUTA_BASE_RUNS, f"pred_{split}", "labels")
        n = len([f for f in os.listdir(ruta_labels) if f.endswith(".txt")]) if os.path.exists(ruta_labels) else 0
        print(f"  ✅ {n} .txt generados")

    print("\n  Predicción completada para todos los splits.")


def _recopilar_frames_en_orden():
    todas_las_frames = []
    for split in ["train", "val", "test"]:
        ruta_imgs = config.CARPETAS_IMAGENES_DIVIDIDAS[split]
        ruta_gt   = config.CARPETAS_LABELS_GT_DIVIDIDAS[split]
        ruta_pred = os.path.join(config.RUTA_BASE_RUNS, f"pred_{split}", "labels")

        archivos = [f for f in os.listdir(ruta_gt) if f.endswith(".txt")]
        for archivo in archivos:
            num = int(''.join(filter(str.isdigit, archivo)) or 0)
            ruta_img_path = os.path.join(ruta_imgs, archivo.replace(".txt", ".png"))
            ruta_gt_txt   = os.path.join(ruta_gt, archivo)
            ruta_pred_txt = os.path.join(ruta_pred, archivo)
            todas_las_frames.append((num, ruta_img_path, ruta_gt_txt, ruta_pred_txt))

    todas_las_frames.sort(key=lambda x: x[0])
    return todas_las_frames


def generar_videos():
    """Genera los vídeos superpuesto y subplot comparando GT vs predicción."""
    print("\n" + "=" * 55)
    print("  GENERANDO VÍDEOS")
    print("=" * 55)

    os.makedirs(config.DIR_SALIDA_VIDEOS, exist_ok=True)
    T, nodo_a_idx = _cargar_conectividad()
    todas_las_frames = _recopilar_frames_en_orden()
    print(f"  Total frames recopiladas: {len(todas_las_frames)}")

    w_out = int(config.IMG_W * config.VIDEO_ESCALA)
    h_out = int(config.IMG_H * config.VIDEO_ESCALA)
    w_sub = w_out * 2 + 4

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ruta_v1 = os.path.join(config.DIR_SALIDA_VIDEOS, "video_superpuesto_1000frames.mp4")
    ruta_v2 = os.path.join(config.DIR_SALIDA_VIDEOS, "video_subplot_1000frames.mp4")

    writer_v1 = cv2.VideoWriter(ruta_v1, fourcc, config.VIDEO_FPS, (w_out, h_out))
    writer_v2 = cv2.VideoWriter(ruta_v2, fourcc, config.VIDEO_FPS, (w_sub, h_out))

    frames_ok = 0
    for num, ruta_img_path, ruta_gt_txt, ruta_pred_txt in todas_las_frames:
        if not os.path.exists(ruta_pred_txt) or not os.path.exists(ruta_img_path):
            continue

        img_bgr = cv2.imread(ruta_img_path)
        if img_bgr is None:
            continue

        gt   = _leer_keypoints_txt(ruta_gt_txt)
        pred = _leer_keypoints_txt(ruta_pred_txt)

        img_small = cv2.resize(img_bgr, (w_out, h_out))
        gt_sc, pred_sc = gt * config.VIDEO_ESCALA, pred * config.VIDEO_ESCALA

        # Vídeo 1: superpuesto
        frame_v1 = img_small.copy()
        _dibujar_malla_cv2(frame_v1, gt_sc, config.COLOR_GT, T, nodo_a_idx)
        _dibujar_malla_cv2(frame_v1, pred_sc, config.COLOR_PRED, T, nodo_a_idx)
        cv2.putText(frame_v1, "Groundtruth (azul)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_GT, 1, cv2.LINE_AA)
        cv2.putText(frame_v1, "Prediccion (rojo)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_PRED, 1, cv2.LINE_AA)
        cv2.putText(frame_v1, f"Frame {num:04d}", (10, h_out - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        writer_v1.write(frame_v1)

        # Vídeo 2: subplot
        lado_gt, lado_pred = img_small.copy(), img_small.copy()
        _dibujar_malla_cv2(lado_gt, gt_sc, config.COLOR_GT, T, nodo_a_idx)
        _dibujar_malla_cv2(lado_pred, pred_sc, config.COLOR_PRED, T, nodo_a_idx)
        cv2.putText(lado_gt, "Groundtruth", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_GT, 1, cv2.LINE_AA)
        cv2.putText(lado_pred, "Prediccion", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_PRED, 1, cv2.LINE_AA)
        cv2.putText(lado_gt, f"Frame {num:04d}", (10, h_out - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        separador = np.zeros((h_out, 4, 3), dtype=np.uint8)
        frame_v2 = np.hstack([lado_gt, separador, lado_pred])
        writer_v2.write(frame_v2)

        frames_ok += 1
        if frames_ok % 100 == 0:
            print(f"  Procesadas {frames_ok}/1000 frames...")

    writer_v1.release()
    writer_v2.release()

    print(f"\n  ✅ {frames_ok} frames procesadas")
    print(f"  🎬 Superpuesto → {ruta_v1}")
    print(f"  🎬 Subplot     → {ruta_v2}")
