"""
modules/entrenamiento.py
Entrena el modelo YOLO-Pose y calcula métricas de error de keypoints
sobre el set de validación.
"""

import os
import time
import csv
import numpy as np
from ultralytics import YOLO

import config


def calcular_metricas(modelo):
    """Calcula distancia píxel a píxel entre keypoints reales y predichos
    sobre el set de validación, y guarda un CSV resumen."""
    print("\n" + "=" * 55)
    print("  CALCULANDO MÉTRICAS DE ERROR DE KEYPOINTS")
    print("=" * 55)

    val_images_dir = config.CARPETAS_IMAGENES_DIVIDIDAS["val"]
    val_labels_dir = config.CARPETAS_LABELS_GT_DIVIDIDAS["val"]

    imagenes = sorted([
        f for f in os.listdir(val_images_dir)
        if f.endswith('.png') or f.endswith('.jpg')
    ])

    if not imagenes:
        print("❌ No hay imágenes en el set de validación.")
        return

    print(f"  Imágenes a evaluar: {len(imagenes)}")

    distancias_todas  = []
    distancias_por_kp = {i: [] for i in range(config.NUM_KEYPOINTS)}
    max_distancia, max_frame, max_kp_idx = 0.0, "", -1

    for img_name in imagenes:
        img_path = os.path.join(val_images_dir, img_name)
        lbl_path = os.path.join(
            val_labels_dir,
            img_name.replace('.png', '.txt').replace('.jpg', '.txt'))

        if not os.path.exists(lbl_path):
            continue

        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            continue

        parts = lines[0].strip().split()
        kp_start = 5
        kps_reales = []

        for i in range(config.NUM_KEYPOINTS):
            idx = kp_start + i * 3
            if idx + 1 < len(parts):
                kx = float(parts[idx])     * config.IMG_W
                ky = float(parts[idx + 1]) * config.IMG_H
                kps_reales.append((kx, ky))

        if len(kps_reales) != config.NUM_KEYPOINTS:
            continue

        results = modelo.predict(img_path, verbose=False)
        if not results or results[0].keypoints is None:
            continue

        kps_pred = results[0].keypoints.xy.cpu().numpy()
        if len(kps_pred) == 0 or len(kps_pred[0]) != config.NUM_KEYPOINTS:
            continue
        kps_pred = kps_pred[0]

        for i, (rx, ry) in enumerate(kps_reales):
            px, py = kps_pred[i]
            dist = np.sqrt((rx - px) ** 2 + (ry - py) ** 2)
            distancias_todas.append(dist)
            distancias_por_kp[i].append(dist)
            if dist > max_distancia:
                max_distancia, max_frame, max_kp_idx = dist, img_name, i

    if not distancias_todas:
        print("⚠️  No se pudieron calcular distancias. Verifica los labels.")
        return

    print(f"\n  📏 DISTANCIA MÁXIMA: {max_distancia:.2f} px (frame {max_frame}, kp_{max_kp_idx:02d})")
    print(f"  📊 DISTANCIA PROMEDIO: {np.mean(distancias_todas):.2f} px")
    print(f"  📊 MEDIANA: {np.median(distancias_todas):.2f} px")
    print(f"  📊 DESVIACIÓN STD: {np.std(distancias_todas):.2f} px")

    os.makedirs(config.DIR_SALIDA_METRICAS, exist_ok=True)
    ruta_csv = os.path.join(config.DIR_SALIDA_METRICAS, "metricas_keypoints.csv")
    with open(ruta_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["keypoint", "promedio_px", "max_px", "n_muestras"])
        for i in range(config.NUM_KEYPOINTS):
            if distancias_por_kp[i]:
                writer.writerow([
                    f"kp_{i:02d}",
                    f"{np.mean(distancias_por_kp[i]):.4f}",
                    f"{np.max(distancias_por_kp[i]):.4f}",
                    len(distancias_por_kp[i])
                ])

    print(f"\n  💾 Métricas guardadas en: {ruta_csv}")
    print("=" * 55)


def entrenar_modelo():
    """Entrena el modelo YOLO-Pose desde cero y calcula métricas finales."""
    print("=" * 55)
    print("  ENTRENAMIENTO DEL MODELO YOLO-POSE")
    print("=" * 55)

    modelo = YOLO(config.MODELO_BASE)
    inicio = time.time()

    modelo.train(
        data=config.RUTA_DATASET_YAML,
        epochs=config.EPOCHS,
        imgsz=config.IMG_W,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        patience=config.PATIENCE,
    )

    tiempo_min = (time.time() - inicio) / 60
    print(f"\n  ✅ Entrenamiento completado en {tiempo_min:.1f} min ({tiempo_min/60:.2f} h)")
    print(f"  Modelo guardado en: runs/train/weights/best.pt")

    calcular_metricas(modelo)
    return modelo
