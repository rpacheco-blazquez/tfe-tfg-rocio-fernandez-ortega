"""
modules/camara.py
Extracción de matrices intrínseca (K) y extrínseca (R, T) de la cámara
de Blender. Este script se ejecuta DENTRO de Blender (requiere bpy),
no como parte del pipeline principal de Python estándar.
"""

import os
import csv
import numpy as np

import config


def extraer_matrices_camara_blender(ruta_salida=None):
    """
    Extrae K, R, T de la cámara activa de la escena de Blender y las
    guarda en un CSV. SOLO funciona ejecutado desde dentro de Blender
    (Scripting tab), ya que requiere el módulo bpy.
    """
    import bpy  # solo disponible dentro de Blender

    if ruta_salida is None:
        ruta_salida = config.RUTA_MATRICES_CAM

    escena = bpy.context.scene
    cam_obj = escena.camera
    cam_data = cam_obj.data

    res_x = escena.render.resolution_x * escena.render.resolution_percentage / 100
    res_y = escena.render.resolution_y * escena.render.resolution_percentage / 100

    sensor_width  = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    focal_length  = cam_data.lens

    if cam_data.sensor_fit == 'VERTICAL':
        sensor_height_eff = sensor_height
        sensor_width_eff  = sensor_height * (res_x / res_y)
    else:
        sensor_width_eff  = sensor_width
        sensor_height_eff = sensor_width * (res_y / res_x)

    f_x = (focal_length / sensor_width_eff) * res_x
    f_y = (focal_length / sensor_height_eff) * res_y
    c_x = res_x / 2.0
    c_y = res_y / 2.0

    # Corrección de convención Blender (mira a -Z) -> OpenCV (mira a +Z)
    R_blender_to_cv = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])

    mat_world_np = np.array(cam_obj.matrix_world)
    R_cam_to_world = mat_world_np[:3, :3]
    t_cam_to_world = mat_world_np[:3, 3]

    R_cam_to_world_cv = R_cam_to_world @ R_blender_to_cv
    R_world_to_cam = R_cam_to_world_cv.T
    t_world_to_cam = -R_world_to_cam @ t_cam_to_world

    with open(ruta_salida, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["parametro", "valor"])
        writer.writerow(["f_x", f_x])
        writer.writerow(["f_y", f_y])
        writer.writerow(["c_x", c_x])
        writer.writerow(["c_y", c_y])
        writer.writerow(["r_11", R_world_to_cam[0, 0]])
        writer.writerow(["r_12", R_world_to_cam[0, 1]])
        writer.writerow(["r_13", R_world_to_cam[0, 2]])
        writer.writerow(["r_21", R_world_to_cam[1, 0]])
        writer.writerow(["r_22", R_world_to_cam[1, 1]])
        writer.writerow(["r_23", R_world_to_cam[1, 2]])
        writer.writerow(["r_31", R_world_to_cam[2, 0]])
        writer.writerow(["r_32", R_world_to_cam[2, 1]])
        writer.writerow(["r_33", R_world_to_cam[2, 2]])
        writer.writerow(["t_1", t_world_to_cam[0]])
        writer.writerow(["t_2", t_world_to_cam[1]])
        writer.writerow(["t_3", t_world_to_cam[2]])
        writer.writerow(["res_x", res_x])
        writer.writerow(["res_y", res_y])

    print(f"✅ Matrices de cámara guardadas en: {ruta_salida}")
