"""
modules/espectro_polar.py
Funciones de dibujo de las rosas polares de espectro direccional
(GT y predicción) y de su comparación apilada.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import config


def cargar_gt_spectrum():
    """Carga T, A, G del Spectrum.out.dat (GT teórico de SeaFEM)."""
    df_spec = pd.read_csv(config.RUTA_SPECTRUM, sep=r'\s+', skiprows=2)
    df_spec.columns = ["TWaves", "AWaves", "GWaves", "PWaves"]
    return df_spec["TWaves"].values, df_spec["AWaves"].values, df_spec["GWaves"].values


def dibujar_rosa(ax, G, T, A, titulo, dir_bins, norm_A, cmap, es_gt=True):
    """Dibuja una rosa polar de espectro direccional sobre el eje dado."""
    colors_local = cmap(norm_A(A))
    if es_gt:
        dirs_unicas = np.unique(G)
        ancho = (dirs_unicas[1] - dirs_unicas[0]) * 0.85
        for i in range(len(T)):
            ax.bar(G[i], T[i], width=ancho, bottom=0,
                   color=colors_local[i], alpha=0.9, linewidth=0.3, edgecolor="white")
    else:
        ancho = (dir_bins[1] - dir_bins[0]) * 0.9
        for i in range(len(G)):
            ax.bar(G[i], T[i], width=ancho, bottom=0,
                   color=colors_local[i], alpha=0.9, linewidth=0.2, edgecolor="none")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_rticks([3, 5, 7, 9])
    ax.set_yticklabels(["3s", "5s", "7s", "9s"], fontsize=8)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{a}°" for a in np.arange(0, 360, 30)], fontsize=8)
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=20)


def generar_polar_individual(G_plot, T_plot, A_plot, dir_bins, dir_salida,
                              ruta_archivo, titulo, A_gt_referencia=None):
    """Genera una figura polar única (solo predicción o solo GT)."""
    if A_gt_referencia is not None:
        A_all = np.concatenate([A_gt_referencia, A_plot])
    else:
        A_all = A_plot
    norm_A = mcolors.Normalize(vmin=A_all.min(), vmax=A_all.max())
    cmap = cm.plasma

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))
    dibujar_rosa(ax, G_plot, T_plot, A_plot, titulo, dir_bins, norm_A, cmap, es_gt=False)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm_A)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, fraction=0.04, label="Amplitud A (m)")
    plt.tight_layout()

    ruta = os.path.join(dir_salida, ruta_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Polar guardada: {ruta}")
    return ruta


def generar_comparacion_gt_vs_pred(G_gt, T_gt, A_gt, G_pred, T_pred, A_pred,
                                    dir_bins, dir_salida, ruta_archivo):
    """Genera la figura apilada GT (arriba) vs Predicción (abajo)."""
    A_all = np.concatenate([A_gt, A_pred])
    norm_A = mcolors.Normalize(vmin=A_all.min(), vmax=A_all.max())
    cmap = cm.plasma

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, subplot_kw={"projection": "polar"}, figsize=(10, 18))

    dibujar_rosa(ax_gt, G_gt, T_gt, A_gt,
                 "Groundtruth (SeaFEM)\nÁngulo=Dirección · Radio=Periodo · Color=Amplitud",
                 dir_bins, norm_A, cmap, es_gt=True)
    dibujar_rosa(ax_pred, G_pred, T_pred, A_pred,
                 "Predicción IA (FFT 3D)\nÁngulo=Dirección · Radio=Periodo · Color=Amplitud",
                 dir_bins, norm_A, cmap, es_gt=False)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm_A)
    sm.set_array([])
    fig.colorbar(sm, ax=[ax_gt, ax_pred], pad=0.08, fraction=0.025, shrink=0.5, label="Amplitud A (m)")
    fig.suptitle("Comparación Espectro Direccional\nGroundtruth vs Predicción IA",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    ruta = os.path.join(dir_salida, ruta_archivo)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Comparación guardada: {ruta}")
    return ruta


def guardar_csv_bins(G_plot, T_plot, A_plot, dir_salida, nombre_archivo):
    pd.DataFrame({
        'Direccion_rad': G_plot, 'Direccion_deg': np.degrees(G_plot),
        'Periodo_s': T_plot, 'Amplitud_m': A_plot
    }).to_csv(os.path.join(dir_salida, nombre_archivo), index=False, sep=';')
