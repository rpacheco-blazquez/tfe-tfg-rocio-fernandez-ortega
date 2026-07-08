"""
modules/energia_gt_alturanodos.py
Graficas de validacion de energia espectral (E vs A) y densidad espectral
S(w) vs w para es GT alturanodos.csv, usando unicamente la FFT 1D por keypoint.

Compara:
  - Espectro FFT 1D de GT (obtenido de alturanodos.csv)
  - Espectro JONSWAP analitico (ecuaciones 8-11 y 8-12 del manual SeaFEM)
    calculado con los Hs, Tp, Tm01 estimados por FFT 1D

Es el equivalente de energia_gt.py.
"""

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from modules import lectura_gt


# ==========================================
# FORMULA JONSWAP — Ecuacion 8-12 SeaFEM
# S(w) = (155*Hs^2 / (Tm^4 * w^5)) * 3.3^Y * e^(-944*Tm^-4*w^-4)
# Y = e^(-[(0.191*w*Tm - 1) / (sigma*sqrt(2))]^2)
# sigma = 0.07 si w <= 5.24/Tm, sigma = 0.09 si w > 5.24/Tm
# Tm = 2*pi*m0/m1
# ==========================================
def jonswap_8_12(omega, Hs, Tm):
    """
    Espectro JONSWAP segun ecuacion 8-12 del manual SeaFEM.
    omega: array de frecuencias angulares (rad/s)
    Hs:    altura significativa (m)
    Tm:    periodo medio Tm01 = 2*pi*m0/m1 (s)
    """
    Tm = float(Tm)
    if not np.isfinite(Tm) or Tm <= 0:
        return np.zeros_like(omega, dtype=float)
    sigma = np.where(omega <= 5.24 / Tm, 0.07, 0.09)
    Y     = np.exp(-((0.191 * omega * Tm - 1) / (sigma * np.sqrt(2))) ** 2)
    S     = (155.0 * Hs**2 / (Tm**4 * omega**5)) * (3.3 ** Y) * np.exp(-944.0 * Tm**(-4) * omega**(-4))
    return np.maximum(S, 0)


# ==========================================
# FORMULA JONSWAP — Ecuacion 8-11 SeaFEM
# S(T) = (5/(32*pi) * Hs^2 * T^5/Tp^4) * eps^Y * e^(-1.25*(Tp/T)^-4) * (1 - 0.287*log(eps))
# Y = e^(-[(0.159*w*Tp - 1) / (sigma*sqrt(2))]^2)
# sigma = 0.07 si w <= 6.28/Tp, sigma = 0.09 si w > 6.28/Tp
# eps = 3.3 (peakedness parameter por defecto JONSWAP)
# ==========================================
def jonswap_8_11(omega, Hs, Tp, eps=3.3):
    """
    Espectro JONSWAP segun ecuacion 8-11 del manual SeaFEM.
    omega: array de frecuencias angulares (rad/s)
    Hs:    altura significativa (m)
    Tp:    periodo pico (s)
    eps:   parametro de picudo (default 3.3)
    """
    Tp = float(Tp)
    if not np.isfinite(Tp) or Tp <= 0:
        return np.zeros_like(omega, dtype=float)
    T     = 2 * np.pi / omega
    sigma = np.where(omega <= 6.28 / Tp, 0.07, 0.09)
    Y     = np.exp(-((0.159 * omega * Tp - 1) / (sigma * np.sqrt(2))) ** 2)
    S     = ((5 / (32 * np.pi)) * Hs**2 * T**5 / Tp**4) * \
            (eps ** Y) * np.exp(-1.25 * (Tp / T) ** (-4)) * \
            (1 - 0.287 * np.log(eps))
    return np.maximum(S, 0)


def generar_graficas_energia_gt_alturanodos():
    """
    Genera graficas S(w) vs w y E vs A del GT alturanodos.csv comparadas con
    el espectro JONSWAP analitico (ec. 8-11 y 8-12 del manual SeaFEM).
    Los parametros Hs, Tp, Tm01 se estiman por FFT 1D sobre alturanodos.csv.
    """
    os.makedirs(config.DIR_SALIDA_ENERGIA_GT, exist_ok=True)

    # Limpiar salidas previas para que cada ejecución deje solo las gráficas nuevas
    for old_file in os.listdir(config.DIR_SALIDA_ENERGIA_GT):
        full_path = os.path.join(config.DIR_SALIDA_ENERGIA_GT, old_file)
        if os.path.isfile(full_path) and old_file.endswith('.png'):
            try:
                os.remove(full_path)
            except OSError:
                pass

    # ── Cargar alturanodos.csv ──────────────────────────────────
    print("Cargando alturanodos.csv...")
    z_all, _ = lectura_gt.cargar_z_gt()
    z_all = z_all.astype(float)

    N_frames = z_all.shape[0]
    print(f"  Frames: {N_frames}  Keypoints: {z_all.shape[1]}")

    # ── FFT 1D por keypoint ──────────────────────────────────────
    print("Calculando FFT 1D por keypoint...")

    Hs_lista, Tp_lista, Tm01_lista, Tm02_lista = [], [], [], []
    freqs_globales = None
    amps_globales  = []
    etiquetas_kp = lectura_gt.etiquetas_gt()

    for kp_id in range(z_all.shape[1]):
        z = z_all[:, kp_id] - np.mean(z_all[:, kp_id])
        N = len(z)

        fft_vals  = np.fft.fft(z)
        freqs     = np.fft.fftfreq(N, d=config.DELTA_T)
        idx_pos   = np.where(freqs > 0)[0]
        freqs_pos = freqs[idx_pos]
        amps_IA   = (2.0 / N) * np.abs(fft_vals[idx_pos])

        idx_filt  = np.where(freqs_pos <= config.FILTRO_FREC_MAX_HZ)[0]
        freqs_fil = freqs_pos[idx_filt]
        amps_fil  = amps_IA[idx_filt]

        omega_fil = 2.0 * np.pi * freqs_fil
        E_fil = 0.5 * amps_fil**2
        dw_fil = np.diff(omega_fil)
        dw_fil = np.append(dw_fil, dw_fil[-1])
        S_kp = E_fil / dw_fil

        suma_A2 = np.sum(amps_fil ** 2)
        H_s     = np.sqrt(8.0 * suma_A2)
        m0 = np.trapezoid(S_kp, omega_fil)
        m1 = np.trapezoid(omega_fil * S_kp, omega_fil)
        m2 = np.trapezoid((omega_fil ** 2) * S_kp, omega_fil)
        T_m01   = 2*np.pi*m0 / m1          if m1 > 0 else 0
        T_m02   = np.sqrt(m0 / m2) if m2 > 0 else 0

        idx_pico = np.argmax(amps_fil)
        f_pico = freqs_fil[idx_pico]
        T_p      = 1.0 / f_pico if f_pico > 0 else 0

        Hs_lista.append(H_s)
        Tp_lista.append(T_p)
        Tm01_lista.append(T_m01)
        Tm02_lista.append(T_m02)

        if freqs_globales is None:
            freqs_globales = freqs_pos
        amps_globales.append(amps_IA)

    # Promediar parametros sobre todos los keypoints
    Hs_med   = np.mean(Hs_lista)
    Tp_med   = np.mean(Tp_lista)
    Tm01_med = np.mean(Tm01_lista)
    Tm02_med = np.mean(Tm02_lista)

    print(f"\n  Parametros medios del GT:")
    print(f"  Hs   = {Hs_med:.3f} m  (GT: 6.0 m)")
    print(f"  Tp   = {Tp_med:.3f} s  (GT: 9.0 s)")
    print(f"  Tm01 = {Tm01_med:.3f} s")
    print(f"  Tm02 = {Tm02_med:.3f} s")

    # Espectro FFT promedio sobre todos los keypoints
    amps_media = np.mean(np.array(amps_globales), axis=0)
    omega_pos  = 2 * np.pi * freqs_globales
    E_fft      = 0.5 * amps_media**2

    # Densidad espectral FFT: S = E / dw
    dw_arr = np.diff(omega_pos)
    dw_arr = np.append(dw_arr, dw_arr[-1])
    S_fft  = E_fft / dw_arr

    # JONSWAP analitico con parametros estimados
    omega_teo  = np.linspace(omega_pos.min(), omega_pos.max(), 600)
    S_jon_8_12 = jonswap_8_12(omega_teo, Hs_med, Tm01_med)
    S_jon_8_11 = jonswap_8_11(omega_teo, Hs_med, Tp_med)

    # ── Grafica por keypoint S(w) vs w ───────────────────────────
    print("\nGenerando graficas por keypoint...")
    generated_kp_files = []
    for kp_id in range(z_all.shape[1]):
        z = z_all[:, kp_id] - np.mean(z_all[:, kp_id])
        N = len(z)

        fft_vals  = np.fft.fft(z)
        freqs     = np.fft.fftfreq(N, d=config.DELTA_T)
        idx_pos   = np.where(freqs > 0)[0]
        freqs_pos = freqs[idx_pos]
        amps_IA   = (2.0 / N) * np.abs(fft_vals[idx_pos])

        omega_kp = 2 * np.pi * freqs_pos
        E_kp     = 0.5 * amps_IA**2
        dw_kp    = np.diff(omega_kp)
        dw_kp    = np.append(dw_kp, dw_kp[-1])
        S_kp     = E_kp / dw_kp

        Hs_kp   = Hs_lista[kp_id]
        Tp_kp   = Tp_lista[kp_id]
        Tm01_kp = Tm01_lista[kp_id]

        omega_teo_kp  = np.linspace(omega_kp.min(), omega_kp.max(), 600)
        S_jon_kp_8_12 = jonswap_8_12(omega_teo_kp, Hs_kp, Tm01_kp)
        S_jon_kp_8_11 = jonswap_8_11(omega_teo_kp, Hs_kp, Tp_kp)

        idx_pico_kp = np.argmax(S_kp)

        plt.figure(figsize=(8, 4))
        plt.plot(omega_teo_kp, S_jon_kp_8_12, color='black', linewidth=2,
                 label=f'JONSWAP ec.8-12 (Tm01={Tm01_kp:.2f}s)')
        plt.plot(omega_teo_kp, S_jon_kp_8_11, color='gray',  linewidth=1.5, linestyle='--',
                 label=f'JONSWAP ec.8-11 (Tp={Tp_kp:.2f}s)')
        plt.plot(omega_kp, S_kp, color='teal', linewidth=1.5,
                 label=f'GT FFT KP_{kp_id:02d}')
        plt.axvline(x=omega_kp[idx_pico_kp], color='crimson', linestyle='--', alpha=0.6,
                    label=f'wp = {omega_kp[idx_pico_kp]:.3f} rad/s')
        plt.xlim(0, 3.5)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.title(f'Espectro FFT vs JONSWAP - {etiquetas_kp[kp_id]}')
        plt.xlabel('Frecuencia angular, w (rad/s)')
        plt.ylabel('S(w) (m^2 s/rad)')
        texto = (f"Hs = {Hs_kp:.2f} m\nTp = {Tp_kp:.2f} s\n"
                 f"Tm01 = {Tm01_kp:.2f} s")
        plt.gca().text(0.70, 0.65, texto, transform=plt.gca().transAxes,
                       bbox=dict(facecolor='white', alpha=0.9,
                                 edgecolor='gray', boxstyle='round,pad=0.5'),
                       fontsize=9)
        plt.legend(loc='upper left', fontsize=7)
        plt.tight_layout()
        out_file = os.path.join(config.DIR_SALIDA_ENERGIA_GT,
                                f"GT_S_vs_w_kp_{kp_id:02d}.png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        generated_kp_files.append(out_file)
        plt.close()

    # Borrar cualquier gráfica antigua de keypoints que no pertenezca a los 24 nodos actuales
    for old_file in glob.glob(os.path.join(config.DIR_SALIDA_ENERGIA_GT, "GT_S_vs_w_kp_*.png")):
        if os.path.abspath(old_file) not in {os.path.abspath(path) for path in generated_kp_files}:
            try:
                os.remove(old_file)
            except OSError:
                pass

    # ── Guardar tabla resumen ────────────────────────────────────
    pd.DataFrame({
        'Keypoint_ID': etiquetas_kp,
        'Hs_m':        Hs_lista,
        'Tp_s':        Tp_lista,
        'Tm01_s':      Tm01_lista,
        'Tm02_s':      Tm02_lista,
    }).to_csv(os.path.join(config.DIR_SALIDA_ENERGIA_GT,
              "resumen_parametros_gt.csv"), index=False, sep=';')

    print(f"\nOK Todo guardado en: {config.DIR_SALIDA_ENERGIA_GT}")