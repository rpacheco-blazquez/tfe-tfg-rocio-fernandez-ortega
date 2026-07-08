"""
modules/fft_analisis.py
Funciones reutilizables de FFT 1D (por keypoint) y NUFFT 3D (coordenadas
irregulares) para extraer Hs, Tp, Tm01, Tm02 y el espectro direccional.
Usado tanto para los datos de prediccion como para el groundtruth.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


def _cargar_coordenadas_kp():
    """Carga coordenadas reales X, Y de los 24 keypoints desde X.csv."""
    df_x = pd.read_csv(os.path.join(config.RUTA_CSVS, "X.csv"))
    nodos = df_x['nodo'].astype(int).tolist()
    coords_x = np.array([df_x.loc[df_x['nodo'] == n, 'x'].values[0] for n in nodos])
    coords_y = np.array([df_x.loc[df_x['nodo'] == n, 'y'].values[0] for n in nodos])
    return coords_x, coords_y


def fft_1d_por_keypoint(z_all, tiempo_vector, dir_salida, etiquetas_kp=None):
    """
    Calcula series temporales y FFT 1D para cada uno de los 24 keypoints.
    Guarda CSVs y PNGs de cada serie y espectro, mas un resumen global
    con Hs, Tp, Tm01, Tm02 por keypoint.

    z_all: array (N_frames, 24) en metros, SIN centrar
    etiquetas_kp: lista opcional de nombres para cada keypoint (para GT con nodo)
    """
    os.makedirs(os.path.join(dir_salida, "series"), exist_ok=True)
    os.makedirs(os.path.join(dir_salida, "espectros"), exist_ok=True)

    if etiquetas_kp is None:
        etiquetas_kp = [f"KP_{i:02d}" for i in range(config.NUM_KEYPOINTS)]

    z_centrada = z_all.copy()
    for kp_id in range(config.NUM_KEYPOINTS):
        z_centrada[:, kp_id] -= np.mean(z_centrada[:, kp_id])

    resultados_globales, resultados_max_onada = [], []

    print("\nProcesando series temporales y FFT 1D por keypoint...")
    for kp_id in range(config.NUM_KEYPOINTS):
        z, z_cal = z_centrada[:, kp_id], z_all[:, kp_id]
        etiqueta = etiquetas_kp[kp_id]
        N = len(z)

        idx_max = np.argmax(np.abs(z))
        h_max = np.abs(z[idx_max])
        tiempo_max = tiempo_vector[idx_max]
        resultados_max_onada.append({'Keypoint_ID': etiqueta, 'Hmax_m': h_max, 'Tiempo_s': tiempo_max})

        pd.DataFrame({
            'Tiempo_s': tiempo_vector, 'z_calibrada_m': z_cal, 'z_centrada_m': z
        }).to_csv(os.path.join(dir_salida, "series", f"serie_{etiqueta}.csv"), index=False, sep=';')

        plt.figure(figsize=(10, 3.5))
        plt.plot(tiempo_vector, z, color='royalblue', linewidth=1,
                 label=f'Elevacion centrada z(t) -- {etiqueta}')
        plt.title(f'Serie Temporal de Altura - {etiqueta}', fontsize=11)
        plt.xlabel('Tiempo (s)'); plt.ylabel('Altura z_m(t) (m)')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(os.path.join(dir_salida, "series", f"serie_{etiqueta}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        fft_vals = np.fft.fft(z)
        freqs = np.fft.fftfreq(N, d=config.DELTA_T)
        idx_pos = np.where(freqs > 0)[0]
        freqs_pos = freqs[idx_pos]
        amps_IA = (2.0 / N) * np.abs(fft_vals[idx_pos])

        idx_filt = np.where(freqs_pos <= config.FILTRO_FREC_MAX_HZ)[0]
        freqs_fil, amps_fil = freqs_pos[idx_filt], amps_IA[idx_filt]

        suma_A2 = np.sum(amps_fil ** 2)
        H_s = np.sqrt(8.0 * suma_A2)
        m0 = 0.5 * suma_A2
        m1 = np.sum(0.5 * amps_fil ** 2 * freqs_fil)
        m2 = np.sum(0.5 * amps_fil ** 2 * freqs_fil ** 2)
        T_m01 = m0 / m1 if m1 > 0 else 0
        T_m02 = np.sqrt(m0 / m2) if m2 > 0 else 0

        idx_pico = np.argmax(amps_IA)
        f_pico = freqs_pos[idx_pico]
        T_p = 1.0 / f_pico if f_pico > 0 else 0

        resultados_globales.append({
            'Keypoint_ID': etiqueta, 'm0': m0, 'm1': m1, 'm2': m2,
            'Hs_m': H_s, 'Tm01_s': T_m01, 'Tm02_s': T_m02, 'Tp_s': T_p, 'fp_Hz': f_pico
        })

        pd.DataFrame({'Frecuencia_Hz': freqs_pos, 'Amplitud_m': amps_IA}).to_csv(
            os.path.join(dir_salida, "espectros", f"fft_{etiqueta}.csv"), index=False, sep=';')

        plt.figure(figsize=(8, 4))
        plt.plot(freqs_pos, amps_IA, color='teal', label=f'Amplitudes FFT -- {etiqueta}')
        plt.axvline(x=f_pico, color='crimson', linestyle='--',
                    label=f'fp = {f_pico:.3f} Hz  ->  Tp = {T_p:.2f} s')
        plt.axvspan(config.FILTRO_FREC_MAX_HZ, freqs_pos.max(), color='gray', alpha=0.15,
                    label=f'Ruido filtrado (>{config.FILTRO_FREC_MAX_HZ} Hz)')
        plt.xlim(0, freqs_pos.max())
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.title(f'Espectro FFT - {etiqueta}')
        plt.xlabel('Frecuencia (Hz)'); plt.ylabel('Amplitud A_i (m)')
        texto = f"Hs = {H_s:.2f} m\nTp = {T_p:.2f} s\nTm01 = {T_m01:.2f} s\nTm02 = {T_m02:.2f} s"
        plt.gca().text(0.70, 0.65, texto, transform=plt.gca().transAxes,
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'),
                        fontsize=9)
        plt.legend(loc='upper left', fontsize=8)
        plt.savefig(os.path.join(dir_salida, "espectros", f"fft_{etiqueta}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    pd.DataFrame(resultados_globales).to_csv(
        os.path.join(dir_salida, "resumen_global_keypoints.csv"), index=False, sep=';')
    pd.DataFrame(resultados_max_onada).to_csv(
        os.path.join(dir_salida, "resumen_max_onada.csv"), index=False, sep=';')

    print("OK Series temporales y FFT 1D completadas.")
    return z_centrada


def fft_3d_direccional(z_centrada, coords_x_kp=None, coords_y_kp=None):
    """
    Calcula el espectro 3D eta(kx, ky, w) usando NUFFT3d1 (finufft),
    que trabaja con coordenadas espaciales irregulares -- no asume rejilla
    regular como fftn, por lo que es mas correcto para la malla real.

    La transformada calculada es:
      eta_hat(kx, ky, w) = sum_{j,t} eta(xj, yj, t) * e^{-i(kx*xj + ky*yj + w*t)}

    FINUFFT requiere coordenadas en [-pi, pi], se reescalan:
      x' = 2*pi * (x - x_min) / Lx  - pi
      y' = 2*pi * (y - y_min) / Ly  - pi
      t' = 2*pi * (t - t_min) / Lt  - pi

    Los numeros de onda reales kx, ky se recuperan a partir de los
    indices del espectro de salida y del dominio espacial.

    Devuelve dir_flat, T_flat, amp_flat (arrays planos, componentes validas).
    """
    import finufft

    n_frames = z_centrada.shape[0]
    n_kp     = z_centrada.shape[1]   # 24

    # Cargar coordenadas reales si no se pasan
    if coords_x_kp is None or coords_y_kp is None:
        coords_x_kp, coords_y_kp = _cargar_coordenadas_kp()

    tiempo = np.arange(n_frames) * config.DELTA_T + 30.0

    print(f"Coordenadas X: {coords_x_kp.min():.2f}m -- {coords_x_kp.max():.2f}m")
    print(f"Coordenadas Y: {coords_y_kp.min():.2f}m -- {coords_y_kp.max():.2f}m")
    print(f"Tiempo:        {tiempo.min():.2f}s -- {tiempo.max():.2f}s")

    # Reescalar coordenadas a [-pi, pi]
    Lx = coords_x_kp.max() - coords_x_kp.min()
    Ly = coords_y_kp.max() - coords_y_kp.min()
    Lt = tiempo.max()      - tiempo.min()

    x_norm = 2 * np.pi * (coords_x_kp - coords_x_kp.min()) / Lx - np.pi
    y_norm = 2 * np.pi * (coords_y_kp - coords_y_kp.min()) / Ly - np.pi
    t_norm = 2 * np.pi * (tiempo       - tiempo.min())      / Lt - np.pi

    # Construir arrays planos (x, y, t, eta)
    # Cada punto de entrada es un triplete (xj, yj, tj) con valor eta_j
    # Expandimos los N_KP puntos espaciales por todos los N_frames instantes
    x_all   = np.tile(x_norm, n_frames)           # (N_KP * N_frames,)
    y_all   = np.tile(y_norm, n_frames)
    t_all   = np.repeat(t_norm, n_kp)             # cada t repetido N_KP veces
    eta_all = z_centrada.flatten(order='C').astype(complex)  # (N_frames * N_KP,)

    print(f"Puntos NUFFT: {len(x_all)}")

    # Resolucion del espectro de salida
    # Nkx, Nky mayores que los 8x3 originales: NUFFT interpola mejorando
    # la resolucion direccional mas alla de lo que daria fftn
    Nkx = 64
    Nky = 64
    Nw  = n_frames

    print(f"Resolucion espectro: Nkx={Nkx}  Nky={Nky}  Nw={Nw}")
    print("Calculando NUFFT3d1...")

    # Llamada a NUFFT3d1
    # iflag=-1 para convenio estandar FFT: e^{-i*k*x}
    eta_hat = finufft.nufft3d1(
        x_all.astype(np.float64),
        y_all.astype(np.float64),
        t_all.astype(np.float64),
        eta_all,
        (Nkx, Nky, Nw),
        isign=-1,
        eps=1e-6
    )
    # eta_hat shape: (Nkx, Nky, Nw)

    # Amplitud normalizada
    amplitud_3d = (2.0 / n_frames) * np.abs(eta_hat)

    # Reconstruir frecuencias reales
    # Los indices NUFFT van de -N/2 a N/2-1 (igual que fftfreq)
    # Las frecuencias reales se recuperan deshaciendo la normalizacion:
    #   k_real = k_indice * (2*pi / L)
    ikx = np.fft.fftshift(np.fft.fftfreq(Nkx, d=1.0/Nkx))   # indices enteros
    iky = np.fft.fftshift(np.fft.fftfreq(Nky, d=1.0/Nky))
    iw  = np.fft.fftshift(np.fft.fftfreq(Nw,  d=1.0/Nw))

    kx_vals = ikx * (2 * np.pi / Lx)   # numero de onda real (rad/m)
    ky_vals = iky * (2 * np.pi / Ly)
    w_vals  = iw  * (2 * np.pi / Lt)   # frecuencia angular (rad/s)
    f_vals  = w_vals / (2 * np.pi)     # frecuencia en Hz

    # Aplicar fftshift al espectro para centrar el cero
    amplitud_3d = np.fft.fftshift(amplitud_3d)

    # Meshgrid de frecuencias
    KX, KY, FT = np.meshgrid(kx_vals, ky_vals, f_vals, indexing='ij')

    # Filtro: solo frecuencias temporales positivas y fisicamente validas
    mask = (FT > 0) & (FT <= config.FILTRO_FREC_MAX_HZ)

    amp_flat = amplitud_3d[mask]
    kx_flat  = KX[mask]
    ky_flat  = KY[mask]
    ft_flat  = FT[mask]

    dir_flat = np.arctan2(ky_flat, kx_flat)
    T_flat   = 1.0 / ft_flat

    print(f"Componentes validas: {len(amp_flat)}")
    print(f"Direcciones: {np.degrees(dir_flat.min()):.1f} -- {np.degrees(dir_flat.max()):.1f} grados")
    print(f"Periodos:    {T_flat.min():.2f}s -- {T_flat.max():.2f}s")
    print(f"Amplitudes:  {amp_flat.min():.6f}m -- {amp_flat.max():.6f}m")

    return dir_flat, T_flat, amp_flat


def agrupar_en_bins(dir_flat, T_flat, amp_flat):
    """Agrupa las componentes FFT 3D en bins de direccion y periodo."""
    dir_bins = np.linspace(-np.pi, np.pi, config.N_BINS_DIR + 1)
    T_bins = np.linspace(config.T_BIN_MIN, config.T_BIN_MAX, config.N_BINS_T + 1)

    G_plot, T_plot, A_plot = [], [], []
    for d_i in range(config.N_BINS_DIR):
        for t_i in range(config.N_BINS_T):
            mask_bin = ((dir_flat >= dir_bins[d_i]) & (dir_flat < dir_bins[d_i + 1]) &
                        (T_flat >= T_bins[t_i]) & (T_flat < T_bins[t_i + 1]))
            if np.sum(mask_bin) > 0:
                G_plot.append((dir_bins[d_i] + dir_bins[d_i + 1]) / 2)
                T_plot.append((T_bins[t_i] + T_bins[t_i + 1]) / 2)
                A_plot.append(np.mean(amp_flat[mask_bin]))

    print(f"Bins con datos: {len(G_plot)}")
    return np.array(G_plot), np.array(T_plot), np.array(A_plot), dir_bins, T_bins