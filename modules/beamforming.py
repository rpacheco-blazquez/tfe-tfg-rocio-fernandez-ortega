"""
modules/beamforming.py
Estimacion direccional del oleaje por beamforming temporal.
Calcula el espectro de potencia en funcion de la direccion usando
correlacion cruzada entre todas las parejas de keypoints.
El color del diagrama polar representa la amplitud A en metros,
igual que en los demas diagramas polares del pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import combinations

import config


def _cargar_coordenadas_kp():
    df_x = pd.read_csv(os.path.join(config.RUTA_CSVS, "X.csv"))
    nodos = df_x['nodo'].astype(int).tolist()
    coords_x = np.array([df_x.loc[df_x['nodo'] == n, 'x'].values[0] for n in nodos])
    coords_y = np.array([df_x.loc[df_x['nodo'] == n, 'y'].values[0] for n in nodos])
    return coords_x, coords_y


def _cargar_z_pred():
    df_z = pd.read_csv(config.RUTA_Z_METROS_PRED, sep=';')
    cols_kp = [c for c in df_z.columns if c.startswith('kp_')]
    z_all = df_z[cols_kp].values
    n_frames = z_all.shape[0]
    tiempo = np.arange(n_frames) * config.DELTA_T + 30.0
    return z_all, tiempo


def _cargar_z_gt():
    from modules.lectura_gt import cargar_z_gt
    return cargar_z_gt()


def beamforming_por_frecuencia(z_all, tiempo, coords_x, coords_y, n_dirs=360):
    """
    Calcula el espectro direccional S(f, theta) por beamforming.

    Para cada frecuencia f y direccion theta:
      1. Calcula el desfase esperado entre cada pareja (i,j):
            delta_phi = -k * (dx*cos(theta) + dy*sin(theta))
         donde k = 2*pi*f/c_f es el numero de onda
      2. Correlaciona espectralmente las senales con ese desfase
      3. La suma sobre todas las parejas da la potencia en (f, theta)

    El resultado se pondera por la energia espectral de cada frecuencia
    (m^2 por bin de frecuencia) para obtener S(f,theta) en m^2.
    Luego se convierte a amplitud A = sqrt(2 * S * df * dtheta) en metros.

    Devuelve:
      thetas      : array de direcciones en radianes
      freqs_pos   : array de frecuencias en Hz
      A_f_theta   : matriz (n_freqs, n_dirs) de amplitud en metros
    """
    g        = 9.81
    n_kp     = z_all.shape[1]
    n_frames = z_all.shape[0]
    dt       = config.DELTA_T

    # Centrar cada keypoint
    z = z_all.copy()
    for i in range(n_kp):
        z[:, i] -= np.mean(z[:, i])

    # FFT de cada keypoint
    Z_fft = np.fft.fft(z, axis=0)                # (n_frames, n_kp)
    freqs = np.fft.fftfreq(n_frames, d=dt)
    idx_pos = np.where((freqs > 0) & (freqs <= config.FILTRO_FREC_MAX_HZ))[0]
    freqs_pos = freqs[idx_pos]
    df_freq   = freqs_pos[1] - freqs_pos[0] if len(freqs_pos) > 1 else 1.0

    parejas = list(combinations(range(n_kp), 2))
    thetas  = np.linspace(-np.pi, np.pi, n_dirs, endpoint=False)
    dtheta  = thetas[1] - thetas[0]

    print(f"  Parejas de keypoints: {len(parejas)}")
    print(f"  Frecuencias validas:  {len(freqs_pos)}")
    print(f"  Direcciones:          {n_dirs}")

    # Energia espectral por frecuencia: E(f) = (2/N^2) * sum_kp |Z_fft(f,kp)|^2
    # Promedio sobre keypoints, en m^2 por bin de frecuencia
    E_f = (2.0 / n_frames**2) * np.mean(np.abs(Z_fft[idx_pos, :])**2, axis=1)

    # Beamforming: S_beam(f, theta) adimensional normalizado
    S_beam = np.zeros((len(freqs_pos), n_dirs))

    for f_idx, f in enumerate(freqs_pos):
        if f <= 0:
            continue
        c_f = g / (2 * np.pi * f)   # velocidad de fase (m/s)
        k   = 2 * np.pi * f / c_f   # numero de onda (rad/m)

        for t_idx, theta in enumerate(thetas):
            suma = 0.0 + 0j
            for (i, j) in parejas:
                dx = coords_x[j] - coords_x[i]
                dy = coords_y[j] - coords_y[i]

                # Signo negativo: convenio olas viajando desde theta
                delta_phi = -k * (dx * np.cos(theta) + dy * np.sin(theta))

                cross = Z_fft[idx_pos[f_idx], i] * np.conj(Z_fft[idx_pos[f_idx], j])
                suma += cross * np.exp(1j * delta_phi)

            S_beam[f_idx, t_idx] = np.abs(suma)

    # Normalizar S_beam por filas para obtener distribucion direccional
    for f_idx in range(len(freqs_pos)):
        row_sum = S_beam[f_idx, :].sum()
        if row_sum > 0:
            S_beam[f_idx, :] /= row_sum   # distribucion direccional: suma=1 por frecuencia

    # S(f, theta) en m^2/rad = E(f) * S_beam(f,theta) / dtheta
    S_f_theta = S_beam * E_f[:, np.newaxis] / dtheta

    # Amplitud por bin: A = sqrt(2 * S * df * dtheta)  [metros]
    # Ecuacion 8-4 SeaFEM: A_ij = sqrt(2 * S(w,theta) * Dw * Dtheta)
    A_f_theta = np.sqrt(np.maximum(2.0 * S_f_theta * df_freq * dtheta, 0))

    print(f"  A_f_theta rango: {A_f_theta.min():.4f}m -- {A_f_theta.max():.4f}m")

    return thetas, freqs_pos, A_f_theta


def _dibujar_rosa_beamforming(thetas, freqs, A_f_theta, dir_salida, nombre, titulo,
                               A_ref_min=None, A_ref_max=None):
    """
    Dibuja el espectro polar del beamforming con color = amplitud A en metros.
    A_ref_min, A_ref_max: rango de referencia para el colormap (para comparacion con GT).
    """
    os.makedirs(dir_salida, exist_ok=True)

    T_vals = 1.0 / freqs
    mask_T = (T_vals >= config.T_BIN_MIN) & (T_vals <= config.T_BIN_MAX)
    T_plot = T_vals[mask_T]
    A_plot_raw = A_f_theta[mask_T, :]

    dir_bins = np.linspace(-np.pi, np.pi, config.N_BINS_DIR + 1)
    T_bins   = np.linspace(config.T_BIN_MIN, config.T_BIN_MAX, config.N_BINS_T + 1)

    G_out, T_out, A_out = [], [], []

    for d_i in range(config.N_BINS_DIR):
        for t_i in range(config.N_BINS_T):
            mask_d = (thetas >= dir_bins[d_i]) & (thetas < dir_bins[d_i+1])
            mask_t = (T_plot >= T_bins[t_i]) & (T_plot < T_bins[t_i+1])

            if np.sum(mask_d) > 0 and np.sum(mask_t) > 0:
                val = A_plot_raw[np.ix_(mask_t, mask_d)].mean()
                if val > 0:
                    G_out.append((dir_bins[d_i] + dir_bins[d_i+1]) / 2)
                    T_out.append((T_bins[t_i]   + T_bins[t_i+1])   / 2)
                    A_out.append(val)

    G_out = np.array(G_out)
    T_out = np.array(T_out)
    A_out = np.array(A_out)

    print(f"  Bins con datos: {len(G_out)}")

    # Colormap en metros
    vmin = A_ref_min if A_ref_min is not None else A_out.min()
    vmax = A_ref_max if A_ref_max is not None else A_out.max()
    norm_A = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap   = cm.plasma
    ancho  = (dir_bins[1] - dir_bins[0]) * 0.9

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(9, 9))

    for i in range(len(G_out)):
        ax.bar(G_out[i], T_out[i], width=ancho, bottom=0,
               color=cmap(norm_A(A_out[i])), alpha=0.9,
               linewidth=0.2, edgecolor="none")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(45)
    ax.set_rticks([3, 5, 7, 9])
    ax.set_yticklabels(["3s", "5s", "7s", "9s"], fontsize=8)
    ax.set_xticks(np.radians(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{a}°" for a in np.arange(0, 360, 30)], fontsize=8)
    ax.set_title(titulo, fontsize=12, fontweight="bold", pad=20)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm_A)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, fraction=0.04, label="Amplitud A (m)")
    plt.tight_layout()

    ruta = os.path.join(dir_salida, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  OK Polar beamforming guardada: {ruta}")
    return G_out, T_out, A_out


def calcular_beamforming(fuente="pred", dir_salida=None):
    """
    Calcula el espectro direccional por beamforming temporal y genera
    el diagrama polar con color = amplitud A en metros.

    fuente: "pred" para datos de prediccion, "gt" para groundtruth
    """
    print(f"\n{'='*55}")
    print(f"  BEAMFORMING TEMPORAL -- fuente: {fuente.upper()}")
    print(f"{'='*55}")

    coords_x, coords_y = _cargar_coordenadas_kp()

    if fuente == "pred":
        z_all, tiempo = _cargar_z_pred()
        if dir_salida is None:
            dir_salida = os.path.join(config.DIR_SALIDA_FFT_PRED, "BEAMFORMING")
        titulo = ("Espectro Direccional Prediccion IA (Beamforming)\n"
                  "Angulo=Direccion . Radio=Periodo . Color=Amplitud A (m)")
        nombre = "espectro_polar_beamforming_pred.png"
    else:
        z_all, tiempo = _cargar_z_gt()
        if dir_salida is None:
            dir_salida = os.path.join(config.DIR_SALIDA_FFT_GT, "BEAMFORMING")
        titulo = ("Espectro Direccional Groundtruth (Beamforming)\n"
                  "Angulo=Direccion . Radio=Periodo . Color=Amplitud A (m)")
        nombre = "espectro_polar_beamforming_gt.png"

    print(f"  Frames: {z_all.shape[0]}  Keypoints: {z_all.shape[1]}")
    print(f"  z rango: {z_all.min():.3f}m -- {z_all.max():.3f}m")

    thetas, freqs, A_f_theta = beamforming_por_frecuencia(
        z_all, tiempo, coords_x, coords_y, n_dirs=360)

    os.makedirs(dir_salida, exist_ok=True)

    # Guardar CSV
    df_out = pd.DataFrame(
        A_f_theta,
        index=np.round(freqs, 5),
        columns=np.round(np.degrees(thetas), 2)
    )
    df_out.to_csv(os.path.join(dir_salida, f"beamforming_{fuente}.csv"), sep=';')

    # Dibujar polar
    _dibujar_rosa_beamforming(thetas, freqs, A_f_theta, dir_salida, nombre, titulo)

    print(f"  OK Beamforming completado para {fuente.upper()}")

    return thetas, freqs, A_f_theta


def generar_espectro_energiaomnidireccional(thetas, freqs, A_f_theta,
                                             dir_salida, fuente="pred",
                                             Hs_ref=6.0, Tm_ref=9.0):
    """
    Genera el espectro omnidireccional S(w) integrando S(w, theta) sobre
    todas las direcciones (ecuacion 8-7 del manual SeaFEM):

        S(w) = integral_{-pi}^{pi} S(w, alpha) d_alpha
             ~ suma_j S(w_i, theta_j) * Dtheta

    Y lo compara con el espectro JONSWAP analitico (ecuacion 8-12 SeaFEM):

        S(w) = (155*Hs^2 / (Tm^4 * w^5)) * 3.3^Y * e^(-944*Tm^-4*w^-4)

    Parametros:
      thetas     : array de direcciones en radianes (de beamforming)
      freqs      : array de frecuencias en Hz (de beamforming)
      A_f_theta  : matriz (n_freqs, n_dirs) de amplitud en metros
      dir_salida : carpeta donde guardar la grafica
      fuente     : "pred" o "gt" (para nombre del archivo)
      Hs_ref     : Hs de referencia para el JONSWAP (m)
      Tm_ref     : Tm01 de referencia para el JONSWAP (s)
    """
    os.makedirs(dir_salida, exist_ok=True)

    dtheta  = thetas[1] - thetas[0]
    df_freq = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    omega   = 2 * np.pi * freqs

    # Reconstruir S(f, theta) en m^2/rad desde A_f_theta
    # A = sqrt(2 * S * df * dtheta)  =>  S = A^2 / (2 * df * dtheta)
    S_f_theta = A_f_theta**2 / (2.0 * df_freq * dtheta)

    # Integrar sobre theta: S(w) = suma_j S(w_i, theta_j) * Dtheta
    # Convertir S de m^2/rad_espacial a m^2*s/rad_temporal
    # S(f,theta)*df*dtheta = S(w,theta)*dw*dtheta  =>  S(w) = S(f)/(2*pi)
    S_omni_f = np.sum(S_f_theta, axis=1) * dtheta      # m^2/Hz integrado en theta -> m^2
    S_omni_w = S_omni_f / (2 * np.pi)                   # convertir a m^2*s/rad

    # Verificar con Hs
    m0 = np.trapezoid(S_omni_w, x=omega)
    Hs_estimada = 4.0 * np.sqrt(m0)
    print(f"  S(w) omnidireccional integrado:")
    print(f"  m0          = {m0:.4f} m^2")
    print(f"  Hs estimada = {Hs_estimada:.3f} m  (referencia: {Hs_ref:.1f} m)")

    # JONSWAP analitico ecuacion 8-12
    def jonswap_8_12(w, Hs, Tm):
        sigma = np.where(w <= 5.24 / Tm, 0.07, 0.09)
        Y     = np.exp(-((0.191 * w * Tm - 1) / (sigma * np.sqrt(2)))**2)
        S     = (155.0 * Hs**2 / (Tm**4 * w**5)) * (3.3**Y) * np.exp(-944.0 * Tm**(-4) * w**(-4))
        return S

    omega_teo  = np.linspace(omega.min(), omega.max(), 600)
    S_jon      = jonswap_8_12(omega_teo, Hs_ref, Tm_ref)

    # Grafica
    plt.figure(figsize=(10, 5))
    plt.plot(omega_teo, S_jon,   color='black',     linewidth=2.5,
             label=f'JONSWAP ec.8-12 (Hs={Hs_ref:.1f}m, Tm={Tm_ref:.1f}s)')
    plt.plot(omega,     S_omni_w, color='limegreen', linewidth=2,
             label=f'Beamforming omnidireccional (Hs_est={Hs_estimada:.2f}m)')

    idx_pico = np.argmax(S_omni_w)
    plt.axvline(x=omega[idx_pico], color='crimson', linestyle='--', alpha=0.7,
                label=f'wp = {omega[idx_pico]:.3f} rad/s  ->  Tp = {2*np.pi/omega[idx_pico]:.2f}s')

    plt.title(f'Espectro Energetico Omnidireccional S(w) -- {fuente.upper()}\n'
              f'Integracion sobre todas las direcciones (ec. 8-7 SeaFEM)')
    plt.xlabel('Frecuencia angular, w (rad/s)')
    plt.ylabel('Densidad Espectral S(w) (m^2 s/rad)')
    plt.xlim(0, 3.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    ruta = os.path.join(dir_salida, f"espectro_omnidireccional_{fuente}.png")
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  OK Espectro omnidireccional guardado: {ruta}")

    # Guardar CSV
    pd.DataFrame({
        'omega_rad_s': omega,
        'S_omni_m2srad': S_omni_w
    }).to_csv(os.path.join(dir_salida, f"espectro_omnidireccional_{fuente}.csv"),
              index=False, sep=';')

    return omega, S_omni_w