"""
compare_nfreqs.py (FIX: dt real por dataset + 4ª columna z_metros)
Genera dos simulaciones sintéticas (10 y 499 frecuencias), calcula
FFT en ambas y compara en 4 columnas:
  - FFT numérico (azul)
  - JONSWAP "target" (negro): Hs=6, TM_FORMULA=9 -> da Tm01~9s
  - JONSWAP "estimado" (naranja): Hs_pred, Tm01_pred derivados de la FFT

FILTRO ESPECTRAL AUTOMÁTICO:
  T_MIN y T_MAX ya NO se fijan a mano. Se calculan automáticamente a
  partir del propio CSV del CFD (alturanodos_CFD_TFG.csv):
    - T_MIN = 2 * dt_cfd      (periodo de Nyquist del muestreo real)
    - T_MAX = N_cfd * dt_cfd  (duración total del registro CFD)
  Estos mismos límites se aplican a TODOS los datasets (10 frec,
  499 frec, CFD original y z_metros.csv) para que la comparación
  sea homogénea.

FIX IMPORTANTE:
  compute_fft() ya NO asume dt=DT=0.48s para todos los datasets.
  Ahora recibe el dt real de cada uno como parámetro (dt_data),
  para que el eje de frecuencias (y por tanto la integral m0/Hs/Tm01/Tm02)
  se calcule correctamente incluso si el dt real difiere de 0.48s.

NUEVA 4ª COLUMNA:
  Se añade una columna extra en el plot para z_metros.csv (la
  predicción), calculada exactamente igual que las otras tres.

Uso:
  python compare_nfreqs_fixed.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# PARÁMETROS FÍSICOS (idénticos a generate_synthetic_2d_waves.py)
# ============================================================
G = 9.81
HS_TARGET = 6.0
TM_FORMULA = 9.0  # parametro Tm de la formula JONSWAP -> da Tm01~9s
GAMMA = 3.3
THETA0 = 0.0
SPREAD_DEG = 90.0  # ancho total del spreading en grados (±SPREAD_DEG/2)
SPREADING_S = 3  # exponente de cos²ˢ (soporte compacto)
DT = 0.48
N_FRAMES = 1000
LX, LY = 100.0, 100.0
GRID_ROWS, GRID_COLS = 3, 8


# ============================================================
# FUNCIONES
# ============================================================
def jonswap_8_12(omega, Hs, Tm):
    Tm = float(Tm)
    omega = np.maximum(omega, 1e-10)
    sigma = np.where(omega <= 5.24 / Tm, 0.07, 0.09)
    Y = np.exp(-(((0.191 * omega * Tm - 1) / (sigma * np.sqrt(2))) ** 2))
    S = (
        (155.0 * Hs**2 / (Tm**4 * omega**5))
        * (GAMMA**Y)
        * np.exp(-944.0 * Tm ** (-4) * omega ** (-4))
    )
    return np.maximum(S, 0)


def spreading_cos2s(theta, theta0, s, spread_deg):
    dtheta = theta - theta0
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    half_width = np.radians(spread_deg / 2.0)
    t = dtheta / half_width
    D_raw = np.where(np.abs(t) <= 1.0, np.cos(np.pi / 2.0 * t) ** (2 * s), 0.0)
    dtheta_step = theta[1] - theta[0]
    Z = np.sum(D_raw) * dtheta_step
    return D_raw / Z if Z > 0 else D_raw


def generate_synthetic(n_freqs_target, seed=42):
    """Genera synthetic_z_metros con n_freqs_target frecuencias."""
    np.random.seed(seed)

    # Posiciones de los 24 puntos
    x_pos = np.linspace(0, LX, GRID_COLS)
    y_pos = np.linspace(0, LY, GRID_ROWS)
    XX, YY = np.meshgrid(x_pos, y_pos)
    x_pts, y_pts = XX.flatten(), YY.flatten()

    # Frecuencias
    freqs_fft = np.fft.fftfreq(N_FRAMES, d=DT)
    idx_pos_fft = np.where(freqs_fft > 0)[0]
    all_freqs_pos = freqs_fft[idx_pos_fft]

    if n_freqs_target < len(all_freqs_pos):
        f_min_target, f_max_target = 0.04, 0.30
        idx_range = np.where(
            (all_freqs_pos >= f_min_target) & (all_freqs_pos <= f_max_target)
        )[0]
        step = max(1, len(idx_range) // n_freqs_target)
        idx_selected = idx_range[::step][:n_freqs_target]
        freqs_gen = all_freqs_pos[idx_selected]
    else:
        freqs_gen = all_freqs_pos

    omega_gen = 2.0 * np.pi * freqs_gen
    n_freqs = len(freqs_gen)
    df_gen = freqs_gen[1] - freqs_gen[0] if n_freqs > 1 else 1.0 / (N_FRAMES * DT)
    dw_gen = 2.0 * np.pi * df_gen

    # Direcciones
    N_DIRS = 144
    theta_edges = np.linspace(-np.pi, np.pi, N_DIRS + 1)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    dtheta = theta_centers[1] - theta_centers[0]

    # Espectro
    S_1d = jonswap_8_12(omega_gen, HS_TARGET, TM_FORMULA)
    D_theta = spreading_cos2s(theta_centers, THETA0, SPREADING_S, SPREAD_DEG)
    E_2d = np.outer(S_1d, D_theta)
    A_ij = np.sqrt(2.0 * E_2d * dw_gen * dtheta)

    # Números de onda (deep water)
    k_i = omega_gen**2 / G

    # Fases aleatorias
    phi_ij = np.random.uniform(0, 2.0 * np.pi, (n_freqs, N_DIRS))
    cos_theta = np.cos(theta_centers)
    sin_theta = np.sin(theta_centers)

    # Tiempo
    t = np.arange(N_FRAMES) * DT
    omega_t = np.outer(omega_gen, t)
    cos_omega_t = np.cos(omega_t)
    sin_omega_t = np.sin(omega_t)

    print(
        f"  Generando con {n_freqs} frecuencias × {N_DIRS} dirs = {n_freqs*N_DIRS} componentes..."
    )
    z_all = np.zeros((N_FRAMES, 24))
    for p in range(24):
        xp, yp = x_pts[p], y_pts[p]
        spatial_term = xp * cos_theta + yp * sin_theta
        psi_ij = np.outer(k_i, spatial_term) + phi_ij
        C_i = np.sum(A_ij * np.cos(psi_ij), axis=1)
        S_i = np.sum(A_ij * np.sin(psi_ij), axis=1)
        eta_t = np.sum(
            C_i[:, np.newaxis] * cos_omega_t + S_i[:, np.newaxis] * sin_omega_t, axis=0
        )
        z_all[:, p] = eta_t

    # CSV
    df_out = pd.DataFrame()
    df_out["frame"] = np.arange(1, N_FRAMES + 1)
    for kp in range(24):
        df_out[f"kp_{kp:02d}"] = z_all[:, kp]

    fname = os.path.join(SCRIPT_DIR, f"synthetic_z_metros_{n_freqs}freq.csv")
    df_out.to_csv(fname, index=False, sep=";")

    z_std = np.mean(np.std(z_all, axis=0))
    print(f"  -> {fname}  (4*sigma = {4*z_std:.2f} m)")
    return z_all, n_freqs


def compute_fft(z_data, label, t_min=None, t_max=None, dt_data=DT):
    """
    Calcula el espectro FFT promedio sobre los KPs y estima Hs, Tp,
    Tm01, Tm02.

    dt_data: paso temporal REAL del dataset que se está analizando.
    Antes este valor estaba fijado internamente a DT=0.48s para
    TODOS los datasets, lo cual desescalaba el eje de frecuencias
    (y por tanto m0/Hs/Tm01/Tm02) para cualquier dataset cuyo dt
    real fuese distinto de 0.48s (p.ej. el CFD). Ahora se recibe
    como parámetro explícito.

    Si se proporcionan t_min/t_max, se filtra la banda de frecuencias
    usada para calcular los momentos espectrales (m0, m1, m2) y por
    tanto Hs_pred, Tm01_pred, Tm02_pred:
        f_min = 1 / t_max
        f_max = min(1 / t_min, f_nyquist_del_dataset)
    Si t_min/t_max son None, se usa todo el rango positivo (sin filtro).
    """
    z_centrada = z_data - np.mean(z_data, axis=0, keepdims=True)
    N_data = z_data.shape[0]
    N_kp = z_data.shape[1]
    amps_all = []
    for kp in range(N_kp):
        fft_vals = np.fft.fft(z_centrada[:, kp])
        freqs = np.fft.fftfreq(N_data, d=dt_data)
        if kp == 0:
            idx_pos = np.where(freqs > 0)[0]
            freqs_pos = freqs[idx_pos]
            omega_pos = 2.0 * np.pi * freqs_pos
        amps_all.append((2.0 / N_data) * np.abs(fft_vals[idx_pos]))
    amps_all = np.array(amps_all)
    df_hz_local = 1.0 / (N_data * dt_data)
    dw_local = 2.0 * np.pi * df_hz_local
    S_all = (amps_all**2) / (2.0 * dw_local)
    S_mean = np.mean(S_all, axis=0)
    S_std = np.std(S_all, axis=0)
    amps_mean = np.mean(amps_all, axis=0)

    # ── Filtro por Tmin/Tmax (derivados del CFD), limitado por Nyquist propio ──
    f_nyquist = freqs_pos.max()
    f_min = 1.0 / t_max if t_max is not None else freqs_pos.min()
    f_max = 1.0 / t_min if t_min is not None else f_nyquist
    f_max = min(f_max, f_nyquist)  # nunca superar Nyquist del propio dataset

    idx_filt = np.where((freqs_pos >= f_min) & (freqs_pos <= f_max))[0]
    amps_filt_all_kps = amps_all[:, idx_filt]  # (N_kp, N_freqs_filt)
    freqs_filt = freqs_pos[idx_filt]

    if t_min is not None and t_max is not None:
        print(
            f"  dt_data={dt_data:.4f}s | Filtro: T=[{t_min:.2f},{t_max:.2f}]s -> f=[{f_min:.4f},{f_max:.4f}]Hz "
            f"({len(idx_filt)}/{len(freqs_pos)} componentes)"
        )

    # ✅ CORRECTO: m0 por KP, luego promediar (NO promediar amplitudes antes)
    suma_A2_per_kp = np.sum(amps_filt_all_kps**2, axis=1)  # (N_kp,)
    suma_A2 = np.mean(suma_A2_per_kp)  # promedio de varianzas
    Hs_pred = np.sqrt(8.0 * suma_A2)
    m0 = 0.5 * suma_A2
    # m1, m2: también por KP
    m1_per_kp = np.sum(0.5 * amps_filt_all_kps**2 * freqs_filt[np.newaxis, :], axis=1)
    m2_per_kp = np.sum(
        0.5 * amps_filt_all_kps**2 * freqs_filt[np.newaxis, :] ** 2, axis=1
    )
    m1 = np.mean(m1_per_kp)
    m2 = np.mean(m2_per_kp)
    Tm01_pred = m0 / m1 if m1 > 0 else 0
    Tm02_pred = np.sqrt(m0 / m2) if m2 > 0 else 0

    # Para plots: amps_mean solo para visualización (SIN filtrar, banda completa)
    amps_mean = np.mean(amps_all, axis=0)
    idx_pico = np.argmax(amps_mean)
    Tp_pred = 1.0 / freqs_pos[idx_pico] if freqs_pos[idx_pico] > 0 else 0
    Hs_4sigma = 4.0 * np.mean(np.std(z_centrada, axis=0))

    print(f"\n{label}:")
    print(
        f"  Hs_FFT={Hs_pred:.3f}m, Hs_4sig={Hs_4sigma:.3f}m, Tp={Tp_pred:.3f}s, Tm01={Tm01_pred:.3f}s"
    )

    return {
        "omega_pos": omega_pos,
        "S_mean": S_mean,
        "S_std": S_std,
        "amps_mean": amps_mean,
        "amps_all": amps_all,
        "Hs_pred": Hs_pred,
        "Tp_pred": Tp_pred,
        "Tm01_pred": Tm01_pred,
        "Tm02_pred": Tm02_pred,
        "z_centrada": z_centrada,
        "freqs_pos": freqs_pos,
        "idx_pico": idx_pico,
        "Hs_4sigma": Hs_4sigma,
        "dt_data": dt_data,
    }


def get_dt_from_index(df, fallback_dt, label):
    """
    Intenta deducir el dt real a partir del índice del DataFrame,
    asumiendo que el índice representa tiempo en segundos.
    Si el índice parece ser un contador de frames (1,2,3...) en vez
    de tiempo real, usa fallback_dt y avisa por consola.
    """
    idx_vals = df.index.values.astype(float)
    diffs = np.diff(idx_vals)
    dt_detected = np.mean(diffs)

    # Heurística simple: si el índice sube de 1 en 1 (frame counter),
    # no es tiempo real -> usar fallback_dt
    if np.allclose(diffs, 1.0, atol=1e-6):
        print(
            f"  [{label}] índice parece ser contador de frames (paso=1.0), "
            f"NO tiempo real. Se usa dt={fallback_dt}s por defecto."
        )
        return fallback_dt
    else:
        print(f"  [{label}] dt real detectado desde índice = {dt_detected:.4f}s")
        return dt_detected


# ============================================================
# GENERAR AMBOS CSVs SINTÉTICOS
# ============================================================
print("=" * 60)
print("GENERANDO SIMULACIONES SINTÉTICAS")
print("=" * 60)
print(f"Hs_target={HS_TARGET} m, TM_FORMULA={TM_FORMULA} -> Tm01~9s")
print(
    f"gamma={GAMMA}, heading={np.degrees(THETA0):.0f} deg, spreading=cos^{2*SPREADING_S}, width={SPREAD_DEG} deg"
)
print()

print("[1/2] 10 frecuencias...")
z_10, _ = generate_synthetic(10, seed=42)
print()
print("[2/2] 499 frecuencias...")
z_499, _ = generate_synthetic(499, seed=42)

# ============================================================
# CARGAR alturanodos_CFD_TFG.csv ORIGINAL (necesario ANTES de las FFT
# para deducir T_MIN / T_MAX a partir del dt y duración reales del CFD)
# ============================================================
print("\n[3/5] alturanodos_CFD_TFG.csv original...")
CSV_ORIG_DIR = os.path.dirname(config.RUTA_GT_ALTURA_CSV)
CSV_ORIG = os.path.join(CSV_ORIG_DIR, "alturanodos_CFD_TFG.csv")
# El CSV viene con índice temporal y columnas de nodos, por ejemplo: 959, 960, 961, ...
df_orig = pd.read_csv(CSV_ORIG, sep=",", header=0, index_col=0)

# Usar exactamente los 24 nodos definidos en config.py, en el mismo orden
# que usa el pipeline para la GT.
columnas = list(df_orig.columns)
selected_nodes = []
for node in config.LISTA_KEYPOINTS_ORDENADOS:
    if node in columnas:
        selected_nodes.append(node)
    elif str(node) in columnas:
        selected_nodes.append(str(node))
    else:
        raise KeyError(f"No se encontró el nodo {node} en alturanodos_CFD_TFG.csv")

z_orig = df_orig[selected_nodes].values.astype(float)
print(f"  N_frames={z_orig.shape[0]}, N_kp={z_orig.shape[1]}")
print(f"  Nodos usados: {selected_nodes}")

# ── Deducir dt real del CFD a partir de su índice temporal ──
t_index = df_orig.index.values.astype(float)
dt_cfd = np.mean(np.diff(t_index))
N_cfd = z_orig.shape[0]

T_MIN = 2.0 * dt_cfd   # periodo de Nyquist del muestreo real del CFD
T_MAX = N_cfd * dt_cfd  # duración total del registro CFD

print(f"  dt_cfd={dt_cfd:.4f}s, N_cfd={N_cfd} -> T_MIN={T_MIN:.2f}s, T_MAX={T_MAX:.2f}s")

# ============================================================
# CARGAR z_metros.csv (PREDICCIÓN) ANTES DE LAS FFT
# para deducir su dt real igual que se hizo con el CFD
# ============================================================
print("\n[4/5] z_metros.csv (predicción)...")
CSV_ZMETROS = config.RUTA_Z_METROS_PRED
df_zmetros = pd.read_csv(CSV_ZMETROS, sep=";", header=0, index_col=0)
cols_zmetros = [f"kp_{kp:02d}" for kp in range(24)]
z_zmetros = df_zmetros[cols_zmetros].values.astype(float)
print(f"  N_frames={z_zmetros.shape[0]}, N_kp={z_zmetros.shape[1]}")

# dt real de z_metros.csv: se intenta deducir del índice; si el índice
# es un contador de frames (1,2,3...) se usa DT=0.48 como en los sintéticos.
dt_zmetros = get_dt_from_index(df_zmetros, fallback_dt=DT, label="z_metros.csv")

# ============================================================
# FFT SOBRE TODOS LOS DATASETS, CON EL MISMO FILTRO T_MIN/T_MAX
# CADA UNO CON SU dt REAL (FIX)
# ============================================================
print("\n" + "=" * 60)
print("ANÁLISIS FFT (con filtro T_MIN/T_MAX derivado del CFD, dt real por dataset)")
print("=" * 60)

res_10 = compute_fft(z_10, "10 frecuencias", t_min=T_MIN, t_max=T_MAX, dt_data=DT)
res_499 = compute_fft(z_499, "499 frecuencias", t_min=T_MIN, t_max=T_MAX, dt_data=DT)
res_orig = compute_fft(
    z_orig, "alturanodos_CFD_TFG.csv original", t_min=T_MIN, t_max=T_MAX, dt_data=dt_cfd
)
print("\n[5/5] FFT z_metros.csv (predicción)...")
res_zmetros = compute_fft(
    z_zmetros, "z_metros.csv (predicción)", t_min=T_MIN, t_max=T_MAX, dt_data=dt_zmetros
)

# ============================================================
# CARGAR ESPECTRO ANALÍTICO DE REFERENCIA
# ============================================================
CSV_ANAL = os.path.join(
    SCRIPT_DIR, "..", "01_validation_1d", "debug_analytical_spectrum.csv"
)
df_anal = pd.read_csv(CSV_ANAL, sep=";")
omega_anal = df_anal["omega_rad_s"].values
S_anal = df_anal["S_m2_s_rad"].values

# ============================================================
# PLOT 4 COLUMNAS: 10 frec | 499 frec | CFD original | z_metros (predicción)
# ============================================================
fig, axes = plt.subplots(3, 4, figsize=(30, 15))
fig.subplots_adjust(top=0.84, bottom=0.06, left=0.04, right=0.99, hspace=0.40, wspace=0.24)

datasets = [
    (res_10, "10 frecuencias\n(sintético)"),
    (res_499, "499 frecuencias\n(sintético)"),
    (res_orig, "alturanodos_CFD_TFG.csv\n(GT)"),
    (res_zmetros, "z_metros.csv\n(predicción)"),
]

for col, (res, title) in enumerate(datasets):
    omega_pos = res["omega_pos"]
    S_mean = res["S_mean"]
    S_std = res["S_std"]
    Hs_p = res["Hs_pred"]
    Tp_p = res["Tp_pred"]
    Tm01_p = res["Tm01_pred"]
    Tm02_p = res["Tm02_pred"]
    amps_mean = res["amps_mean"]
    amps_all = res["amps_all"]
    z_centrada = res["z_centrada"]
    freqs_pos = res["freqs_pos"]
    dt_used = res["dt_data"]
    f_pico = freqs_pos[res["idx_pico"]]

    omega_smooth = np.linspace(0.01, omega_pos.max(), 600)

    # ── JONSWAP "target" (Hs=6, TM_FORMULA=9) ──
    S_jonswap_target = jonswap_8_12(omega_smooth, HS_TARGET, TM_FORMULA)

    # ── JONSWAP "estimado" (Hs_pred, Tm01_pred) ──
    S_jonswap_est = jonswap_8_12(omega_smooth, Hs_p, Tm01_p)

    # ── Panel 1: S(ω) ──
    ax = axes[0, col]
    ax.fill_between(
        omega_pos, S_mean - S_std, S_mean + S_std, color="steelblue", alpha=0.2
    )
    ax.plot(
        omega_pos,
        S_mean,
        color="steelblue",
        linewidth=1.2,
        label=f"FFT numérico | Hs={Hs_p:.2f} Tp={Tp_p:.1f} Tm01={Tm01_p:.2f}",
    )
    ax.plot(
        omega_smooth,
        S_jonswap_target,
        color="black",
        linewidth=2.5,
        label=f"JONSWAP target (Hs={HS_TARGET}, TM_formula={TM_FORMULA})",
    )
    ax.plot(
        omega_smooth,
        S_jonswap_est,
        color="darkorange",
        linewidth=2.0,
        linestyle="--",
        label=f"JONSWAP estimado (Hs={Hs_p:.2f}, Tm01={Tm01_p:.2f})",
    )
    # ── Líneas verticales mostrando la banda de filtro T_MIN/T_MAX ──
    f_min_plot = 1.0 / T_MAX
    f_max_plot = min(1.0 / T_MIN, omega_pos.max() / (2 * np.pi))
    ax.axvline(x=2 * np.pi * f_min_plot, color="green", linestyle=":", alpha=0.6)
    ax.axvline(
        x=2 * np.pi * f_max_plot,
        color="green",
        linestyle=":",
        alpha=0.6,
        label=f"Filtro T=[{T_MIN:.1f},{T_MAX:.1f}]s",
    )
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("S(ω) (m²·s/rad)")
    ax.set_title(
        f"{title}  (dt={dt_used:.3f}s)\n"
        f"Hs_FFT={Hs_p:.2f}m  Hs_4sig={res['Hs_4sigma']:.2f}m  "
        f"Tp={Tp_p:.1f}s  Tm01={Tm01_p:.2f}s  Tm02={Tm02_p:.2f}s",
        fontsize=10,
    )
    ax.legend(fontsize=6.5)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlim(0, omega_pos.max())
    ax.axvline(x=2 * np.pi * f_pico, color="steelblue", linestyle=":", alpha=0.4)

    # ── Panel 2: Amplitudes ──
    ax2 = axes[1, col]
    amps_std = np.std(amps_all, axis=0)
    ax2.fill_between(
        omega_pos, amps_mean - amps_std, amps_mean + amps_std, color="teal", alpha=0.2
    )
    ax2.plot(omega_pos, amps_mean, color="teal", linewidth=1.0)
    ax2.axvline(
        x=2 * np.pi * f_pico,
        color="crimson",
        linestyle="--",
        label=f"fp={f_pico:.4f} Hz",
    )
    ax2.set_xlabel("ω (rad/s)")
    ax2.set_ylabel("A(ω) (m)")
    ax2.set_title(f"Amplitudes FFT — {title}")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_xlim(0, omega_pos.max())

    # ── Panel 3: Serie temporal ──
    ax3 = axes[2, col]
    z_prom = np.mean(z_centrada, axis=1)
    t_vec = np.arange(z_centrada.shape[0]) * dt_used
    ax3.plot(
        t_vec,
        z_centrada[:, 0],
        color="royalblue",
        linewidth=0.4,
        alpha=0.5,
        label="KP_00",
    )
    ax3.plot(t_vec, z_prom, color="crimson", linewidth=1.2, label="Promedio 24 KPs")
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("z(t) (m)")
    ax3.set_title(f"Serie temporal  |  4·σ_prom={4*np.std(z_prom):.2f} m")
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle=":", alpha=0.5)

# ── Línea de referencia analítica en los 4 paneles superiores ──
for col in range(4):
    axes[0, col].plot(
        omega_anal,
        S_anal,
        color="gray",
        linewidth=0.8,
        alpha=0.4,
        label="debug_analytical (Hs=6, Tm01=9)",
    )
    axes[0, col].legend(fontsize=6)

plt.tight_layout()
ruta = os.path.join(SCRIPT_DIR, "compare_nfreqs.png")
plt.savefig(ruta, dpi=200, bbox_inches="tight")
print(f"\n✅ {ruta}")
plt.show()