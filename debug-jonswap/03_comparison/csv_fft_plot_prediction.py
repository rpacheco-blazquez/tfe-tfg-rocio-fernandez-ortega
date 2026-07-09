"""
csv_fft_plot_prediction.py
Lee z_metros.csv (serie temporal de 24 keypoints, frame a 0.48 s),
aplica FFT 1D sobre cada keypoint y calcula el espectro promedio de
los 24 keypoints. Compara con:
  - El espectro analítico JONSWAP de debug_analytical_spectrum.csv (Hs=6m, Tm=9s)
  - Un espectro JONSWAP generado con los parámetros Hs, Tm estimados desde la FFT

Uso:
  python csv_fft_plot_prediction.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Añadir el directorio raíz del proyecto al path para importar config
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import config

# ============================================================
# CONSTANTES
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_Z_METROS = os.path.join(SCRIPT_DIR, "..", "z_metros.csv")
CSV_ANALYTICAL = os.path.join(
    SCRIPT_DIR, "..", "01_validation_1d", "debug_analytical_spectrum.csv"
)

DELTA_T = config.DELTA_T  # 0.48 s
FILTRO_FREC_MAX_HZ = config.FILTRO_FREC_MAX_HZ

print(f"DELTA_T = {DELTA_T} s")
print(f"FILTRO_FREC_MAX_HZ = {FILTRO_FREC_MAX_HZ} Hz")

# ============================================================
# 1. LEER z_metros.csv
# ============================================================
df_z = pd.read_csv(CSV_Z_METROS, sep=";")
print(f"\nDimensiones z_metros.csv: {df_z.shape}")
print(f"Columnas: {df_z.columns.tolist()}")

# Extraer columnas de keypoints (kp_00 a kp_23)
kp_cols = [c for c in df_z.columns if c.startswith("kp_")]
print(f"Keypoints detectados: {len(kp_cols)}")

# Matriz z_all: (N_frames, N_kp)
z_all = df_z[kp_cols].values.astype(float)
N_frames, N_kp = z_all.shape
print(f"N_frames = {N_frames}, N_kp = {N_kp}")

# Vector de tiempo
tiempo = np.arange(N_frames) * DELTA_T
T_total = N_frames * DELTA_T

print(f"T_total = {T_total:.1f} s")
print(
    f"z_all stats: min={z_all.min():.3f}, max={z_all.max():.3f}, std={z_all.std():.3f} m"
)

# ============================================================
# 2. FFT 1D POR KEYPOINT (sin usar fft_analisis.py, numpy puro)
# ============================================================
# Centrar cada keypoint
z_centrada = z_all - np.mean(z_all, axis=0, keepdims=True)

# FFT sobre cada keypoint
N = N_frames
dt = DELTA_T
df_hz = 1.0 / (N * dt)
dw = 2.0 * np.pi * df_hz

print(f"\n--- Parámetros FFT ---")
print(f"N = {N}, dt = {dt} s, T_total = {N*dt:.1f} s")
print(f"df = {df_hz:.4f} Hz, dw = {dw:.4f} rad/s")

# Calcular FFT para cada KP
amps_all = []  # lista de arrays de amplitudes
freqs_pos = None
omega_pos = None

for kp_idx in range(N_kp):
    z = z_centrada[:, kp_idx]
    fft_vals = np.fft.fft(z)
    freqs = np.fft.fftfreq(N, d=dt)

    if kp_idx == 0:
        idx_pos = np.where(freqs > 0)[0]
        freqs_pos = freqs[idx_pos]
        omega_pos = 2.0 * np.pi * freqs_pos

    amps = (2.0 / N) * np.abs(fft_vals[idx_pos])
    amps_all.append(amps)

# Convertir a array (N_kp, N_freqs_pos)
amps_all = np.array(amps_all)

# Densidad espectral S(ω) para cada KP
S_all = (amps_all**2) / (2.0 * dw)  # shape: (N_kp, N_freqs_pos)

# Promedio de S(ω) sobre todos los KPs
S_mean = np.mean(S_all, axis=0)
# Desviación estándar entre KPs
S_std = np.std(S_all, axis=0)

# Amplitud promedio (para calcular Hs, Tp, etc.)
amps_mean = np.mean(amps_all, axis=0)

print(f"\nS_mean stats: min={S_mean.min():.4f}, max={S_mean.max():.4f}")

# ============================================================
# 3. PARÁMETROS ESPECTRALES (del promedio)
# ============================================================
idx_filt = np.where(freqs_pos <= FILTRO_FREC_MAX_HZ)[0]
amps_filt = amps_mean[idx_filt]
freqs_filt = freqs_pos[idx_filt]

# Hs, Tm01, Tp del promedio de KPs
suma_A2 = np.sum(amps_filt**2)
Hs_pred = np.sqrt(8.0 * suma_A2)
m0 = 0.5 * suma_A2
m1 = np.sum(0.5 * amps_filt**2 * freqs_filt)
m2 = np.sum(0.5 * amps_filt**2 * freqs_filt**2)
Tm01_pred = m0 / m1 if m1 > 0 else 0
Tm02_pred = np.sqrt(m0 / m2) if m2 > 0 else 0

idx_pico = np.argmax(amps_mean)
f_pico = freqs_pos[idx_pico]
Tp_pred = 1.0 / f_pico if f_pico > 0 else 0

print(f"\n--- Parámetros espectrales (promedio 24 KPs) ---")
print(f"Hs   = {Hs_pred:.3f} m")
print(f"Tp   = {Tp_pred:.3f} s  (fp = {f_pico:.4f} Hz)")
print(f"Tm01 = {Tm01_pred:.3f} s")
print(f"Tm02 = {Tm02_pred:.3f} s")
print(f"4*σ  = {4 * np.mean(np.std(z_centrada, axis=0)):.3f} m  (estimación temporal)")

# ============================================================
# 4. LEER ESPECTRO ANALÍTICO JONSWAP (Hs=6m, Tm=9s)
# ============================================================
if os.path.exists(CSV_ANALYTICAL):
    df_anal = pd.read_csv(CSV_ANALYTICAL, sep=";")
    omega_anal = df_anal["omega_rad_s"].values
    S_anal = df_anal["S_m2_s_rad"].values
    Hs_anal = 6.0
    Tp_anal = 2.0 * np.pi / omega_anal[np.argmax(S_anal)]
    Tm01_anal = 9.0
    print(f"\n--- Espectro analítico cargado ---")
    print(
        f"Hs_anal={Hs_anal:.1f} m, Tp_anal={Tp_anal:.2f} s, Tm01_anal={Tm01_anal:.1f} s"
    )
else:
    print(f"\n⚠️  No se encontró {CSV_ANALYTICAL}")
    omega_anal, S_anal = None, None


# ============================================================
# 5. GENERAR ESPECTRO JONSWAP CON LOS PARÁMETROS ESTIMADOS
#    (para comparar forma espectral con mismos Hs, Tm)
# ============================================================
def jonswap_8_12(omega, Hs, Tm):
    """JONSWAP ec. 8-12 SeaFEM: S(ω) = (155*Hs²/(Tm⁴*ω⁵)) * 3.3^Y * exp(-944*Tm⁻⁴*ω⁻⁴)"""
    Tm = float(Tm)
    if Tm <= 0:
        return np.zeros_like(omega)
    sigma = np.where(omega <= 5.24 / Tm, 0.07, 0.09)
    Y = np.exp(-(((0.191 * omega * Tm - 1) / (sigma * np.sqrt(2))) ** 2))
    S = (
        (155.0 * Hs**2 / (Tm**4 * omega**5))
        * (3.3**Y)
        * np.exp(-944.0 * Tm ** (-4) * omega ** (-4))
    )
    return np.maximum(S, 0)


# JONSWAP con parámetros estimados de la predicción
omega_smooth = np.linspace(0.01, omega_pos.max(), 600)
S_jonswap_pred = jonswap_8_12(omega_smooth, Hs_pred, Tm01_pred)

print(f"\n--- JONSWAP generado con parámetros estimados ---")
print(f"Hs={Hs_pred:.2f} m, Tm01={Tm01_pred:.2f} s")

# ============================================================
# 6. PLOT COMPARATIVO
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

# ── Panel 1: S(ω) vs ω ──
ax = axes[0]

# Espectro FFT promedio (puntos)
ax.fill_between(
    omega_pos,
    S_mean - S_std,
    S_mean + S_std,
    color="steelblue",
    alpha=0.2,
    label=f"±1σ entre KPs",
)
ax.plot(
    omega_pos,
    S_mean,
    color="steelblue",
    linewidth=1.5,
    alpha=0.9,
    label=f"FFT promedio 24 KPs  |  Hs={Hs_pred:.2f}m  Tp={Tp_pred:.2f}s  Tm01={Tm01_pred:.2f}s",
)

# JONSWAP con parámetros estimados (línea discontinua)
ax.plot(
    omega_smooth,
    S_jonswap_pred,
    color="darkorange",
    linewidth=2.5,
    linestyle="--",
    label=f"JONSWAP (Hs={Hs_pred:.2f}, Tm01={Tm01_pred:.2f}) — misma Hs/Tm",
)

# JONSWAP analítico original (Hs=6, Tm=9) si existe
if omega_anal is not None:
    ax.plot(
        omega_anal,
        S_anal,
        color="black",
        linewidth=2.5,
        label=f"JONSWAP analítico  |  Hs={Hs_anal:.1f}m  Tm01={Tm01_anal:.1f}s",
    )

ax.set_xlabel("Frecuencia angular ω (rad/s)", fontsize=11)
ax.set_ylabel("Densidad espectral S(ω) (m²·s/rad)", fontsize=11)
ax.set_title(
    f"Comparación FFT predicción (promedio 24 KPs) vs JONSWAP\n"
    f"N={N}, dt={dt}s, T_total={T_total:.1f}s",
    fontsize=13,
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_xlim(0, omega_pos.max())

# Línea vertical en fp
ax.axvline(x=2 * np.pi * f_pico, color="steelblue", linestyle=":", alpha=0.5)

# ── Panel 2: A(ω) vs ω (amplitudes, no densidad) ──
ax2 = axes[1]
amps_std = np.std(amps_all, axis=0)
ax2.fill_between(
    omega_pos,
    amps_mean - amps_std,
    amps_mean + amps_std,
    color="teal",
    alpha=0.2,
)
ax2.plot(
    omega_pos, amps_mean, color="teal", linewidth=1.2, label=f"Amplitud promedio 24 KPs"
)
ax2.axvline(
    x=2 * np.pi * f_pico,
    color="crimson",
    linestyle="--",
    label=f"fp = {f_pico:.4f} Hz → Tp = {Tp_pred:.2f} s",
)
ax2.axvspan(
    2 * np.pi * FILTRO_FREC_MAX_HZ,
    omega_pos.max(),
    color="gray",
    alpha=0.15,
    label=f"Ruido (>{FILTRO_FREC_MAX_HZ} Hz)",
)
ax2.set_xlabel("Frecuencia angular ω (rad/s)", fontsize=11)
ax2.set_ylabel("Amplitud A(ω) (m)", fontsize=11)
ax2.set_title("Amplitudes FFT (promedio 24 KPs)", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.set_xlim(0, omega_pos.max())

# ── Panel 3: Serie temporal de un KP de ejemplo (kp_00) y del promedio ──
ax3 = axes[2]
z_promedio = np.mean(z_centrada, axis=1)  # promedio de los 24 KPs en cada frame
ax3.plot(
    tiempo,
    z_centrada[:, 0],
    color="royalblue",
    linewidth=0.5,
    alpha=0.6,
    label="KP_00 (ejemplo)",
)
ax3.plot(
    tiempo,
    z_promedio,
    color="crimson",
    linewidth=1.5,
    label="Promedio 24 KPs (±1σ sombreado)",
)
std_temporal = np.std(z_centrada, axis=1)
ax3.fill_between(
    tiempo,
    z_promedio - std_temporal,
    z_promedio + std_temporal,
    color="crimson",
    alpha=0.15,
)
ax3.set_xlabel("Tiempo (s)", fontsize=11)
ax3.set_ylabel("Elevación z(t) (m)", fontsize=11)
ax3.set_title(
    f"Serie temporal — KP_00 y promedio 24 KPs\n"
    f"4·σ_promedio = {4*np.std(z_promedio):.2f} m",
    fontsize=13,
)
ax3.legend(fontsize=9)
ax3.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
ruta_fig = os.path.join(SCRIPT_DIR, "csv_fft_plot_prediction.png")
plt.savefig(ruta_fig, dpi=200, bbox_inches="tight")
print(f"\n✅ Figura guardada: {ruta_fig}")
plt.show()
