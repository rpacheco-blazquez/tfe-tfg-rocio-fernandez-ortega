"""
csv_fft_plot.py
Lee un CSV con serie temporal (Tiempo_s, z_m), aplica:
  A) FFT 1D con numpy puro (raw)
  B) FFT 1D usando fft_1d_por_keypoint de fft_analisis.py
  C) Compara ambas contra el espectro JONSWAP analítico (debug_analytical_spectrum.csv)

Uso:
  python csv_fft_plot.py [ruta_csv]

  Si no se pasa argumento, usa 'debug_numeric_timeseries.csv' por defecto.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Añadir el directorio raíz del proyecto al path para importar config y modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import config
from modules import fft_analisis

# ============================================================
# ARGUMENTOS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if len(sys.argv) >= 2:
    CSV_ENTRADA = sys.argv[1]
else:
    CSV_ENTRADA = os.path.join(SCRIPT_DIR, "debug_numeric_timeseries.csv")

CSV_ANALYTICAL = os.path.join(SCRIPT_DIR, "debug_analytical_spectrum.csv")
DIR_DUMMY = "_fftmod_output"  # subcarpeta dentro de debug-jonswap/

print(f"Leyendo serie temporal de: {CSV_ENTRADA}")
print(f"Directorio dummy para fft_1d_por_keypoint: {DIR_DUMMY}/")

# ============================================================
# 1. LEER SERIE TEMPORAL
# ============================================================
df = pd.read_csv(CSV_ENTRADA, sep=";")
tiempo = df["Tiempo_s"].values
z_1col = df["z_m"].values.astype(float)

N = len(z_1col)
dt = tiempo[1] - tiempo[0] if len(tiempo) > 1 else config.DELTA_T
df_hz = 1.0 / (N * dt)
dw = 2.0 * np.pi * df_hz

print(f"N={N}  dt={dt}s  T_total={N*dt:.1f}s  df={df_hz:.4f}Hz  dw={dw:.4f}rad/s")

# ============================================================
# 2. FFT RAW CON NUMPY (réplica exacta de fft_analisis.py líneas ~79-88)
# ============================================================
z_centrada_raw = z_1col - np.mean(z_1col)

fft_vals_raw = np.fft.fft(z_centrada_raw)
freqs_raw = np.fft.fftfreq(N, d=dt)
idx_pos_raw = np.where(freqs_raw > 0)[0]
freqs_pos_raw = freqs_raw[idx_pos_raw]
omega_pos_raw = 2.0 * np.pi * freqs_pos_raw
amps_raw = (2.0 / N) * np.abs(fft_vals_raw[idx_pos_raw])

# Convertir amplitudes a densidad espectral: S = A² / (2*dw)
S_numpy = (amps_raw**2) / (2.0 * dw)

# ── Parámetros espectrales FFT numpy raw (misma lógica que fft_analisis.py) ──
idx_filt_raw = np.where(freqs_pos_raw <= config.FILTRO_FREC_MAX_HZ)[0]
amps_filt_raw = amps_raw[idx_filt_raw]
freqs_filt_raw = freqs_pos_raw[idx_filt_raw]

suma_A2_np = np.sum(amps_filt_raw**2)
Hs_np = np.sqrt(8.0 * suma_A2_np)
m0_np = 0.5 * suma_A2_np
m1_np = np.sum(0.5 * amps_filt_raw**2 * freqs_filt_raw)
Tm01_np = m0_np / m1_np if m1_np > 0 else 0

idx_pico_np = np.argmax(amps_raw)
Tp_np = 1.0 / freqs_pos_raw[idx_pico_np] if freqs_pos_raw[idx_pico_np] > 0 else 0

print(f"FFT numpy raw — Hs={Hs_np:.2f}m  Tp={Tp_np:.2f}s  Tm01={Tm01_np:.2f}s")

# ============================================================
# 3. FFT USANDO fft_1d_por_keypoint de fft_analisis.py
#    La función espera (N_frames, 24 keypoints) → duplicamos la señal 24 veces.
# ============================================================
z_24cols = np.tile(z_1col.reshape(-1, 1), (1, config.NUM_KEYPOINTS))

# ⚠️  fft_1d_por_keypoint usa config.DELTA_T hardcodeado (0.48 s).
#     Para que funcione con cualquier dt, lo sobrescribimos temporalmente.
delta_t_original = config.DELTA_T
config.DELTA_T = dt

z_centrada_fftmod = fft_analisis.fft_1d_por_keypoint(
    z_all=z_24cols,
    tiempo_vector=tiempo,
    dir_salida=DIR_DUMMY,
    etiquetas_kp=None,
)

config.DELTA_T = delta_t_original  # restaurar

# Leer la salida del keypoint 00
ruta_fft_kp00 = os.path.join(DIR_DUMMY, "espectros", "fft_KP_00.csv")
df_kp = pd.read_csv(ruta_fft_kp00, sep=";")

freqs_fftmod = df_kp["Frecuencia_Hz"].values
amps_fftmod = df_kp["Amplitud_m"].values
omega_fftmod = 2.0 * np.pi * freqs_fftmod

# Convertir a densidad espectral (usando dw del propio dataset)
S_fftmod = (amps_fftmod**2) / (2.0 * dw)

# ── Parámetros espectrales de fft_1d_por_keypoint (leídos del resumen) ──
ruta_resumen = os.path.join(DIR_DUMMY, "resumen_global_keypoints.csv")
df_resumen = pd.read_csv(ruta_resumen, sep=";")
# Todos los KP son idénticos, cogemos KP_00
Hs_fft = df_resumen["Hs_m"].values[0]
Tp_fft = df_resumen["Tp_s"].values[0]
Tm01_fft = df_resumen["Tm01_s"].values[0]

print(
    f"FFT fft_1d_por_keypoint — Hs={Hs_fft:.2f}m  Tp={Tp_fft:.2f}s  Tm01={Tm01_fft:.2f}s"
)

# ============================================================
# 4. LEER ESPECTRO ANALÍTICO
# ============================================================
if not os.path.exists(CSV_ANALYTICAL):
    print(
        f"⚠️  No se encontró {CSV_ANALYTICAL}. Ejecuta primero debug_spectrum_numeric_write_csv.py"
    )
    sys.exit(1)

df_anal = pd.read_csv(CSV_ANALYTICAL, sep=";")
omega_anal = df_anal["omega_rad_s"].values
S_anal = df_anal["S_m2_s_rad"].values

# ── Parámetros del JONSWAP analítico ──
Hs_anal = 6.0  # Hs de entrada
Tm01_anal = 9.0  # Tm de entrada
idx_pico_anal = np.argmax(S_anal)
Tp_anal = 2.0 * np.pi / omega_anal[idx_pico_anal]  # Tp del pico espectral

print(
    f"JONSWAP analítico — Hs={Hs_anal:.1f}m  Tp={Tp_anal:.2f}s  Tm01={Tm01_anal:.1f}s"
)
print(f"  (Tp/Tm01 = {Tp_anal/Tm01_anal:.3f} — relación teórica JONSWAP γ=3.3)")

# ============================================================
# 5. PLOT COMPARATIVO
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Curva analítica (línea negra continua)
ax.plot(
    omega_anal,
    S_anal,
    color="black",
    linewidth=2.5,
    label=f"JONSWAP analítico  |  Hs={Hs_anal:.1f}  Tp={Tp_anal:.1f}  Tm01={Tm01_anal:.1f}",
)

# FFT numpy raw: cruces rojas sin línea
ax.scatter(
    omega_pos_raw,
    S_numpy,
    marker="x",
    color="crimson",
    s=30,
    alpha=0.8,
    zorder=5,
    label=f"FFT numpy raw  |  Hs={Hs_np:.2f}  Tp={Tp_np:.2f}  Tm01={Tm01_np:.2f}",
)

# FFT vía fft_1d_por_keypoint: círculos azules sin línea
ax.scatter(
    omega_fftmod,
    S_fftmod,
    marker="o",
    facecolors="none",
    edgecolors="royalblue",
    s=25,
    alpha=0.8,
    zorder=5,
    label=f"FFT fft_1d_por_keypoint  |  Hs={Hs_fft:.2f}  Tp={Tp_fft:.2f}  Tm01={Tm01_fft:.2f}",
)

ax.set_xlabel("Frecuencia angular ω (rad/s)", fontsize=12)
ax.set_ylabel("Densidad espectral S(ω) (m²·s/rad)", fontsize=12)
ax.set_title(
    f"Validación FFT 1D: analítico vs numpy vs fft_1d_por_keypoint\n"
    f"Hs=6m, Tm=9s, N={N}, dt={dt}s  |  CSV: {CSV_ENTRADA}",
    fontsize=13,
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle=":", alpha=0.4)
ax.set_xlim(0, 2.5)

plt.tight_layout()
plt.savefig("debug_csv_fft_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
print("\n✅ Figura guardada como debug_csv_fft_comparison.png")
