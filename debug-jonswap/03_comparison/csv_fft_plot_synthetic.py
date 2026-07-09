"""
csv_fft_plot_synthetic.py
Versión de csv_fft_plot_prediction.py adaptada para leer
synthetic_z_metros.csv (generado por generate_synthetic_2d_waves.py).
Compara el espectro FFT promedio contra el JONSWAP analítico (Hs=6m, Tm=9s).
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
CSV_Z_METROS = os.path.join(
    SCRIPT_DIR, "..", "02_synthetic_2d", "synthetic_z_metros.csv"
)
CSV_ANALYTICAL = os.path.join(
    SCRIPT_DIR, "..", "01_validation_1d", "debug_analytical_spectrum.csv"
)

DELTA_T = config.DELTA_T
FILTRO_FREC_MAX_HZ = config.FILTRO_FREC_MAX_HZ

print(f"CSV: {CSV_Z_METROS}")
print(f"DELTA_T = {DELTA_T} s, FILTRO={FILTRO_FREC_MAX_HZ} Hz")

# ── Leer ──
df_z = pd.read_csv(CSV_Z_METROS, sep=";")
kp_cols = [c for c in df_z.columns if c.startswith("kp_")]
z_all = df_z[kp_cols].values.astype(float)
N_frames, N_kp = z_all.shape
tiempo = np.arange(N_frames) * DELTA_T
print(f"N_frames={N_frames}, N_kp={N_kp}, T_total={N_frames*DELTA_T:.1f}s")
print(f"z stats: min={z_all.min():.3f}, max={z_all.max():.3f}, std={z_all.std():.3f} m")

# ── FFT ──
z_centrada = z_all - np.mean(z_all, axis=0, keepdims=True)
N, dt = N_frames, DELTA_T
df_hz = 1.0 / (N * dt)
dw = 2.0 * np.pi * df_hz
print(f"df={df_hz:.4f} Hz, dw={dw:.4f} rad/s")

amps_all = []
for kp_idx in range(N_kp):
    fft_vals = np.fft.fft(z_centrada[:, kp_idx])
    freqs = np.fft.fftfreq(N, d=dt)
    if kp_idx == 0:
        idx_pos = np.where(freqs > 0)[0]
        freqs_pos = freqs[idx_pos]
        omega_pos = 2.0 * np.pi * freqs_pos
    amps_all.append((2.0 / N) * np.abs(fft_vals[idx_pos]))

amps_all = np.array(amps_all)
S_all = (amps_all**2) / (2.0 * dw)
S_mean = np.mean(S_all, axis=0)
S_std = np.std(S_all, axis=0)
amps_mean = np.mean(amps_all, axis=0)

# ── Parámetros ──
idx_filt = np.where(freqs_pos <= FILTRO_FREC_MAX_HZ)[0]
amps_filt = amps_mean[idx_filt]
freqs_filt = freqs_pos[idx_filt]

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

print(f"\n--- FFT promedio 24 KPs ---")
print(
    f"Hs={Hs_pred:.3f}m, Tp={Tp_pred:.3f}s, Tm01={Tm01_pred:.3f}s, Tm02={Tm02_pred:.3f}s"
)
print(f"4*σ temporal = {4*np.mean(np.std(z_centrada, axis=0)):.3f} m")

# ── JONSWAP analítico ──
df_anal = pd.read_csv(CSV_ANALYTICAL, sep=";")
omega_anal = df_anal["omega_rad_s"].values
S_anal = df_anal["S_m2_s_rad"].values


# ── JONSWAP con parámetros estimados ──
def jonswap_8_12(omega, Hs, Tm):
    Tm = float(Tm)
    omega = np.maximum(omega, 1e-10)
    sigma = np.where(omega <= 5.24 / Tm, 0.07, 0.09)
    Y = np.exp(-(((0.191 * omega * Tm - 1) / (sigma * np.sqrt(2))) ** 2))
    S = (
        (155.0 * Hs**2 / (Tm**4 * omega**5))
        * (3.3**Y)
        * np.exp(-944.0 * Tm ** (-4) * omega ** (-4))
    )
    return np.maximum(S, 0)


omega_smooth = np.linspace(0.01, omega_pos.max(), 600)
S_jonswap_pred = jonswap_8_12(omega_smooth, Hs_pred, Tm01_pred)

# ── PLOT ──
fig, axes = plt.subplots(3, 1, figsize=(14, 14))

ax = axes[0]
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
    label=f"FFT promedio 24 KPs | Hs={Hs_pred:.2f}m Tp={Tp_pred:.2f}s Tm01={Tm01_pred:.2f}s",
)
ax.plot(
    omega_smooth,
    S_jonswap_pred,
    color="darkorange",
    linewidth=2.5,
    linestyle="--",
    label=f"JONSWAP estimado (Hs={Hs_pred:.2f}, Tm01={Tm01_pred:.2f})",
)
ax.plot(
    omega_anal,
    S_anal,
    color="black",
    linewidth=3,
    label="JONSWAP analítico (Hs=6.0, Tm01=9.0)",
)
ax.set_xlabel("ω (rad/s)")
ax.set_ylabel("S(ω) (m²·s/rad)")
ax.set_title(
    f"Espectro FFT sintético (2D JONSWAP, heading=0°, spreading=±45°) vs Analítico\n"
    f"N={N}, dt={dt}s, T_total={N*dt:.1f}s"
)
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_xlim(0, omega_pos.max())
ax.axvline(x=2 * np.pi * f_pico, color="steelblue", linestyle=":", alpha=0.5)

ax2 = axes[1]
amps_std = np.std(amps_all, axis=0)
ax2.fill_between(
    omega_pos, amps_mean - amps_std, amps_mean + amps_std, color="teal", alpha=0.2
)
ax2.plot(omega_pos, amps_mean, color="teal", linewidth=1.2)
ax2.axvline(
    x=2 * np.pi * f_pico,
    color="crimson",
    linestyle="--",
    label=f"fp={f_pico:.4f}Hz → Tp={Tp_pred:.2f}s",
)
ax2.axvspan(2 * np.pi * FILTRO_FREC_MAX_HZ, omega_pos.max(), color="gray", alpha=0.15)
ax2.set_xlabel("ω (rad/s)")
ax2.set_ylabel("A(ω) (m)")
ax2.set_title("Amplitudes FFT")
ax2.legend(fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.set_xlim(0, omega_pos.max())

ax3 = axes[2]
z_promedio = np.mean(z_centrada, axis=1)
ax3.plot(
    tiempo, z_centrada[:, 0], color="royalblue", linewidth=0.5, alpha=0.6, label="KP_00"
)
ax3.plot(tiempo, z_promedio, color="crimson", linewidth=1.5, label="Promedio 24 KPs")
std_t = np.std(z_centrada, axis=1)
ax3.fill_between(
    tiempo, z_promedio - std_t, z_promedio + std_t, color="crimson", alpha=0.15
)
ax3.set_xlabel("Tiempo (s)")
ax3.set_ylabel("z(t) (m)")
ax3.set_title(f"Serie temporal sintética | 4·σ_prom={4*np.std(z_promedio):.2f}m")
ax3.legend(fontsize=9)
ax3.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
ruta = os.path.join(SCRIPT_DIR, "csv_fft_plot_synthetic.png")
plt.savefig(ruta, dpi=200, bbox_inches="tight")
print(f"\n✅ {ruta}")
plt.show()
