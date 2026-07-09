"""
debug_spectrum_numeric_write_csv.py
Genera una serie temporal sintética a partir del espectro JONSWAP (ec. 8-12)
y guarda:
  1. debug_numeric_timeseries.csv  → (Tiempo_s, z_m) — serie temporal
  2. debug_analytical_spectrum.csv → (omega_rad_s, S_m2_s_rad) — espectro analítico
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# 1. ESPECTRO JONSWAP ANALÍTICO — Ecuación 8-12 SeaFEM
# ============================================================
def jonswap_8_12(omega, Hs, Tm):
    """
    S(w) = (155*Hs^2 / (Tm^4 * w^5)) * 3.3^Y * exp(-944*Tm^-4*w^-4)
    """
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


# ============================================================
# PARÁMETROS
# ============================================================
Hs = 6.0
Tm = 9.0
T_total = 100.0
dt = 0.48  # 0.48 # 5

N = int(T_total / dt)
T_total_actual = N * dt

# ============================================================
# 2. ESPECTRO NUMÉRICO → SERIE TEMPORAL vía IFFT (escalado corregido)
# ============================================================
freqs_full = np.fft.fftfreq(N, d=dt)
omega_full = 2.0 * np.pi * np.abs(freqs_full)
S_full = jonswap_8_12(omega_full, Hs, Tm)
S_full[omega_full == 0] = 0.0

df = 1.0 / T_total_actual
dw = 2.0 * np.pi * df
A_full = np.sqrt(2.0 * S_full * dw)

np.random.seed(42)
n_pos = N // 2
phases = np.random.uniform(0, 2 * np.pi, n_pos)

spectrum_complex = np.zeros(N, dtype=complex)

# Positivas (k=1..N/2-1): X[k] = (A * N/2) * exp(iφ)
for k in range(1, n_pos):
    spectrum_complex[k] = (A_full[k] * N / 2.0) * np.exp(1j * phases[k])

# DC
spectrum_complex[0] = A_full[0] * N

# Nyquist (si N par)
if N % 2 == 0:
    spectrum_complex[N // 2] = A_full[N // 2] * N

# Negativas: conjugadas
spectrum_complex[n_pos + 1 :] = np.conj(spectrum_complex[1:n_pos][::-1])

tiempo = np.arange(N) * dt
z_t = np.real(np.fft.ifft(spectrum_complex))

print(f"N={N}  T_total={T_total_actual:.1f}s  dt={dt}s  df={df:.4f}Hz")
print(
    f"z(t): min={z_t.min():.3f}  max={z_t.max():.3f}  std={z_t.std():.3f}  4*std={4*z_t.std():.2f}m"
)

# ============================================================
# 3. GUARDAR CSV DE SERIE TEMPORAL
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

df_ts = pd.DataFrame({"Tiempo_s": tiempo, "z_m": z_t})
ruta_ts = os.path.join(base_dir, "debug_numeric_timeseries.csv")
df_ts.to_csv(ruta_ts, index=False, sep=";")
print(f"✅ {ruta_ts} guardado")

# ============================================================
# 4. GUARDAR CSV DEL ESPECTRO ANALÍTICO (para postprocesado)
# ============================================================
omega_teo = np.linspace(0.01, 2.5, 600)
S_teo = jonswap_8_12(omega_teo, Hs, Tm)

df_spec = pd.DataFrame({"omega_rad_s": omega_teo, "S_m2_s_rad": S_teo})
ruta_spec = os.path.join(base_dir, "debug_analytical_spectrum.csv")
df_spec.to_csv(ruta_spec, index=False, sep=";")
print(f"✅ {ruta_spec} guardado")
print(f"   omega: {omega_teo.min():.4f} – {omega_teo.max():.4f} rad/s")
print(f"   S_max: {S_teo.max():.4f} m²·s/rad")
