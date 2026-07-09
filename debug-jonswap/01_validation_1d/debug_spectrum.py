"""
debug_spectrum.py
Script de validación del pipeline FFT 1D:
  1. Genera el espectro JONSWAP analítico (ec. 8-12) con Hs=6m, Tm=9s
  2. Genera una serie temporal sintética vía IFFT a partir del espectro
  3. Aplica FFT 1D para recuperar el espectro numérico
  4. Compara ambos: línea analítica vs puntos numéricos S(w) vs w
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. ESPECTRO JONSWAP ANALÍTICO — Ecuación 8-12 SeaFEM
# ============================================================
def jonswap_8_12(omega, Hs, Tm):
    """
    S(w) = (155*Hs^2 / (Tm^4 * w^5)) * 3.3^Y * exp(-944*Tm^-4*w^-4)
    Y = exp(-[(0.191*w*Tm - 1) / (sigma*sqrt(2))]^2)
    sigma = 0.07 si w <= 5.24/Tm, 0.09 si w > 5.24/Tm
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
Hs = 6.0  # altura significativa (m)
Tm = 9.0  # periodo medio Tm01 (s)
T_total = 480.0  # duración total de la simulación (s) — ~10 min como el dataset
dt = 0.48  # intervalo temporal (s), mismo que config.DELTA_T

# ============================================================
# 2. GENERAR ESPECTRO NUMÉRICO y SERIE TEMPORAL vía IFFT
# ============================================================
N = int(T_total / dt)  # número de puntos
T_total_actual = N * dt  # ajustar T_total al múltiplo exacto

# Frecuencias para la IFFT (positive + negative)
freqs_full = np.fft.fftfreq(N, d=dt)  # incluye negativas
omega_full = 2.0 * np.pi * np.abs(freqs_full)

# Espectro JONSWAP en frecuencias angulares positivas
S_full = jonswap_8_12(omega_full, Hs, Tm)
# En f=0 el espectro es 0 (evitar división por cero en omega=0 de la fórmula)
S_full[omega_full == 0] = 0.0

# Amplitudes: A(w) = sqrt(2 * S(w) * df)
# S(w) es densidad espectral en m^2/(rad/s), pasamos a por bin de frecuencia (Hz)
df = 1.0 / T_total_actual  # resolución en Hz
dw = 2.0 * np.pi * df  # resolución en rad/s
A_full = np.sqrt(2.0 * S_full * dw)

# Fases aleatorias (una para cada frecuencia positiva;
# las negativas son conjugadas para señal real)
np.random.seed(42)
n_pos = N // 2
phases = np.random.uniform(0, 2 * np.pi, n_pos)

# Construir espectro complejo Hermitian para que ifft dé señal real.
# Relación: X[k] = (A_k * N / 2) * exp(i*φ) para k=1..N/2-1
#           X[0] = A_0 * N                     (DC, sin factor 1/2)
#           X[N/2] = A_{N/2} * N               (Nyquist, sin factor 1/2)
# Así, np.fft.ifft(spectrum_complex) devuelve directamente z(t) en metros.
spectrum_complex = np.zeros(N, dtype=complex)

# Frecuencias positivas (índices 1..n_pos-1): factor N/2
for k in range(1, n_pos):
    spectrum_complex[k] = (A_full[k] * N / 2.0) * np.exp(1j * phases[k])

# DC (k=0): factor N, sin 1/2
spectrum_complex[0] = A_full[0] * N if N > 0 else 0.0

# Frecuencia de Nyquist (si N es par): factor N, sin 1/2
if N % 2 == 0:
    spectrum_complex[N // 2] = A_full[N // 2] * N

# Conjugadas para frecuencias negativas
spectrum_complex[n_pos + 1 :] = np.conj(spectrum_complex[1:n_pos][::-1])

# IFFT: ya está correctamente escalada, no multiplicar por N
tiempo = np.arange(N) * dt
z_t = np.real(np.fft.ifft(spectrum_complex))  # serie temporal sintética

print(f"N = {N} puntos, T_total = {T_total_actual:.1f} s, dt = {dt} s")
print(f"df = {df:.4f} Hz, dw = {dw:.4f} rad/s")
print(f"z(t) stats: min={z_t.min():.3f}, max={z_t.max():.3f}, std={z_t.std():.3f} m")
print(f"Hs estimado de z(t) = 4*std = {4*z_t.std():.2f} m  (esperado ~{Hs} m)")

# ============================================================
# 3. FFT 1D SOBRE LA SERIE TEMPORAL
# ============================================================
fft_vals = np.fft.fft(z_t)
freqs_fft = np.fft.fftfreq(N, d=dt)

# Solo frecuencias positivas
idx_pos = np.where(freqs_fft > 0)[0]
freqs_pos = freqs_fft[idx_pos]
omega_pos = 2.0 * np.pi * freqs_pos

# Amplitud unilateral: 2/N * |FFT|
amps = (2.0 / N) * np.abs(fft_vals[idx_pos])

# Densidad espectral numérica: S_num = amps^2 / (2 * dw)
S_numerical = (amps**2) / (2.0 * dw)

# Filtrar hasta 1 Hz para visualización limpia
filtro_hz = 1.0
idx_plot = np.where(freqs_pos <= filtro_hz)[0]

# ============================================================
# 4. PLOT: ANALÍTICO (línea) + NUMÉRICO (puntos sin línea)
# ============================================================
omega_teo = np.linspace(0.01, 2 * np.pi * filtro_hz, 1000)
S_teo = jonswap_8_12(omega_teo, Hs, Tm)

fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# ── Panel superior: S(w) vs w ──
ax = axes[0]
ax.plot(
    omega_teo, S_teo, color="black", linewidth=2.5, label="JONSWAP analítico (ec. 8-12)"
)
ax.scatter(
    omega_pos[idx_plot],
    S_numerical[idx_plot],
    color="crimson",
    s=18,
    alpha=0.7,
    zorder=5,
    label="FFT numérico (recuperado)",
)
ax.set_xlabel("Frecuencia angular ω (rad/s)", fontsize=11)
ax.set_ylabel("Densidad espectral S(ω) (m²·s/rad)", fontsize=11)
ax.set_title(
    f"Espectro JONSWAP analítico vs FFT 1D recuperado\nHs={Hs} m, Tm={Tm} s, N={N}, dt={dt} s",
    fontsize=13,
)
ax.legend(fontsize=10)
ax.grid(True, linestyle=":", alpha=0.5)
ax.set_xlim(0, 2 * np.pi * filtro_hz)

# ── Panel inferior: serie temporal sintética ──
ax2 = axes[1]
ax2.plot(tiempo, z_t, color="royalblue", linewidth=0.6)
ax2.set_xlabel("Tiempo (s)", fontsize=11)
ax2.set_ylabel("Elevación z(t) (m)", fontsize=11)
ax2.set_title(
    f"Serie temporal sintética generada por IFFT\nmin={z_t.min():.2f}, max={z_t.max():.2f}, 4·σ={4*z_t.std():.2f} m",
    fontsize=13,
)
ax2.grid(True, linestyle=":", alpha=0.5)
ax2.set_xlim(0, min(120, T_total_actual))  # primeros 120s para ver detalle

plt.tight_layout()
plt.savefig("debug_spectrum.png", dpi=200, bbox_inches="tight")
plt.show()
print("\n✅ Figura guardada como debug_spectrum.png")
