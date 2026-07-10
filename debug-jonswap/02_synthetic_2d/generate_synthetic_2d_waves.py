"""
generate_synthetic_2d_waves.py
Genera series temporales sintéticas en 24 puntos de una malla de
100 m × 100 m usando un espectro JONSWAP direccional (2D).

Parámetros:
  - Hs = 6 m, Tm01 = 9 s (deep water)
  - Heading principal: 0º (olas viajando hacia +x)
  - Spreading direccional: ±45º (cos²ˢ, s=2)
  - Frecuencias: T ∈ [4, 12] s
  - dt = 0.48 s, N_frames = 1000, T_total = 480 s
  - 24 puntos en grid 3×8 sobre 100m × 100m

Genera:
  - synthetic_z_metros.csv  (formato idéntico a z_metros.csv)
"""

import os
import numpy as np
import pandas as pd

# ============================================================
# CONSTANTES FÍSICAS Y DE SIMULACIÓN
# ============================================================
G = 9.81  # gravedad (m/s²)

# Parámetros del espectro
HS = 6.0  # altura significativa (m)
TM01 = 9.0  # periodo medio Tm01 (s)
GAMMA = 3.3  # peak enhancement factor

# Parámetros direccionales
THETA0 = 0.0  # heading principal (rad) — 0 = hacia +x
SPREAD_DEG = 90.0  # ancho total del spreading en grados (±SPREAD_DEG/2)
SPREADING_S = 3  # exponente de cos²ˢ (3 = cae suave, toca ~0 en los bordes)

# Dominio temporal
DT = 0.48  # intervalo entre frames (s)
N_FRAMES = 1000  # número de frames
T_TOTAL = N_FRAMES * DT  # 480 s

# Dominio espacial
LX = 100.0  # tamaño en x (m)
LY = 100.0  # tamaño en y (m)
GRID_ROWS = 3  # filas de puntos
GRID_COLS = 8  # columnas de puntos

# Rango de frecuencias (periodos de 4s a 12s)
T_MIN = 4.0
T_MAX = 12.0
F_MIN = 1.0 / T_MAX  # Hz
F_MAX = 1.0 / T_MIN  # Hz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

print("=" * 60)
print("GENERADOR DE OLAS SINTÉTICAS 2D (JONSWAP direccional)")
print("=" * 60)
print(f"  Hs={HS} m, Tm01={TM01} s, gamma={GAMMA}")
print(
    f"  Heading: {np.degrees(THETA0):.0f} deg, spreading: cos^{2*SPREADING_S}, width={SPREAD_DEG} deg"
)
print(f"  Periodos: {T_MIN}-{T_MAX} s")
print(f"  Grid: {GRID_ROWS}×{GRID_COLS} puntos sobre {LX}×{LY} m")
print(f"  Tiempo: {N_FRAMES} frames, dt={DT} s, T_total={T_TOTAL} s")


# ============================================================
# 1. ESPECTRO JONSWAP 1D — Ec. 8-12 SeaFEM
# ============================================================
def jonswap_8_12(omega, Hs, Tm):
    """
    S(ω) = (155·Hs²/(Tm⁴·ω⁵)) · 3.3^Y · exp(-944·Tm⁻⁴·ω⁻⁴)
    """
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


# ============================================================
# 2. SPREADING DIRECCIONAL — soporte compacto en ±SPREAD_DEG/2
#    D(θ) = cos²ˢ(π/2 · t) para |t|≤1, 0 fuera
#    con t = (θ-θ₀)/(Δθ/2),  Δθ = SPREAD_DEG en radianes
# ============================================================
def spreading_cos2s(theta, theta0, s, spread_deg):
    """
    D(θ) con soporte compacto en [θ₀ - spread/2, θ₀ + spread/2].
    Normalizado: ∫_{-π}^{π} D(θ) dθ = 1
    """
    dtheta = theta - theta0
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # wrap a [-π, π]
    half_width = np.radians(spread_deg / 2.0)  # Δθ/2
    t = dtheta / half_width  # t ∈ [-π/hw, π/hw]
    D_raw = np.where(np.abs(t) <= 1.0, np.cos(np.pi / 2.0 * t) ** (2 * s), 0.0)
    # Normalizar para que Σ D_j · Δθ = 1
    dtheta_step = theta[1] - theta[0]
    Z = np.sum(D_raw) * dtheta_step
    return D_raw / Z if Z > 0 else D_raw


# ============================================================
# 3. Tm DE LA FÓRMULA JONSWAP (SeaFEM ec. 8-12)
#    Para γ=3.3: Tp ≈ 1.09·Tm01, y ω_p = 5.24/Tm_formula
#    → Tm_formula ≈ 5.24·Tp/(2π) ≈ 5.24·1.09·Tm01/(2π) ≈ 0.91·Tm01
#    Usamos Tm_formula ≈ 9.0 para Tm01 ≈ 9.0 (aproximación directa)
# ============================================================
TM_FORMULA = 9.0  # valor directo: el pico ω_p = 5.24/9 = 0.582 rad/s → Tp ≈ 10.8s

# Verificación rápida
omega_cal = np.linspace(0.01, 6.0, 20000)
dom = omega_cal[1] - omega_cal[0]
S_cal = jonswap_8_12(omega_cal, HS, TM_FORMULA)
m0_cal = np.sum(S_cal) * dom
m1_cal = np.sum(S_cal * omega_cal) * dom
Hs_cal = 4.0 * np.sqrt(m0_cal)
Tm01_cal = m0_cal / m1_cal
omega_p_cal = omega_cal[np.argmax(S_cal)]
Tp_cal = 2.0 * np.pi / omega_p_cal
print(f"\n--- Verificación JONSWAP con Tm_formula={TM_FORMULA} ---")
print(f"  Hs={Hs_cal:.3f} m, Tm01={Tm01_cal:.3f} s, Tp ~ {Tp_cal:.2f} s")

# ── Nº de frecuencias a usar (10 para replicar discretización SeaFEM) ──
N_FREQS_TARGET = 10  # ← CAMBIA AQUÍ: 10, 50, 100, 499 (todas)
# ============================================================
# 4. DISCRETIZAR FRECUENCIAS Y DIRECCIONES
# ============================================================
freqs_fft = np.fft.fftfreq(N_FRAMES, d=DT)
idx_pos_fft = np.where(freqs_fft > 0)[0]
all_freqs_pos = freqs_fft[idx_pos_fft]

if N_FREQS_TARGET < len(all_freqs_pos):
    # Seleccionar N_FREQS_TARGET frecuencias cubriendo el rango del espectro
    # Desde T=20s (0.05 Hz) hasta T=4s (0.25 Hz), espaciado uniforme en Hz
    f_min_target = 0.04  # Hz  (T=25s)
    f_max_target = 0.30  # Hz  (T≈3.3s)
    idx_range = np.where(
        (all_freqs_pos >= f_min_target) & (all_freqs_pos <= f_max_target)
    )[0]
    step = max(1, len(idx_range) // N_FREQS_TARGET)
    idx_selected = idx_range[::step][:N_FREQS_TARGET]
    freqs_gen = all_freqs_pos[idx_selected]
else:
    freqs_gen = all_freqs_pos

omega_gen = 2.0 * np.pi * freqs_gen
N_FREQS = len(freqs_gen)

# Direcciones: discretizar [-π, π] con resolución fina
N_DIRS = 144  # cada 2.5°
theta_edges = np.linspace(-np.pi, np.pi, N_DIRS + 1)
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
dtheta = theta_centers[1] - theta_centers[0]

print(f"\n--- Discretización (N_FREQS_TARGET={N_FREQS_TARGET}) ---")
print(f"  Frecuencias: {N_FREQS} componentes")
for i, (f, w) in enumerate(zip(freqs_gen, omega_gen)):
    print(f"    [{i:2d}] f={f:.4f} Hz  w={w:.4f} rad/s  T={1/f:.1f}s")
print(
    f"    df ~ {freqs_gen[1]-freqs_gen[0]:.4f} Hz  dw ~ {omega_gen[1]-omega_gen[0]:.4f} rad/s"
)
print(f"  Direcciones: {N_DIRS} direcciones (dtheta = {np.degrees(dtheta):.1f} deg)")
print(f"  Total componentes (w,theta): {N_FREQS} x {N_DIRS} = {N_FREQS * N_DIRS}")

# ============================================================
# 5. CONSTRUIR EL ESPECTRO 2D: E(ω,θ) = S(ω) · D(θ)
# ============================================================
# Espectro 1D en cada ω
S_1d = jonswap_8_12(omega_gen, HS, TM_FORMULA)

# Spreading direccional (mismo para todas las frecuencias)
D_theta = spreading_cos2s(theta_centers, THETA0, SPREADING_S, SPREAD_DEG)
# Verificar normalización: ∫D(θ)dθ debe ser ≈ 1
print(f"  Sum D_j*dtheta = {np.sum(D_theta) * dtheta:.6f} (must be ~1)")

# Espectro 2D: E(ω,θ) = S(ω) · D(θ)
# Amplitud por componente: A_ij = √(2 · E(ω_i,θ_j) · Δω · Δθ)
df_gen = freqs_gen[1] - freqs_gen[0] if N_FREQS > 1 else 1.0 / T_TOTAL
dw_gen = 2.0 * np.pi * df_gen

# Meshgrid para broadcasting
S_2d = np.outer(S_1d, D_theta)  # (N_freqs, N_dirs)
E_2d = S_2d  # ya está en unidades correctas

# Amplitud de cada componente
A_ij = np.sqrt(2.0 * E_2d * dw_gen * dtheta)  # (N_freqs, N_dirs)

# Verificar energía total: Σ a_ij²/2 debe ser ≈ m0 = (Hs/4)²
m0_discrete = np.sum(A_ij**2) / 2.0
Hs_discrete = 4.0 * np.sqrt(m0_discrete)
print(f"  Hs desde espectro discretizado: {Hs_discrete:.3f} m (esperado: {HS:.1f} m)")
print(f"  m0_discrete = {m0_discrete:.4f} (esperado: {(HS/4)**2:.4f})")

# ── DEBUG: gráfico del espectro direccional E(ω,θ) ──
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel izquierdo: spreading D(θ)
ax1.plot(np.degrees(theta_centers), D_theta, "b-", linewidth=2)
ax1.fill_between(np.degrees(theta_centers), 0, D_theta, alpha=0.2, color="b")
ax1.set_xlabel("Dirección θ (°)")
ax1.set_ylabel("D(θ) (rad⁻¹)")
ax1.set_title(
    f"Spreading  cos^{2*SPREADING_S}(pi/2 * t)  |  width={SPREAD_DEG} deg  |  Sum D*dtheta = {np.sum(D_theta)*dtheta:.3f}"
)
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.axvline(x=-45, color="gray", linestyle="--", alpha=0.5)
ax1.axvline(x=45, color="gray", linestyle="--", alpha=0.5)

# Panel derecho: espectro 2D E(ω,θ) en dB
E_dB = 10 * np.log10(np.maximum(E_2d, 1e-10))
im = ax2.pcolormesh(
    freqs_gen, np.degrees(theta_centers), E_dB.T, shading="auto", cmap="viridis"
)
ax2.set_xlabel("Frecuencia f (Hz)")
ax2.set_ylabel("Dirección θ (°)")
ax2.set_title(f"Espectro direccional E(ω,θ) [dB]  (N_freqs={N_FREQS}, N_dirs={N_DIRS})")
plt.colorbar(im, ax=ax2, label="dB")
fig.tight_layout()
ruta_debug = os.path.join(SCRIPT_DIR, "debug_directional_spectrum.png")
fig.savefig(ruta_debug, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> {ruta_debug}")
# ────────────────────────────────────────────────────────────────

# ============================================================
# 6. NÚMERO DE ONDA (deep water: ω² = g·k)
# ============================================================
k_i = omega_gen**2 / G  # (N_freqs,)
print(f"  Wavenumbers: k_min={k_i[0]:.4f}, k_max={k_i[-1]:.4f} rad/m")
print(f"  Wavelengths: L_min={2*np.pi/k_i[-1]:.1f}, L_max={2*np.pi/k_i[0]:.1f} m")

# ============================================================
# 7. POSICIONES DE LOS 24 PUNTOS (grid 3×8 sobre 100m×100m)
# ============================================================
x_pos = np.linspace(0, LX, GRID_COLS)
y_pos = np.linspace(0, LY, GRID_ROWS)
XX, YY = np.meshgrid(x_pos, y_pos)
x_pts = XX.flatten()  # 24 puntos
y_pts = YY.flatten()

print(f"\n--- Posiciones de los 24 puntos ---")
for i in range(24):
    print(f"  KP_{i:02d}: x={x_pts[i]:5.1f} m, y={y_pts[i]:5.1f} m")

# ============================================================
# 8. GENERAR FASES ALEATORIAS Y PRECOMPUTAR FASES ESPACIALES
# ============================================================
np.random.seed(SEED)
phi_ij = np.random.uniform(0, 2.0 * np.pi, (N_FREQS, N_DIRS))

# Precomputar fase espacial: ψ(x,y) = k·(x cosθ + y sinθ)
# Para cada punto: ψ_ij(p) = k_i * (x_p cosθ_j + y_p sinθ_j)
cos_theta = np.cos(theta_centers)  # (N_dirs,)
sin_theta = np.sin(theta_centers)

print(f"\n--- Generando series temporales en {24} puntos... ---")

# Vector de tiempo
t = np.arange(N_FRAMES) * DT  # (N_frames,)

# Precomputar ω_i · t: (N_freqs, N_frames)
omega_t = np.outer(omega_gen, t)  # (N_freqs, N_frames)
cos_omega_t = np.cos(omega_t)  # (N_freqs, N_frames)
sin_omega_t = np.sin(omega_t)  # (N_freqs, N_frames)

# Para cada punto, generar η(t)
z_all = np.zeros((N_FRAMES, 24))

for p in range(24):
    xp, yp = x_pts[p], y_pts[p]

    # Fase espacial: ψ_ij = k_i·(xp·cosθ_j + yp·sinθ_j)
    spatial_term = xp * cos_theta + yp * sin_theta  # (N_dirs,)
    psi_ij = np.outer(k_i, spatial_term) + phi_ij  # (N_freqs, N_dirs)

    # Expandir usando identidad trigonométrica:
    #   η(t) = Σ_ij A_ij · cos(ψ_ij - ω_i·t)
    #        = Σ_i [C_i·cos(ω_i·t) + S_i·sin(ω_i·t)]
    #   donde C_i = Σ_j A_ij·cos(ψ_ij), S_i = Σ_j A_ij·sin(ψ_ij)
    C_i = np.sum(A_ij * np.cos(psi_ij), axis=1)  # (N_freqs,)
    S_i = np.sum(A_ij * np.sin(psi_ij), axis=1)  # (N_freqs,)

    # η(t) para todos los frames: (N_freqs, N_frames) -> (N_frames,)
    eta_t = np.sum(
        C_i[:, np.newaxis] * cos_omega_t + S_i[:, np.newaxis] * sin_omega_t, axis=0
    )

    z_all[:, p] = eta_t

    if (p + 1) % 6 == 0:
        print(f"  Punto {p+1}/{24} completado...")

print("  ¡Generación completada!")

# ============================================================
# 9. VERIFICACIÓN RÁPIDA
# ============================================================
print(f"\n--- Verificación ---")
print(f"  z min: {z_all.min():.3f} m")
print(f"  z max: {z_all.max():.3f} m")
print(f"  z std (promedio KPs): {np.mean(np.std(z_all, axis=0)):.3f} m")
print(
    f"  4*sigma ~ Hs: {4.0 * np.mean(np.std(z_all, axis=0)):.2f} m  (expected: {HS:.1f} m)"
)

# ============================================================
# 10. GUARDAR CSV (mismo formato que z_metros.csv)
# ============================================================
df_out = pd.DataFrame()
df_out["frame"] = np.arange(1, N_FRAMES + 1)
for kp in range(24):
    df_out[f"kp_{kp:02d}"] = z_all[:, kp]

ruta_csv = os.path.join(SCRIPT_DIR, "synthetic_z_metros.csv")
df_out.to_csv(ruta_csv, index=False, sep=";")
print(f"\n>>> CSV saved: {ruta_csv}")
print(f"   Dimensiones: {df_out.shape}")
print(f"   Formato: frame; kp_00; kp_01; ...; kp_23")
print(f"\nAhora ejecuta: python csv_fft_plot_prediction.py")
print(f"  (cambiando CSV_Z_METROS a 'synthetic_z_metros.csv' en el script)")
