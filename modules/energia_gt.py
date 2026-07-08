"""
modules/energia_gt.py
Gráficas de validación de energía espectral (E vs A) y densidad espectral
S(w) vs w del groundtruth, calculadas desde Spectrum.out.dat.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

g = 9.81


def jonswap_seafem(omega, Hs, Tm):
    """
    Espectro JONSWAP según SeaFEM Theory Manual.
    Parámetros:
        omega : frecuencia angular (rad/s)
        Hs    : altura significativa (m)
        Tm    : periodo medio (s)
    """

    sigma = np.where(
        omega <= 16.1 / Tm,
        0.07,
        0.09
    )

    Y = np.exp(
        -((0.191 * omega * Tm - 1.0) ** 2) /
        (2.0 * sigma ** 2)
    )

    S = (
        (155.0 * Hs ** 2)
        / (Tm ** 4 * omega ** 5)
        * np.exp(-944.0 / (Tm ** 4 * omega ** 4))
        * (3.3 ** Y)
    )

    return S


def generar_graficas_energia_gt():
    """Genera las gráficas globales y por dirección de S(w) vs w y E vs A del GT."""
    os.makedirs(config.DIR_SALIDA_ENERGIA_GT, exist_ok=True)

    print("Cargando Spectrum.out.dat...")
    df = pd.read_csv(config.RUTA_SPECTRUM, sep=r'\s+', skiprows=2)
    df.columns = ["TWaves", "AWaves", "GWaves", "PWaves"]

    T, A, G = df["TWaves"].values, df["AWaves"].values, df["GWaves"].values
    omega = 2.0 * np.pi / T
    E = 0.5 * (A ** 2)

    print(f"Componentes leídas: {len(T)}")
    print(f"Periodos:    {T.min():.2f}s — {T.max():.2f}s")
    print(f"Amplitudes:  {A.min():.4f}m — {A.max():.4f}m")
    print(f"Direcciones: {np.degrees(G.min()):.1f}° — {np.degrees(G.max()):.1f}°")

    # ── Gráfica global S(w) vs w ──
    orden = np.argsort(omega)
    w_ord, E_ord = omega[orden], E[orden]
    dw = np.diff(w_ord)
    dw = np.append(dw, dw[-1])
    S_omega = E_ord / dw
    Hs = 6.0
    Tm = 9.0

    # Índice del pico espectral (antes se usaba sin definir -> NameError)
    idx_pico = np.argmax(S_omega)

    w_teo = np.linspace(
        w_ord.min(),
        w_ord.max(),
        600
    )

    S_teo = jonswap_seafem(w_teo, Hs, Tm)

    plt.figure(figsize=(10, 5))

    # Curva analítica
    plt.plot(
        w_teo,
        S_teo,
        color='black',
        linewidth=2.5,
        label='JONSWAP analítico'
    )

    # Curva numérica
    plt.plot(
        w_ord,
        S_omega,
        color='limegreen',
        linewidth=2,
        label='SeaFEM numérico'
    )

    # Componentes discretas
    plt.stem(
        w_ord,
        S_omega,
        linefmt='crimson',
        markerfmt='rx',
        basefmt=' ',
        label='Componentes discretas'
    )

    plt.axvline(x=w_ord[idx_pico], color='black', linestyle='--', alpha=0.6,
                label=f'$\\omega_p$ = {w_ord[idx_pico]:.3f} rad/s  →  Tp = {2*np.pi/w_ord[idx_pico]:.2f} s')
    plt.title('Espectro de Densidad Energética GT (SeaFEM)\nTodas las componentes')
    plt.xlabel(r'Frecuencia angular, $\omega$ (rad/s)')
    plt.ylabel(r'Densidad Espectral, $S(\omega)$ ($m^2 \cdot s / rad$)')
    plt.xlim(0, 3.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_SALIDA_ENERGIA_GT, "GT_S_vs_w_global.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ GT_S_vs_w_global.png guardada")

    # ── Gráfica global E vs A ──
    plt.figure(figsize=(7.5, 4.5))
    plt.scatter(A, E, color='royalblue', alpha=0.7, edgecolors='black', label='Componentes del oleaje GT')
    a_teorica = np.linspace(0, A.max(), 100)
    plt.plot(a_teorica, 0.5 * a_teorica ** 2, color='darkorange', linestyle='--', linewidth=1.5,
              label=r'$E = \frac{1}{2}A^2$ (Manual SeaFEM)')
    plt.title('Validación de Energía Espectral GT (SeaFEM)\nTodas las componentes')
    plt.xlabel('Amplitud armónica de la ola, $A$ (m)')
    plt.ylabel('Energía del componente, $E$ ($m^2$)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(config.DIR_SALIDA_ENERGIA_GT, "GT_E_vs_A_global.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ GT_E_vs_A_global.png guardada")

    # ── Gráficas por dirección ──
    dirs_unicas = np.unique(G)
    print(f"\nGenerando gráficas por dirección ({len(dirs_unicas)} direcciones)...")

    for d_idx, dir_val in enumerate(dirs_unicas):
        mask = G == dir_val
        T_dir, A_dir, E_dir = T[mask], A[mask], E[mask]

        orden_dir = np.argsort(2 * np.pi / T_dir)
        w_dir = (2 * np.pi / T_dir)[orden_dir]
        A_dir, E_dir = A_dir[orden_dir], E_dir[orden_dir]

        dw_dir = np.diff(w_dir)
        dw_dir = np.append(dw_dir, dw_dir[-1])
        S_dir = E_dir / dw_dir
        dir_deg = np.degrees(dir_val)

        plt.figure(figsize=(7.5, 4.5))
        plt.plot(w_dir, S_dir, color='limegreen', linewidth=2, label=r'Envolvente $S(\omega)$')
        plt.stem(w_dir, S_dir, linefmt='crimson', markerfmt='rx', basefmt=' ', label='Armónicos FFT')
        idx_p = np.argmax(S_dir)
        plt.axvline(x=w_dir[idx_p], color='black', linestyle='--', alpha=0.4,
                    label=f'$\\omega_p$ = {w_dir[idx_p]:.3f} rad/s')
        plt.title(f'Espectro GT — Dirección {dir_deg:.1f}°\n({len(T_dir)} componentes)')
        plt.xlabel(r'Frecuencia angular, $\omega$ (rad/s)')
        plt.ylabel(r'$S(\omega)$ ($m^2 \cdot s / rad$)')
        plt.xlim(0, 3.5)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(config.DIR_SALIDA_ENERGIA_GT,
                    f"01_GT_S_vs_w_dir_{d_idx:02d}_{dir_deg:.1f}deg.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(7.5, 4.5))
        plt.scatter(A_dir, E_dir, color='royalblue', alpha=0.7, edgecolors='black',
                    label=f'Componentes dirección {dir_deg:.1f}°')
        a_teo = np.linspace(0, A_dir.max(), 100)
        plt.plot(a_teo, 0.5 * a_teo ** 2, color='darkorange', linestyle='--', linewidth=1.5,
                  label=r'$E = \frac{1}{2}A^2$')
        plt.title(f'Validación Energía GT — Dirección {dir_deg:.1f}°')
        plt.xlabel('Amplitud armónica, $A$ (m)')
        plt.ylabel('Energía, $E$ ($m^2$)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(config.DIR_SALIDA_ENERGIA_GT,
                    f"02_GT_E_vs_A_dir_{d_idx:02d}_{dir_deg:.1f}deg.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Todo guardado en: {config.DIR_SALIDA_ENERGIA_GT}")
