"""
Herramienta de depuración para probar las fórmulas JONSWAP sobre alturanodos.csv.
Se puede ejecutar directamente desde VS Code para revisar los parámetros estimados.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from modules import lectura_gt
from modules.energia_gt_alturanodos import jonswap_8_12, jonswap_8_11


def estimar_parametros_kp(z, delta_t, filtro_hz):
    z = np.asarray(z, dtype=float)
    z = z - np.mean(z)
    n = len(z)

    fft_vals = np.fft.fft(z)
    freqs = np.fft.fftfreq(n, d=delta_t)
    idx_pos = np.where(freqs > 0)[0]
    freqs_pos = freqs[idx_pos]
    amps = (2.0 / n) * np.abs(fft_vals[idx_pos])

    idx_filt = np.where(freqs_pos <= filtro_hz)[0]
    freqs_fil = freqs_pos[idx_filt]
    amps_fil = amps[idx_filt]

    omega_fil = 2.0 * np.pi * freqs_fil
    E_fil = 0.5 * amps_fil**2
    dw_fil = np.diff(omega_fil)
    dw_fil = np.append(dw_fil, dw_fil[-1])
    S_kp = E_fil / dw_fil

    suma_A2 = np.sum(amps_fil**2)
    hs = np.sqrt(8.0 * suma_A2)
    m0 = np.trapezoid(S_kp, omega_fil)
    m1 = np.trapezoid(omega_fil * S_kp, omega_fil)
    m2 = np.trapezoid((omega_fil**2) * S_kp, omega_fil)

    tm01 = 2 * np.pi * m0 / m1 if m1 > 0 else 0.0
    tm02 = np.sqrt(m0 / m2) if m2 > 0 else 0.0

    idx_pico = np.argmax(amps_fil)
    f_pico = freqs_fil[idx_pico]
    tp = 1.0 / f_pico if f_pico > 0 else 0.0

    return hs, tp, tm01, tm02, freqs_fil, omega_fil, S_kp


def generar_debug_jonswap(output_dir=None, mostrar=False):
    os.makedirs(output_dir or config.DIR_SALIDA_ENERGIA_GT, exist_ok=True)

    print("Cargando alturanodos.csv para depuración JONSWAP...")
    z_all, _ = lectura_gt.cargar_z_gt()
    z_all = z_all.astype(float)

    resultados = []
    etiquetas = lectura_gt.etiquetas_gt()

    for kp_id in range(z_all.shape[1]):
        hs, tp, tm01, tm02, freqs_fil, omega_fil, S_kp = estimar_parametros_kp(
            z_all[:, kp_id], config.DELTA_T, config.FILTRO_FREC_MAX_HZ
        )
        resultados.append((etiquetas[kp_id], hs, tp, tm01, tm02))

        omega_teo = np.linspace(omega_fil.min(), omega_fil.max(), 600)
        s_8_12 = jonswap_8_12(omega_teo, hs, tm01)
        s_8_11 = jonswap_8_11(omega_teo, hs, tp)

        plt.figure(figsize=(8, 4))
        plt.plot(omega_teo, s_8_12, color='black', linewidth=2, label='JONSWAP 8-12')
        plt.plot(omega_teo, s_8_11, color='gray', linewidth=1.5, linestyle='--', label='JONSWAP 8-11')
        plt.plot(omega_fil, S_kp, color='teal', linewidth=1.5, label=f'FFT {etiquetas[kp_id]}')
        plt.xlim(0, 3.5)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.title(f'Debug JONSWAP - {etiquetas[kp_id]}')
        plt.xlabel('Frecuencia angular, w (rad/s)')
        plt.ylabel('S(w) (m^2 s/rad)')
        plt.tight_layout()
        out_file = os.path.join(output_dir or config.DIR_SALIDA_ENERGIA_GT, f'debug_jonswap_{kp_id:02d}.png')
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()

    df = pd.DataFrame(resultados, columns=['keypoint', 'Hs_m', 'Tp_s', 'Tm01_s', 'Tm02_s'])
    df.to_csv(os.path.join(output_dir or config.DIR_SALIDA_ENERGIA_GT, 'debug_jonswap_resumen.csv'), index=False, sep=';')

    print(f"Resumen guardado en: {output_dir or config.DIR_SALIDA_ENERGIA_GT}")
    return df


if __name__ == '__main__':
    generar_debug_jonswap()
