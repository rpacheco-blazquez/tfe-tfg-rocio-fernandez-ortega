# 02 — Generación de olas sintéticas 2D (JONSWAP direccional)

**Objetivo:** Generar series temporales en 24 puntos de una malla 100 m × 100 m
usando un espectro JONSWAP direccional, para validar que la dimensionalidad
(headings) no afecta al espectro 1D de frecuencias medido en un punto.

## Parámetros de la simulación

| Parámetro | Valor |
|-----------|-------|
| $H_s$ | 6 m |
| $T_m$ (fórmula) | 9 s → $T_{m01} \approx 9$ s |
| $\gamma$ | 3.3 |
| Heading principal | 0° (olas hacia +x) |
| Spreading | $\cos^4(\theta/2)$, ±45° |
| Agua | profunda ($\omega^2 = gk$) |
| Dominio espacial | 100 m × 100 m |
|$\Delta x$ | 0.1m |
| Puntos de muestreo | 24 (grid 3×8): $x \in \{0, 14.3, 28.6, 42.9, 57.1, 71.4, 85.7, 100\}$ m, $y \in \{0, 50, 100\}$ m |
| Frecuencias | 10 o 499 (configurable) |
| Direcciones | 144 (cada 2.5°) |
| $dt$ | 0.48 s, |
|$N$ | 1000 frames|
| $T$ |  480 s |

---

## Pipeline

### Paso único — Generar los CSVs sintéticos

```bash
python generate_synthetic_2d_waves.py
```

Para cada componente $(\omega_i, \theta_j)$:
1. Amplitud: $a_{ij} = \sqrt{2 \cdot S(\omega_i) \cdot D(\theta_j) \cdot \Delta\omega \cdot \Delta\theta}$
2. Fase aleatoria $\phi_{ij} \sim U(0, 2\pi)$
3. Elevación: $\eta(x,y,t) = \sum_{i,j} a_{ij} \cos(k_i(x\cos\theta_j + y\sin\theta_j) - \omega_i t + \phi_{ij})$

Guarda `synthetic_z_metros.csv` con formato idéntico a `z_metros.csv`
(frame; kp_00; ...; kp_23). Editar `N_FREQS_TARGET` dentro del script
para cambiar entre 10 y 499 frecuencias.

---

## Lo que se observó

- **Con 10 frecuencias**, la integral discreta **sobreestima** $H_s$ (6.61 m en vez de 6.0 m) porque los bins anchos ($\Delta f = 0.025$ Hz) no capturan bien el pico espectral
- **Con 499 frecuencias**, la integral es casi exacta (6.00 m)
- El $H_s$ realizado ($4\sigma$) es menor que el esperado debido a la duración finita (480 s) y las fases aleatorias
- La dimensionalidad del spreading **no afecta** al espectro 1D: un punto
  fijo mide $\int E(\omega,\theta)d\theta = S(\omega)$.

## Archivos generados

| Archivo | Descripción |
|---------|-------------|
| `generate_synthetic_2d_waves.py` | Generador principal |
| `synthetic_z_metros.csv` | Salida (formato idéntico a `z_metros.csv`) |
| `synthetic_z_metros_10freq.csv` | 10 frecuencias (≈ discretización SeaFEM) |
| `synthetic_z_metros_499freq.csv` | 499 frecuencias (todas las de la FFT) |
