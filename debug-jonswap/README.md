# debug-jonswap

Carpeta de validación del pipeline FFT 1D (`fft_analisis.py`).  
Validamos que la FFT recupera correctamente el espectro a partir de series
temporales, tanto sintéticas como del modelo SeaFEM original.

---

## Estructura

```
debug-jonswap/
├── README.md                  ← este archivo
├── z_metros.csv               ← datos originales de SeaFEM (24 KPs, 1000 frames)
├── 01_validation_1d/          → [README](01_validation_1d/README.md)
│   └── Validación básica: JONSWAP 1D → IFFT → FFT
├── 02_synthetic_2d/           → [README](02_synthetic_2d/README.md)
│   └── Generación de olas 2D (JONSWAP direccional, 24 puntos)
└── 03_comparison/             → [README](03_comparison/README.md)
    └── Comparativa: sintético vs original SeaFEM (3 columnas)
```

---

## Resumen de hallazgos

| Carpeta | Conclusión principal |
|---------|---------------------|
| [`01_validation_1d`](01_validation_1d/README.md) | ✅ La FFT 1D funciona. El bug de escalado IFFT→FFT (×4) fue corregido. |
| [`02_synthetic_2d`](02_synthetic_2d/README.md) | ✅ La dimensionalidad (headings) no afecta al espectro 1D. Con 10 frecuencias la integral discreta sobreestima Hs. |
| [`03_comparison`](03_comparison/README.md) | ❌ `z_metros.csv` **no** se generó con Hs=6, Tm01=9. Su Tm01 real ≈ 7.3 s. El método de análisis es correcto. |

## Dependencias

- Python 3.10+
- `numpy`, `pandas`, `matplotlib`
- `config.py` y `modules/fft_analisis.py` (del proyecto padre)
