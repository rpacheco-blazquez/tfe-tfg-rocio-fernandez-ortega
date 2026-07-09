# Corrección del escalado IFFT → FFT en el pipeline

## Problema detectado

Al validar la FFT 1D con un caso sintético sencillo en `debug_spectrum.py` — generar una serie temporal a partir del espectro JONSWAP analítico mediante IFFT y luego recuperar el espectro con FFT — se observó que **el espectro recuperado era ~4× mayor** que el espectro JONSWAP original. El bug estaba en la construcción del espectro complejo dentro de `debug_spectrum.py`, **no** en `fft_analisis.py`.

![debug_spectrum](debug_spectrum.png)

## Causa: error en la construcción del espectro complejo para `np.fft.ifft`

### Convención de numpy para la FFT

Numpy usa la siguiente convención, que es la misma que emplea `fft_analisis.py` al llamar a `np.fft`:

```python
# FFT directa
X[k] = sum_{n=0}^{N-1} x[n] * exp(-i * 2π * k * n / N)

# IFFT inversa
x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+i * 2π * k * n / N)
```

### Relación entre amplitud de una sinusoide y su coeficiente DFT

Para una sinusoide real de amplitud $A$, frecuencia $f_k = k/(N \cdot \Delta t)$ y fase $\phi$:

$$x[n] = A \cdot \cos\left(\frac{2\pi k n}{N} + \phi\right)$$

Su Transformada de Fourier Discreta (DFT) es:

$$X[k] = \frac{A \cdot N}{2} \cdot e^{i\phi}, \qquad X[N-k] = \frac{A \cdot N}{2} \cdot e^{-i\phi}$$

> **El coeficiente DFT escala con $N/2$**, no con la amplitud directamente.

### Lo que estaba mal

#### ❌ Código incorrecto (versión original)

```python
# Se construía el espectro usando la AMPLITUD directamente (sin N/2):
spectrum_complex[k] = A_full[k] * np.exp(1j * phases[k])

# Se compensaba multiplicando por N después de ifft:
z_t = np.fft.ifft(spectrum_complex) * N
```

#### Trazado del error

| Paso | Qué ocurre | Resultado |
|------|-----------|-----------|
| 1. `spectrum_complex[k] = A * exp(iφ)` | Coeficiente DFT = $A$ en vez de $A \cdot N/2$ | ❌ |
| 2. `ifft(spectrum_complex)` | `(1/N) * A * exp(...)` + c.c. | $= \frac{2A}{N}\cos(...)$ |
| 3. `* N` | $\frac{2A}{N} \cdot N$ | $= 2A \cos(...)$ |
| **Amplitud en serie temporal** | | **$2A$** en vez de $A$ ❌ |
| 4. `fft(z_t)` | $X[k] = 2A \cdot N/2$ | $= A \cdot N$ |
| 5. `amps = (2/N) * |X[k]|` | $(2/N) \cdot A \cdot N = 2A$ |
| **Amplitud recuperada** | | **$2A$** en vez de $A$ ❌ |
| 6. `S_num = amps² / (2*dw)` | $(2A)^2 / (2dw) = 4A^2/(2dw)$ | $= 4 \cdot S_{original}$ |
| **Espectro recuperado** | | **×4** ❌ |

### ¿De dónde sale el ×4? Descomposición paso a paso

El factor ×4 **no depende de N**, se cancela en el proceso. Viene de dos errores encadenados:

#### 🔴 Error 1: ×2 en amplitud (IFFT mal escalada)

El código original mete `X[k] = A·e^{iφ}` (sin `N/2`). La IFFT de numpy **divide por N**:

$$\text{ifft}(X)[n] = \frac{1}{N} \cdot A e^{i\phi} e^{i 2\pi k n/N} + \frac{1}{N} \cdot A e^{-i\phi} e^{-i 2\pi k n/N} = \frac{2A}{N}\cos(\dots)$$

Luego `* N`:  $z_t[n] = \mathbf{2A} \cdot \cos(\dots)$  →  **amplitud = 2A en vez de A**.

#### 🔴 Error 2: ×2 en amplitud se convierte en ×4 en densidad espectral

La FFT de vuelta recupera `amps = 2A`. Al pasarlo a densidad espectral:

$$S_{num} = \frac{amps^2}{2 \cdot \Delta\omega} = \frac{(2A)^2}{2\Delta\omega} = \frac{4A^2}{2\Delta\omega}$$

Frente al valor correcto $S_{teo} = \frac{A^2}{2\Delta\omega}$:

$$\frac{S_{num}}{S_{teo}} = \frac{4A^2}{A^2} = \mathbf{4}$$

```
Amplitud deseada:  A
         │
         ▼  X[k] = A·e^iφ  (sin N/2)
         │
     ┌───┴───┐
     │ IFFT  │  → ifft·N → 2A  ← ×2 en amplitud
     └───┬───┘
         │
     ┌───┴───┐
     │  FFT  │  → amps = 2A   ← se mantiene ×2
     └───┬───┘
         │
         ▼  S = (2A)²/(2dw) = 4·A²/(2dw)  ← (×2)² = ×4
```

### Ejemplo numérico

Con $A = 1.0$ m (amplitud deseada) y $N = 1000$:

| Método | `spectrum_complex[k]` | Amplitud en `z_t` | `|fft(z_t)[k]|` | `amps` | $S_{num}$ |
|--------|----------------------|-------------------|-----------------|--------|------------|
| ❌ Original | $1.0$ | $2.0$ m | $1000$ | $2.0$ | $4 \cdot S_{teo}$ |
| ✅ Corregido | $1.0 \cdot 500 = 500$ | $1.0$ m | $500$ | $1.0$ | $S_{teo}$ |

## Corrección aplicada

#### ✅ Código corregido

```python
# Frecuencias positivas (k = 1 .. N/2-1):
# El coeficiente DFT es (A_k * N / 2) * exp(i*φ)
spectrum_complex[k] = (A_full[k] * N / 2.0) * np.exp(1j * phases[k])

# DC (k=0): no hay par conjugado → factor N, sin dividir por 2
spectrum_complex[0] = A_full[0] * N

# Nyquist (k=N/2, si N es par): sin par conjugado → factor N
spectrum_complex[N // 2] = A_full[N // 2] * N

# IFFT ya está correctamente escalada, NO multiplicar por N
z_t = np.real(np.fft.ifft(spectrum_complex))
```

### Regla mnemotécnica

| Componente | Factor para `spectrum_complex[k]` |
|------------|-----------------------------------|
| $k = 0$ (DC) | $A_0 \cdot N$ |
| $k = 1 \dots N/2-1$ (positivas) | $A_k \cdot N/2$ |
| $k = N/2$ (Nyquist, si N par) | $A_{N/2} \cdot N$ |
| $k > N/2$ (negativas) | Conjugada de su par positivo |

Después, `np.fft.ifft()` **sin multiplicar por N** devuelve la señal en el dominio del tiempo.

## Relevancia para `fft_analisis.py`

La función `fft_1d_por_keypoint` en `modules/fft_analisis.py` **no tiene este problema** porque solo hace FFT directa sobre datos ya existentes (de YOLO o de SEAFEM), nunca construye un espectro sintético desde cero vía IFFT. El bug descrito en este documento pertenece exclusivamente a la primera versión de `debug_spectrum.py`.

Sin embargo, si en el futuro se quisiera:
- Generar series sintéticas de validación desde un espectro teórico (como hace `debug_spectrum_numeric_write_csv.py`)
- Hacer filtrado en frecuencia y reconstruir la señal temporal mediante IFFT
- Simular olas sintéticas para tests

...habría que aplicar esta misma corrección de escala (factor $N/2$ en el coeficiente DFT).

## Fórmula de la FFT 1D usada en el pipeline

Para referencia, esto es lo que hace `fft_1d_por_keypoint` (líneas ~85-88 de `fft_analisis.py`):

```python
fft_vals = np.fft.fft(z)                         # DFT de la serie centrada
freqs = np.fft.fftfreq(N, d=config.DELTA_T)      # frecuencias (Hz)
idx_pos = np.where(freqs > 0)[0]                 # solo positivas
amps_IA = (2.0 / N) * np.abs(fft_vals[idx_pos])  # amplitudes unilaterales
```

Y luego para pasar de amplitudes a densidad espectral:
$$S(\omega_k) = \frac{A_k^2}{2 \cdot \Delta\omega}, \quad \Delta\omega = \frac{2\pi}{N \cdot \Delta t}$$

Esta misma conversión se usa en `csv_fft_plot.py` (paso 2) y en `debug_spectrum.py` (paso 3) para comparar contra el JONSWAP analítico.
