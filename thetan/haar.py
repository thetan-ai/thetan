import numpy as np
import pywt
import matplotlib.pyplot as plt

# Create a noisy piecewise-constant signal
n = 512
x = np.zeros(n)
x[50:150] = 1
x[200:300] = 2
x[350:450] = 0.5
x += 0.1 * np.random.randn(n)

# create a sine wave signal
n = 512
x = np.sin(2 * np.pi * 5 * np.arange(n) / n) + 0.1 * np.random.randn(n)

# Haar wavelet decomposition
coeffs = pywt.wavedec(x, 'haar', level=5)

# Threshold detail coefficients (keep block structure)
threshold = 0.999
coeffs_thresh = [coeffs[0]] + [
    pywt.threshold(c, threshold, mode='hard') for c in coeffs[1:]
]

# Reconstruct rectangularized signal
x_rect = pywt.waverec(coeffs_thresh, 'haar')

# Plot
plt.figure(figsize=(10,4))
plt.plot(x, label='Original', alpha=0.5)
plt.plot(x_rect, label='Rectangularized (Haar)', linewidth=2)
plt.legend()
plt.title("Haar-based Rectangularization")
plt.show()
