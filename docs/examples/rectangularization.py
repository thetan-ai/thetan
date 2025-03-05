import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt

# Generate a noisy sine wave
fs = 1000  # Sampling frequency
T = 1.0    # Duration (seconds)
t = np.linspace(0, T, fs, endpoint=False)
freq = 5   # Frequency of sine wave
x = np.sin(2 * np.pi * freq * t) + 0.3 * np.random.randn(fs)

# 1. High-pass Filtering
sos = signal.butter(4, 2*freq/fs, btype='highpass', output='sos')
hp_filtered = signal.sosfilt(sos, x)

# 2. Clipping & Thresholding
clipped = np.sign(x)

# 3. Fourier Series Approximation
n_harmonics = 9  # Odd harmonics to retain
y_fourier = np.zeros_like(x)
for k in range(1, n_harmonics, 2):
    y_fourier += (4 / (np.pi * k)) * np.sin(2 * np.pi * k * freq * t)

# 4. Schmitt Trigger (Hysteresis)
def schmitt_trigger(signal, low=-0.2, high=0.2):
    state = -1 if signal[0] < low else 1
    out = np.zeros_like(signal)
    for i in range(len(signal)):
        if signal[i] > high:
            state = 1
        elif signal[i] < low:
            state = -1
        out[i] = state
    return out

hysteresis = schmitt_trigger(x)

# 5. Wavelet Denoising
coeffs = pywt.wavedec(x, 'db4', level=3)
threshold = np.std(coeffs[-1])  # Adaptive threshold
coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
wav_denoised = pywt.waverec(coeffs, 'db4')[:len(x)]

# 6. Pulse Shaping (Rectangular Windowing)
pulse_shaped = signal.convolve(clipped, np.ones(10)/10, mode='same')

# Plot results
plt.figure(figsize=(12, 10))
plt.subplot(7, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.title('Original Noisy Sine Wave')
plt.subplot(7, 1, 2)
plt.plot(t, hp_filtered, label='High-pass Filtered')
plt.title('High-pass Filtering')
plt.subplot(7, 1, 3)
plt.plot(t, clipped, label='Clipping & Thresholding')
plt.title('Clipping & Thresholding')
plt.subplot(7, 1, 4)
plt.plot(t, y_fourier, label='Fourier Approximation')
plt.title('Fourier Series Approximation')
plt.subplot(7, 1, 5)
plt.plot(t, hysteresis, label='Schmitt Trigger (Hysteresis)')
plt.title('Schmitt Trigger')
plt.subplot(7, 1, 6)
plt.plot(t, wav_denoised, label='Wavelet Denoising')
plt.title('Wavelet Denoising')
plt.subplot(7, 1, 7)
plt.plot(t, pulse_shaped, label='Pulse Shaping')
plt.title('Pulse Shaping with Rectangular Window')
plt.tight_layout()
plt.show()
