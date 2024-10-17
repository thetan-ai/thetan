from time import time
import numpy as np
from thetan.berserker import berserker2
import pandas as pd
from patchify import patchify
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap  
from matplotlib.colors import Normalize

# Random sine wave
sine_signal = np.sin(np.linspace(0, 50, 1000))

# Random pulse signals
duration = 10  # total time in seconds
sampling_rate = 1000  # Hz
time_ = np.arange(0, duration, 1/sampling_rate)
pulse_signal = np.zeros_like(time_)
num_pulses = 15
for _ in range(num_pulses):
    pulse_start = np.random.randint(0, len(time_) - 1)
    pulse_width = np.random.randint(50, 200)  # pulse width between 50 ms and 200 ms
    pulse_end = min(pulse_start + pulse_width, len(time_))
    pulse_signal[pulse_start:pulse_end] = 1

# Sine wave + pulse signal
fs = 1000  # Sampling frequency (samples per second)
t_total = np.linspace(0, 10, 10 * fs)  # Total time of 10 seconds with 1000 samples per second

# Create distinct segments with varying patterns
segment1 = np.sin(2 * np.pi * 3 * t_total[0:fs*2])  # Sine wave with 3 Hz frequency (2 seconds)
segment2 = np.sign(np.sin(2 * np.pi * 5 * t_total[fs*4:fs*6]))  # Pulse wave (2 seconds)
segment3 = np.cos(2 * np.pi * 2 * t_total[fs*8:fs*10])  # Cosine wave with 2 Hz frequency (2 seconds)

# Concatenate all segments to form the complex signal
sine_pulse_combo = np.concatenate((segment1, segment2, segment3))

#Plot lines function
def plot_lines_with_color_gradient(x, y, labels, theta_value, scale, window_size, signal_type):
    colormap = get_cmap('Accent')
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize the labels to map to the colormap
    norm = Normalize(vmin=np.min(labels), vmax=np.max(labels))
    
    # Create LineCollection with colors mapped to labels using the colormap
    lc = LineCollection(segments, cmap=colormap, norm=norm)
    lc.set_array(np.array(labels))  # Assign labels for coloring
    lc.set_linewidth(4)

    fig, ax = plt.subplots(figsize=(20, 10))

    # Add the colored line collection to the plot
    ax.add_collection(lc)

    # Set plot parameters
    ax.set_xlim(x.min(), x.min() + (x.max() - x.min()) // 2)
    ax.set_ylim(y.min() - 0.2, y.max() + 0.2)
    ax.set_facecolor('#E6F0FF')
    ax.grid(True, which='both', axis='both', color='blue', linestyle='--', alpha=0.2)

    ax.set_title(f'Complex Wave (Theta: {theta_value}, Scale: {scale})', fontsize=16)
    ax.set_xlabel('Time (ms)', fontsize=18)
    ax.set_ylabel('Amplitude', fontsize=18)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    labels = np.array(labels)
    print(f"Cluster 2 size: {np.sum(labels == 1)}")
    print(f"Cluster 3 size: {np.sum(labels == 2)}")

    if signal_type == 'sine':
        plt.savefig(f'sine_wave_theta_{theta_value}_scale_{scale}_window_{window_size}.svg')
        print('Saving sine wave plot as sine_wave_theta_{theta_value}_scale_{scale}_window_{window_size}.svg')
        plt.close(fig)
    elif signal_type == 'pulse':
        plt.savefig(f'pulse_wave_theta_{theta_value}_scale_{scale}_window_{window_size}.svg')
        print('Saving pulse wave plot as pulse_wave_theta_{theta_value}_scale_{scale}_window_{window_size}.svg')
        plt.close(fig)
    else:
        plt.savefig(f'complex_wave_theta_{theta_value}_scale_{scale}_window_{window_size}.svg')
        print('Saving complex wave plot as complex_wave_theta_{theta_value}_scale_{scale}_window_{window_size}.svg')
        plt.close(fig)

# Sine_wave parameters
sine_theta_value = 90
sine_scale = 100
sine_window_size =3

# Pulse_wave parameters
pulse_theta_value = 130
pulse_scale = 100
pulse_window_size = 5

# Complex_wave parameters
complex_theta_value = 95
complex_scale = 100
complex_window_size = 3

print('Running Berseker2 for sine wave...')
sine_start_time = time()
data = patchify(sine_signal, (sine_window_size,))
data = sine_scale * np.ascontiguousarray(data).astype('f8')
sine_centroids, sine_labels = berserker2(data, sine_theta_value, all_out=False)
ext_sine_labels = []
ext_sine_labels.extend((sine_window_size // 2) * [sine_labels[0]])
ext_sine_labels.extend(list(sine_labels))
ext_sine_labels.extend((sine_window_size // 2) * [sine_labels[-1]])
plot_lines_with_color_gradient(np.arange(len(sine_signal)), sine_signal, ext_sine_labels, sine_theta_value, sine_scale, sine_window_size, 'sine')
sine_end_time = time() - sine_start_time
print(f'Sine wave completed in {sine_end_time} seconds')

print('Running Berseker2 for pulse wave...')
pulse_start = time()
data = patchify(pulse_signal, (pulse_window_size,))
data = pulse_scale * np.ascontiguousarray(data).astype('f8')
pulse_centroids, pulse_labels = berserker2(data, pulse_theta_value, all_out=False)
ext_pulse_labels = []
ext_pulse_labels.extend((pulse_window_size // 2) * [pulse_labels[0]])
ext_pulse_labels.extend(list(pulse_labels))
ext_pulse_labels.extend((pulse_window_size // 2) * [pulse_labels[-1]])
plot_lines_with_color_gradient(np.arange(len(pulse_signal)), pulse_signal, ext_pulse_labels, pulse_theta_value, pulse_scale, pulse_window_size, 'pulse')
pulse_end_time = time() - pulse_start
print(f'Pulse wave completed in {pulse_end_time} seconds')

print('Running Berseker2 for complex wave...')
complex_start = time()
data = patchify(sine_pulse_combo, (complex_window_size,))
data = complex_scale * np.ascontiguousarray(data).astype('f8')
complex_centroids, complex_labels = berserker2(data, complex_theta_value, all_out=False)
ext_complex_labels = []
ext_complex_labels.extend((complex_window_size // 2) * [complex_labels[0]])
ext_complex_labels.extend(list(complex_labels))
ext_complex_labels.extend((complex_window_size // 2) * [complex_labels[-1]])
plot_lines_with_color_gradient(np.arange(len(sine_pulse_combo)), sine_pulse_combo, ext_complex_labels, complex_theta_value, complex_scale, complex_window_size, 'complex')  
complex_end_time = time() - complex_start
print(f'Complex wave completed in {complex_end_time} seconds')

print('Total time taken for all signals: ', sine_end_time + pulse_end_time + complex_end_time)