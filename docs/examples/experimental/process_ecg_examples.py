from time import time
import numpy as np
from thetan.berserker import berserker2
import pandas as pd
import wfdb
import ast
from thetan.patchify import patchify
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter1d
from matplotlib.cm import get_cmap  
from matplotlib.colors import Normalize


# Loading the ecg data
dpath = r'/home/reddysai/thetan_ecg_experiments/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records500/00000/'

fname_dat = dpath + r'00001_hr.dat'
fname_hea = dpath + r'00001_hr.hea'


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = r'/home/reddysai/thetan_ecg_experiments/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/' #path/to/ptbxl/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

signal = X_train[0, :, 0]


# Plotting function
def plot_lines_with_color_gradient(x, y, labels, theta_value, scale, window_size, sigma_1, sigma_2, type):
    colormap = get_cmap('Accent')
    # colormap = LinearSegmentedColormap.from_list("custom_cmap", color_map, N=256)
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

    ax.set_title(f'ECG Signal (Theta: {theta_value}, Scale: {scale})', fontsize=16)
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
   
    unique_clusters = np.unique(labels)
    for cluster in unique_clusters:
        mask = labels == cluster
        x_cluster = pd.Series(x[mask])
        y_cluster = pd.Series(y[mask])
        # dynamic_baseline = y_cluster.rolling(window=16, min_periods=1, center=True).min().values # incase if we need dynamic baseline
        segments = np.split(np.arange(len(x_cluster)), np.where(np.diff(x_cluster) != 1)[0] + 1)
        for segment in segments:
            ax.fill_between(x_cluster[segment], y_cluster[segment] , -0.2, color=colormap(norm(cluster)), alpha=0.3)           

    if type == 'qrs_t_combined':
        plt.savefig(f'ecg_qrs_t_combined_theta_{theta_value}_scale_{scale}_window_{window_size}_sigma_{sigma_1}_{sigma_2}.svg')
        print('Saving common QRS and T wave patterns plot as ecg_qrs_t_combined_theta_{theta_value}_scale_{scale}_window_{window_size}_sigma_{sigma_1}_{sigma_2}.svg')
        plt.close(fig)
    elif type == 'half_and_half':
        plt.savefig(f'ecg_half_and_half_theta_{theta_value}_scale_{scale}_window_{window_size}_sigma_{sigma_1}_{sigma_2}.svg')
        print('Saving half and half QRS and T wave patterns plot as ecg_half_and_half_theta_{theta_value}_scale_{scale}_window_{window_size}_sigma_{sigma_1}_{sigma_2}.svg')
        plt.close(fig)
    else:
        plt.savefig(f'ecg_seperated_theta_{theta_value}_scale_{scale}_window_{window_size}_sigma_{sigma_1}_{sigma_2}.svg')
        print('Saving seperated QRS and T wave patterns plot as ecg_seperated_theta_{theta_value}_scale_{scale}_window_{window_size}_sigma_{sigma_1}_{sigma_2}.svg')
        plt.close(fig)


# Parameters for common QRS and T wave patterns: qrs_t_combioned
qt_theta = 1
qt_scale = 100
qt_window_size = 3
qt_sigma_1 = 9
qt_sigma_2 = 9

# Parameters for half and half QRS and T wave patterns: half_and_half
hh_theta = 6
hh_scale = 100
hh_window_size = 7
hh_sigma_1 = 6
hh_sigma_2 = 4

# Parameters for seperated QRS and T wave patterns: seperated
sep_theta = 26
sep_scale = 200
sep_window_size = 3
sep_sigma_1 = 11
sep_sigma_2 = 2

print('Running Berseker2 for common QRS and T wave patterns...')
qt_start_time = time()
filtered_signal = gaussian_filter1d(signal, sigma=qt_sigma_1)
subracted_signal = signal - filtered_signal
filtered_signal = gaussian_filter1d(subracted_signal, sigma=qt_sigma_2)
data = patchify(filtered_signal, (qt_window_size,))
data = qt_scale * np.ascontiguousarray(data).astype('f8')
qt_centroids, qt_labels = berserker2(data, qt_theta, all_out=False)
ext_qt_labels = []
ext_qt_labels.extend((qt_window_size // 2) * [qt_labels[0]])
ext_qt_labels.extend(list(qt_labels))
ext_qt_labels.extend((qt_window_size // 2) * [qt_labels[-1]])
plot_lines_with_color_gradient(np.arange(len(signal)), signal, ext_qt_labels, qt_theta, qt_scale, qt_window_size, qt_sigma_1, qt_sigma_2, 'qrs_t_combined')
qt_end_time = time() - qt_start_time
print(f'Common QRS and T wave patterns completed in {qt_end_time} seconds')

print('Running Berseker2 for half and half QRS and T wave patterns...')
hh_start_time = time()
filtered_signal = gaussian_filter1d(signal, sigma=hh_sigma_1)
subracted_signal = signal - filtered_signal
filtered_signal = gaussian_filter1d(subracted_signal, sigma=hh_sigma_2)
data = patchify(filtered_signal, (hh_window_size,))
data = hh_scale * np.ascontiguousarray(data).astype('f8')
hh_centroids, hh_labels = berserker2(data, hh_theta, all_out=False)
ext_hh_labels = []
ext_hh_labels.extend((hh_window_size // 2) * [hh_labels[0]])
ext_hh_labels.extend(list(hh_labels))
ext_hh_labels.extend((hh_window_size // 2) * [hh_labels[-1]])
plot_lines_with_color_gradient(np.arange(len(signal)), signal, ext_hh_labels, hh_theta, hh_scale, hh_window_size, hh_sigma_1, hh_sigma_2, 'half_and_half')
hh_end_time = time() - hh_start_time
print(f'Half and half QRS and T wave patterns completed in {hh_end_time} seconds')

print('Running Berseker2 for seperated QRS and T wave patterns...')
sep_start_time = time()
filtered_signal = gaussian_filter1d(signal, sigma=sep_sigma_1)
subracted_signal = signal - filtered_signal
filtered_signal = gaussian_filter1d(subracted_signal, sigma=sep_sigma_2)
data = patchify(filtered_signal, (sep_window_size,))
data = sep_scale * np.ascontiguousarray(data).astype('f8')
sep_centroids, sep_labels = berserker2(data, sep_theta, all_out=False)
ext_sep_labels = []
ext_sep_labels.extend((sep_window_size // 2) * [sep_labels[0]])
ext_sep_labels.extend(list(sep_labels))
ext_sep_labels.extend((sep_window_size // 2) * [sep_labels[-1]])
plot_lines_with_color_gradient(np.arange(len(signal)), signal, ext_sep_labels, sep_theta, sep_scale, sep_window_size, sep_sigma_1, sep_sigma_2, 'seperated')
sep_end_time = time() - sep_start_time
print(f'Seperated QRS and T wave patterns completed in {sep_end_time} seconds')

print('Total time taken for all signals: ', qt_end_time + hh_end_time + sep_end_time)