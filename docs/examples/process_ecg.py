
from time import time
import numpy as np
import pylab as pl
from thetan.berserker import berserker2
import pandas as pd
import wfdb
import ast
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap



def plot_lines_with_point_colors(x, y, colors):


    # x = np.linspace(0, 3 * np.pi, 500)
    # y = np.sin(x)
    # dydx = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

    # # Create a continuous norm to map from data points to colors
    # norm = plt.Normalize(dydx.min(), dydx.max())
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # # Set the values used for colormapping
    # lc.set_array(dydx)
    # lc.set_linewidth(2)
    # line = axs[0].add_collection(lc)
    # fig.colorbar(line, ax=axs[0])

    # Use a boundary norm instead
    # cmap = ListedColormap(['r', 'g', 'b'])
    # norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    lc = LineCollection(segments)
    lc.set_array(colors)
    lc.set_linewidth(2)
    line = axs[1].add_collection(lc)
    fig.colorbar(line, ax=axs[1])

    # axs[0].set_xlim(x.min(), x.max())
    # axs[0].set_ylim(-1.1, 1.1)
    plt.show()



dpath = r'C:\Users\Eleftherios\Data\ptb_xl_ecg\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\'

fname_dat = dpath + r'00001_hr.dat'
fname_hea = dpath + r'00001_hr.hea'


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = dpath #'path/to/ptbxl/'
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





t = 1

# pl.plot(X_train[0, :, 0])

signal = X_train[0, :, 0]

window_size = 5

data = patchify(signal, (window_size,))

scale = 100

data = scale * np.ascontiguousarray(data).astype('f8')

theta_value = 40.

print('Starting Berserker')
t = time()
centroids, labels = berserker2(data, theta_value, all_out=False)
print('Duration is ', time() - t)

# TODO use unpatchify
ext_labels = []
ext_labels.extend((window_size // 2) * [labels[0]])
ext_labels.extend(list(labels))
ext_labels.extend((window_size // 2) * [labels[-1]])
t = 1

colors = np.random.rand(centroids.shape[0], 3)

fcolors = np.zeros((signal.shape[0], 3))
for i, l in enumerate(ext_labels):
    fcolors[i] = colors[l]

# pl.plot(range(signal.shape[0]), signal, c=fcolors)
# plot_lines_with_point_colors(np.arange(signal.shape[0]), signal, fcolors)

from fury import actor, window

scene = window.Scene()
# scene.background((1, 1, 1))
scene.projection('parallel')



pts = np.ascontiguousarray(np.vstack((np.arange(signal.shape[0]), scale * signal, np.zeros(signal.shape[0]))).T)

cline = actor.line([pts], fcolors, linewidth=2)

scene.add(cline)

window.show(scene)

t = 2