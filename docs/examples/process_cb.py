"""Please, do:

git submodule update --remote

dataset url: https://clustering-benchmarks.gagolewski.com/weave/data-v1.html#wut/circles

"""
from dataclasses import make_dataclass
import os
from os.path import join as pjoin
import time


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import scipy as sp

from thetan.berserker import berserker2


DATA_PATH = pjoin(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'data',
                  'clustering-data-v1')
DATA_PATH = os.path.abspath(DATA_PATH)

col = ["#000000", "#DF536B", "#61D04F", "#2297E6", "#28E2E5", "#CD0BBC", "#F5C710"] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab10"].colors]  + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20"].colors]  + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20b"].colors] + \
    [matplotlib.colors.to_hex(c) for c in plt.colormaps["tab20c"].colors]

mrk = ["o", "^", "+", "x", "D", "v", "s", "*", "<", ">", "2"]

DataSet = make_dataclass(
    "ClusteringBenchmark",
    ["battery", "dataset", "description", "data", "labels", "n_clusters"]
)


def plot_dataset(dataset, axis=None, title=None, xlabel=None, ylabel=None,
                 start=0, xlim=None, ylim=None, **kwargs):
    """Draws a scatter plot of the labels."""
    if not isinstance(dataset, (list, tuple)):
        dataset = [dataset, ]
    nb_subplot = sum(len(data.labels) for data in dataset)
    print(nb_subplot)
    for k, data in enumerate(dataset, 1):
        for i in range(len(data.labels)):
            plt.subplot(1, nb_subplot, k+i)
            plot_scatter(data.data, labels=data.labels[i]-1,
                         axis="equal", title=f"labels{k+i}", alpha=0.05)
    plt.show()


def plot_scatter(X, y=None, labels=None, axis=None, title=None, xlabel=None,
                 ylabel=None, xlim=None, ylim=None, **kwargs):
    """Draws a scatter plot

    NOTE
    ----
    base on https://github.com/gagolews/genieclust/blob/master/genieclust/plots.py
    """
    X = np.array(X)
    if X.ndim == 2:
        if not X.shape[1] == 2:
            raise ValueError("`X` must have 2 columns.")
        if y is not None:
            raise ValueError("If `X` is a matrix, `y` should not be given.")
    elif X.ndim == 1:
        if y is None:
            raise ValueError("If `X` is a vector, `y` should be provided.")

        y = np.array(y)
        if y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("`X` and `y` must have the same shape.")
        X = np.column_stack((X, y))
    else:
        raise ValueError("Invalid `X`.")

    if labels is None:
        labels = np.repeat(0, X.shape[0])
    else:
        labels = np.array(labels)
    if labels.ndim != 1 or X.shape[0] != labels.shape[0]:
        raise ValueError("Incorrect shape of `labels`.")

    for i in np.unique(labels):  # 0 = black, 1 = red, etc.
        plt.scatter(
            X[labels == i, 0],
            X[labels == i, 1],
            c=col[(i) % len(col)],
            marker=mrk[(i) % len(mrk)],
            **kwargs
        )

    if axis is not None:
        plt.axis(axis)

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)


def get_battery_names():
    """Get the names of the batteries."""
    batteries = [f for f in os.listdir(DATA_PATH)
                 if os.path.isdir(pjoin(DATA_PATH, f))
                 if not f.startswith(".")]
    return sorted(batteries)


def get_dataset_names(*, battery_name='mnist'):
    """Get the names of the datasets."""
    dataset = [f.replace(".data.gz", "")
               for f in os.listdir(pjoin(DATA_PATH, battery_name))
               if os.path.isfile(pjoin(DATA_PATH, battery_name, f))
               if f.endswith(".data.gz")]
    return sorted(dataset)


def load_dataset(battery, dataset, *, preprocess=True, random_state=None):
    """Load a dataset from the clustering benchmark."""

    data_file = pjoin(DATA_PATH, battery, f"{dataset}.data.gz")
    data = np.loadtxt(data_file, ndmin=2)

    if data.ndim != 2:
        raise ValueError("Not a matrix.")

    if preprocess:
        # Normalise a data matrix

        # Removes all columns of zero variance (constant). Centres the data
        # around the centroid (so that each column mean is 0).
        # Scales all columns proportionally (so that the total variance is 1;
        # note that this is not the same as standardisation: standard
        # deviations in each column might still be different).
        # Adds a tiny amount of noise to minimise the risk of having
        # duplicate points.
        data = data[:, data.var(axis=0) > 0]

        # centre + scale all columns proportionally (not: standardise)
        data = (data-data.mean(axis=0))/data.std(axis=None, ddof=1)

        # add a tiny bit of white noise:
        noise_factor = 1e-6
        data += sp.stats.norm.rvs(
            loc=0.0, scale=data.std(ddof=1)*noise_factor, size=data.shape,
            random_state=random_state
        )

    labels, idx = [], 0
    while True:
        label_file = pjoin(DATA_PATH, battery, f"{dataset}.labels{idx}.gz")
        if not np.DataSource().exists(label_file):
            break  # stop trying to find more label vectors

        label = np.loadtxt(label_file, dtype="int")
        if label.ndim != 1 or label.shape[0] != data.shape[0]:
            raise ValueError(f"Incorrect number of labels in '{label_file}'.")

        labels.append(label)
        idx += 1

    n_clusters = np.array([np.max(ll) for ll in labels])

    desc_file = pjoin(DATA_PATH, battery, f"{dataset}.txt")
    with np.DataSource().open(desc_file, "r") as readme_file:
        description = readme_file.read()

    return DataSet(
        battery=battery,
        dataset=dataset,
        description=description,
        data=data,
        labels=labels,
        n_clusters=n_clusters,
    )


if __name__ == "__main__":
    print(get_battery_names())
    print(get_dataset_names(battery_name='mnist'))
    # dataset = load_dataset('wut', 'x2')
    # dataset = load_dataset('wut', 'circles')
    # dataset = load_dataset('wut', 'cross')
    # dataset = load_dataset('wut', 'z3')
    dataset = load_dataset('sipu', 'birch1')
    # dataset = load_dataset('other', 'hdbscan')
    # plot_dataset(dataset)

    print('Starting Berserker')
    # theta_value = 1.85  # x2
    # theta_value = 1.20  # circles -> GOOD
    # theta_value = 2.7 # cross
    # theta_value = 1.1 # z3 -> GOOD
    theta_value = 0.25 # birch1 -> GOOD
    t_start = time.perf_counter()
    centroids, labels = berserker2(np.ascontiguousarray(dataset.data),
                                   theta_value, all_out=False, verbose=False)
    print('Duration is ', time.perf_counter() - t_start)
    print(f'cluster size: {len(centroids)} / {dataset.n_clusters} expected')
    results = DataSet(
        battery=dataset.battery,
        dataset=dataset.dataset,
        description=dataset.description,
        data=dataset.data,
        labels=[labels],
        n_clusters=[len(centroids)]
    )
    plot_dataset([results, dataset])
