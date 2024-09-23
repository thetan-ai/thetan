import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
# import os
# os.environ["XDG_SESSION_TYPE"] = "xcb"

def plot_centroids_clusters(data, centroids, labels=None, title=None, exp_type=None, gt_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True, maxwindow=False, savefn=None, centroids_label=True):

    new_colors = np.zeros((len(data), 3))

    if centroids is not None:
        no_clusters = len(centroids)
        cluster_colors = np.random.rand(no_clusters, 3)
    else:
        if labels is not None:
            no_clusters = np.max(labels) + 1
            cluster_colors = np.random.rand(no_clusters, 3)

    if labels is not None:

        for ind in range(len(data)):
            new_colors[ind] = cluster_colors[labels[ind]]

    fig = plt.figure()
    plt.title(title)
    plt.axis('equal')
    plt.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)

    if centroids is not None:

        plt.plot(centroids[:, 0], centroids[:, 1], 'rx')

        if centroids_label:
            for c in range(centroids.shape[0]):
                plt.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if gt_centroids is not None:
        plt.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'bx')

    if uncertain_data is not None:
        plt.plot(uncertain_data[:, 0], uncertain_data[:, 1], 'g.')

    cnt = 0
    if final_counts is not None:
        for i in range(final_counts.shape[0]):
            prev = cnt
            cnt += final_counts[i]
            if final_counts[i] > 1:
                plt.text(data[i, 0], data[i, 1], str(final_nearc_value[prev: cnt]) + ' ' + str(labels[i]), fontsize=8)

    if merged_centroids is not None:
        plt.plot(merged_centroids[:, 0], merged_centroids[:, 1], 'mo')

    # pl.get_current_fig_manager().full_screen_toggle()

    if maxwindow:
        plt.get_current_fig_manager().window.showMaximized()

    if show:
        plt.show()

    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight')



def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    curves = []
    ys = []
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y, 
                             curve+y, zorder=len(data)-i+1, color=fill)
        plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
    if labels:
        plt.yticks(ys, labels)

    plt.show()


def plot_stacked_dists():

    data = [norm.rvs(loc=i, scale=2, size=50) for i in range(8)]
    ridgeline(data, overlap=.85, fill='y')


if __name__ == '__main__':

    # plot_stacked_dists()
    pass