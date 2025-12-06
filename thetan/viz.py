import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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


def plot_centroids_colormap_by_counts(data, centroids, centroid_counts, labels=None, title=None,
                                      cmap='magma', alpha_points=0.2, centroids_label=True,
                                      show=True, maxwindow=False, savefn=None):
    """Plot data with same point coloring behavior as `plot_centroids_clusters`,
    but color centroids using a colormap according to `centroid_counts`.

    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

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
    plt.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=alpha_points)

    # Validate centroid_counts length
    counts = np.asarray(centroid_counts)
    if centroids is None:
        raise ValueError('centroids must be provided')
    if counts.shape[0] != centroids.shape[0]:
        raise ValueError('centroid_counts must have the same length as centroids')

    # Normalize counts to colormap and use counts directly in scatter so matplotlib
    # can associate a mappable for the colorbar.
    norm = mcolors.Normalize(vmin=counts.min(), vmax=counts.max())
    cmap_obj = cm.get_cmap(cmap)

    # Marker sizes scaled by counts for visibility
    if counts.max() > counts.min():
        sizes = 30 + (counts - counts.min()) / (counts.max() - counts.min()) * 170
    else:
        sizes = np.full_like(counts, 60.0, dtype=float)

    ax = plt.gca()
    sc = ax.scatter(centroids[:, 0], centroids[:, 1], c=counts, cmap=cmap_obj, norm=norm,
                    s=sizes, edgecolors='k')

    # Add colorbar mapping centroid counts -> colormap, attach to the figure and axes
    try:
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(counts)
        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label('centroid count')
    except Exception:
        pass

    if centroids_label:
        for c in range(centroids.shape[0]):
            plt.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if maxwindow:
        plt.get_current_fig_manager().window.showMaximized()

    if show:
        plt.show()

    if savefn is not None:
        plt.savefig(savefn, bbox_inches='tight')



if __name__ == '__main__':

    # plot_stacked_dists()
    pass


