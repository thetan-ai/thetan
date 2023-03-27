import numpy as np
import matplotlib.pyplot as plt


def plot_centroids_clusters(data, centroids, labels=None, title=None, exp_type=None, gt_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True):

    new_colors = np.zeros((len(data), 3))

    if centroids is not None:
        no_clusters = len(centroids)
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
    plt.get_current_fig_manager().window.showMaximized()

    if show:
        plt.show()
