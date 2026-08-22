from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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
    import matplotlib.colors as mcolors
    from matplotlib import cm

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

def project_to_2d(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

def visualize_separation(data, centroids, labels, result, save_path=None, show=True):
    """
    Visualize the separation (discs with bridge in between) results for n-dimensional data.
    Automatically projects to 2D if data has more than 2 dimensions.

    Parameters:
    * data: Original Data.
    * centroids: Original centroids.
    * labels: Original labels.
    * results: dict which the filtered_data, filtered_centroids, filtered_labels.
    * save_path: Path to save figure.
    * show: Whether to display the plot.

    Returns:
    * A plot.
    """
    n_dims = data.shape[1]
    if n_dims > 2:
        print(f'Projecting {n_dims}D data to 2D using PCA.')
        data_2d = project_to_2d(data)
        centroids_2d = project_to_2d(centroids)
        filtered_data_2d = project_to_2d(result['filtered_data'])
        filtered_centroids_2d = project_to_2d(result['filtered_centroids'])
        removed_data_2d = project_to_2d(result['removed_data']) if len(result['removed_data']) > 0 else np.array([]).reshape(0,2)
    else:
        data_2d = data
        centroids_2d = centroids
        filtered_data_2d = result['filtered_data']
        filtered_centroids_2d = result['filtered_centroids']
        removed_data_2d = result['removed_data'] if len(result['removed_data']) > 0 else np.array([]).reshape(0,2)

    counts = result['counts']
    sparse_indices = result['sparse_indices']
    removed_mask = result['removed_mask']

    fig, axes = plt.subplots(2,2,figsize=(14,14))

    ax = axes[0,0]
    ax.scatter(data_2d[:,0], data_2d[:,1],c=labels,cmap='tab20',s=1,alpha=0.5)
    ax.scatter(centroids_2d[:,0],centroids_2d[:,1],c='red',marker='x',s=50,linewidths=2)
    ax.set_title(f'Original Clustering \n({len(centroids)} centroids, {len(data)} points)')
    ax.set_aspect('equal')

    ax = axes[0,1]
    ax.scatter(data_2d[:,0],data_2d[:,1],c='lightgray',s=1,alpha=0.3)
    scatter = ax.scatter(centroids_2d[:,0], centroids_2d[:,1], c=counts, cmap='RdYlGn', s=100, edgecolors='black',linewidths=1)
    if len(sparse_indices) > 0:
        sparse_centroids_2d = centroids_2d[sparse_indices]
        ax.scatter(sparse_centroids_2d[:,0], sparse_centroids_2d[:,1],facecolors='none',edgecolors='red',s=250,linewidths=3)
    plt.colorbar(scatter, ax=ax, label='Neighbor Count')
    ax.set_title('Centroid Neighbor Counts \n(Red Circles = Sparse Centroids, to remove)')
    ax.set_aspect('equal')

    ax = axes[1,0]
    kept_2d = data_2d[~removed_mask]
    ax.scatter(kept_2d[:, 0], kept_2d[:, 1], c='lightblue', s=1, alpha=0.5, label='Kept')
    if len(removed_data_2d) > 0:
        ax.scatter(removed_data_2d[:,0],removed_data_2d[:,1],c='red', s=2,alpha=0.8,label='Removed')
    if len(sparse_indices) > 0:
        sparse_centroids_2d = centroids_2d[sparse_indices]
        ax.scatter(sparse_centroids_2d[:, 0], sparse_centroids_2d[:,1],c='darkred', marker='x', s=100, linewidths=2)
    ax.legend()
    ax.set_title(f'Removed points ({np.sum(removed_mask)} points, {len(sparse_indices)} centroids)')
    ax.set_aspect('equal')

    ax = axes[1,1]
    ax.scatter(filtered_data_2d[:,0],filtered_data_2d[:,1],c=result['filtered_labels'],cmap='tab20',s=1,alpha=0.5)
    ax.scatter(filtered_centroids_2d[:,0], filtered_centroids_2d[:,1],c='red', marker='x', s=50, linewidths=2)
    ax.set_title(f'Separated Regions \n({len(result['filtered_centroids'])} centroids, {len(result['filtered_data'])} points)')
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved figure to {save_path}')
    if show:
        plt.show()
    else:
        plt.close()


def plot_scf(data, mask, scf_result, slice_numbers, threshold_value, overlay_perc=40, save_fig=False):
    data = np.asarray(data)
    mask = np.asarray(mask)
    scf_result = np.asarray(scf_result)

    if data.ndim != 3:
        raise ValueError("data, mask and scf_result must be a 3D array")
    if data.shape != mask.shape or data.shape != scf_result.shape:
        raise ValueError("data, mask, and scf_result must have the same shape")

    if isinstance(slice_numbers, (int, np.integer)):
        slice_numbers = [int(slice_numbers)]
    else:
        slice_numbers = list(slice_numbers)

    if not slice_numbers:
        raise ValueError("At least one slice number must be supplied")

    if not 0 <= overlay_perc <= 100:
        raise ValueError("overlay_perc must be between 0 and 100")

    for z in slice_numbers:
        if not isinstance(z, (int, np.integer)):
            raise ValueError("slice numbers must be integers")
        if not 0 <= z < data.shape[2]:
            raise ValueError(f'Slice {z} is outside the valid range 0 to {data.shape[2] - 1}')

    number_of_slices = len(slice_numbers)
    fig, axes = plt.subplots(number_of_slices, 5, figsize = (18, 4 * number_of_slices), squeeze=False)

    overlay_alpha = overlay_perc / 100

    for row, z in enumerate(slice_numbers):
        data_slice = data[:, : ,z]
        mask_slice = mask[:, :, z]
        scf_slice = scf_result[:, :, z]

        thresholded_scf = scf_slice > threshold_value

        axes[row, 0].imshow(data_slice.T, cmap='gray', origin='lower')
        axes[row, 0].set_title(f'Original data: slice {z}')

        axes[row, 1].imshow(mask_slice.T, cmap='gray', origin='lower')
        axes[row, 1].set_title('Mask')

        axes[row, 2].imshow(scf_slice.T, cmap='magma', origin='lower')
        axes[row, 2].set_title('SCF result')

        axes[row, 3].imshow(thresholded_scf.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[row, 3].set_title(f'SCF > {threshold_value:g}')

        axes[row, 4].imshow(data_slice.T, cmap='gray', origin='lower')
        overlay = np.ma.masked_where(~thresholded_scf.T, thresholded_scf.T)

        axes[row, 4].imshow(overlay, cmap='Reds', origin='lower', alpha= overlay_alpha, vmin=0, vmax=1)
        axes[row, 4].set_title(f'SCF overlay: {overlay_perc:g}%')

        for axis in axes[row]:
            axis.axis('off')
    fig.tight_layout()

    if save_fig:
        output_path = (Path('scf_plot.png') if save_fig is True else Path(save_fig))
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f'Saved_figure: {output_path.resolve()}')

    return fig
if __name__ == '__main__':

    # plot_stacked_dists()
    pass
