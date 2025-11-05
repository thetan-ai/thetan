import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


import numpy as np
import matplotlib.pyplot as plt


def diem_stat(N, maxV, minV):
    """
    Generates the Normalizing and Detrending values to correctly compute the DIEM Similarity.

    Parameters:
        N (int): Number of elements in vectors a and b.
        maxV (float): Maximum value of the distribution.
        minV (float): Minimum value of the distribution.

    Returns:
        exp_center (float): Median of Euclidean distances.
        vard (float): Variance of Euclidean distance distribution.
        std_one (float): Standard deviation of DIEM.
        orth_med (float): Median DIEM of orthogonal projections.
        min_DIEM (float): Minimum DIEM.
        max_DIEM (float): Maximum DIEM.
    """

    num_samples = int(1e5)

    # Generate uniform distributions
    a = [np.random.uniform(minV, maxV, (N, 1)) for _ in range(num_samples)]
    b = [np.random.uniform(minV, maxV, (N, 1)) for _ in range(num_samples)]

    # Compute orthogonal vectors
    ort = [np.linalg.svd(a_i.T)[2][-1].reshape(-1, 1) for a_i in a]

    # Compute Euclidean distances
    d = np.array([np.linalg.norm(a_i - b_i) for a_i, b_i in zip(a, b)])
    dort = np.array([np.linalg.norm(a_i - ort_i) for a_i, ort_i in zip(a, ort)])

    # Compute DIEM statistics
    exp_center = np.median(d)
    vard = np.var(d)  # Variance of Euclidean distance distribution
    orth_med = (maxV - minV) * (np.median(dort) - exp_center) / vard  # Median DIEM of orthogonal quantities
    adjusted_dist = (maxV - minV) * (d - exp_center) / vard  # Centering and scaling DIEM
    std_one = np.std(adjusted_dist)  # One standard deviation of DIEM

    min_DIEM = -(maxV - minV) * (exp_center / vard)  # Minimum DIEM
    max_DIEM = (maxV - minV) * (np.sqrt(N) * (maxV - minV) - exp_center) / vard  # Maximum DIEM

    return exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM



def diem(synMat1, synMat2, maxV, minV, exp_center, vard):
    """
    Compute the DIEM similarity matrix between two matrices/vectors and optionally plot the results.

    Parameters:
        synMat1 (numpy.ndarray): First matrix (features as rows, samples as columns).
        synMat2 (numpy.ndarray): Second matrix (features as rows, samples as columns).
        maxV (float): Maximum value for normalization.
        minV (float): Minimum value for normalization.
        exp_center (float): Expected center value for scaling.
        vard (float): Variance scaling factor.


    Returns:
        DIEM (numpy.ndarray): Matrix of DIEM similarities.

    """
    # Default parameters
    # plot_toggle = kwargs.get('plot', 'off').lower()
    # text_toggle = kwargs.get('text', 'off').lower()
    # text_size = kwargs.get('text_size', 10)

    # Compute DIEM similarities
    DIEM = (maxV - minV) * ((cdist(synMat1.T, synMat2.T, metric='euclidean') - exp_center) / vard)

    # Ensure symmetric and upper triangular if needed
    if np.allclose(DIEM, DIEM.T):
        DIEM = np.triu(DIEM)
        DIEM[DIEM == 0] = np.nan

    # ax = None
    # if plot_toggle == 'on':
    #     ax = plot_DIEM(DIEM, text_toggle, text_size, 1.1 * np.nanmin(DIEM), np.nanmax(DIEM))

    return DIEM


def plot_diem(DIEM, text_toggle, text_size, minD, maxD):
    """
    Plot the DIEM similarity matrix.

    Parameters:
        DIEM (numpy.ndarray): Matrix of DIEM similarities.
        text_toggle (str): 'on' or 'off' to toggle text display on the plot.
        text_size (int): Font size for text in the plot.
        minD (float): Minimum value for colormap normalization.
        maxD (float): Maximum value for colormap normalization.

    Returns:
        ax (matplotlib.axes.Axes): Axes object of the DIEM plot.
    """
    # Custom colormap
    MITred = np.array([163, 31, 52]) / 255
    cmap = ListedColormap(np.vstack([[1, 1, 1], plt.cm.autumn(np.linspace(0, 1, 256))]))

    # Create the plot
    fig, ax = plt.subplots()
    img = ax.imshow(DIEM, cmap=cmap, vmin=minD, vmax=maxD)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('|DIEM|', fontsize=10)

    ax.set_xticks(range(DIEM.shape[1]))
    ax.set_yticks(range(DIEM.shape[0]))
    ax.set_xticklabels(range(1, DIEM.shape[1] + 1))
    ax.set_yticklabels(range(1, DIEM.shape[0] + 1))
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

    # Add text to the plot
    if text_toggle == 'on':
        for i in range(DIEM.shape[0]):
            for j in range(DIEM.shape[1]):
                if not np.isnan(DIEM[i, j]):
                    ax.text(j, i, f"{DIEM[i, j]:.0f}", ha='center', va='center',
                            fontsize=text_size, fontweight='bold', color='black')

    fig.patch.set_facecolor('white')
    plt.show()

    return ax


if __name__ == '__main__':

    N = 100000
    D = 4
    for D in np.arange(2, 200, 2):
        print()
        r1 = np.random.rand(D, N)
        center = r1.T.mean(axis=0)
        res = cdist(r1.T, center[None, :])


        corner = np.array([D *(1,)])
        res2 = cdist(corner, center[None, :])
        print('D', D, 'min', res.min(), 'max', res.max(), 'corner_center', res2[0])


    t = 1

    for shift in [0 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:

        print()
        print('>>>', shift)

        synMat1 = r1 + shift
        synMat2 = r1 + 0

        print('Vector 1', synMat1)
        print('Vector 2', synMat2)

        # maxV = 1
        # minV = 0
        # exp_center = 0.5
        # vard = 0.2

        maxV = max(synMat1.max(), synMat2.max())
        minV = min(synMat1.min(), synMat2.min())

        print('minV', minV, 'maxV', maxV)

        results = diem_stat(D, maxV, minV)

        print("exp_center:", results[0])
        print("vard:", results[1])
        print("std_one:", results[2])
        print("orth_med:", results[3])
        print("min_DIEM:", results[4])
        print("max_DIEM:", results[5])

        exp_center, vard, std_one, orth_med, min_DIEM, max_DIEM = results

        # Compute and plot DIEM
        res = diem(synMat1, synMat2, maxV, minV, exp_center, vard)

        print('L2D Res >> min', res.min(), 'mean', res.mean(), 'max', res.max(), 'std', res.std())
        # print('DIEM >>', res)
        res2 = cdist(synMat1.T, synMat2.T, metric='euclidean')
        print('L2  Res >> min', res2.min(), 'mean', res2.mean(), 'max', res2.max(), 'std', res2.std())


    t=1