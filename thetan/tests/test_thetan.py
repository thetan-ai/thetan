from time import time
import numpy as np
import numpy.testing as npt
from thetan import reorder_labels, sequential
from thetan import centroids_from_labels
import pylab as pl
from thetan.berserker import candidate_labels, berserker_dbg
import matplotlib.pyplot as plt
from PIL import Image
from thetan.viz import plot_centroids_clusters
import matplotlib
matplotlib.use('Agg', force=True)
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_simple_hashing():

    X = 1000 * (np.random.rand(100000, 3) - 0.5)
    spacing = 40
    Xf = np.floor(X/spacing)
    big_ints = np.random.randint(100000, 200000, X.shape[1])
    hash_prep = Xf * big_ints
    hashes = np.sum(hash_prep, axis=1)
    unique_hashes = np.unique(hashes)
    print(unique_hashes.shape[0] / hashes.shape[0])


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

    fig = pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)

    if centroids is not None:

        pl.plot(centroids[:, 0], centroids[:, 1], 'rx')

        for c in range(centroids.shape[0]):
            pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if gt_centroids is not None:
        pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'bx')

    if uncertain_data is not None:
        pl.plot(uncertain_data[:, 0], uncertain_data[:, 1], 'g.')

    cnt = 0
    if final_counts is not None:
        for i in range(final_counts.shape[0]):
            prev = cnt
            cnt += final_counts[i]
            if final_counts[i] > 1:
                pl.text(data[i, 0], data[i, 1], str(final_nearc_value[prev: cnt]) + ' ' + str(labels[i]), fontsize=8)

    if merged_centroids is not None:
        pl.plot(merged_centroids[:, 0], merged_centroids[:, 1], 'mo')

    # pl.get_current_fig_manager().full_screen_toggle()
    # pl.get_current_fig_manager().window.showMaximized()

    # pl.show()
    matplotlib.pyplot.close(fig)

def build_nice2(centroids, labels):
    nice = np.ones(labels.shape[0], dtype=labels.dtype)
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    # clusters = {}
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    for c in clusters:
        if len(c) > 1:
            maxc = np.max(c)
            candidates.append(maxc)

    # deleting will change the labels of other Xs too which is not ideal
    # centroids_reduced = np.delete(final_centroids, candidates, axis=0)

    for i, f in enumerate(labels):
        for can in candidates:
            if labels[i] == can:
                nice[i] = 0

    return nice, candidates, clusters


def test_build_nice():

    centroids = np.array([[0, 0, 0],
                          [0.2, 0, 0],
                          [3, 3, 3.],
                          [4, 4, 4.],
                          [5, 5, 5.],
                          [6, 6, 6],
                          [6.2, 6, 6]])

    labels = np.array([0, 0, 1, 2, 3, 4, 4, 4])

    nice, candidates, clusters = build_nice2(centroids, labels)


    t = 1


def test_reorder_labels():

    centroids = np.random.rand(7, 3)
    labels = np.array([0, 0, 1, 2, 4, 5, 6, 6])
    labels_updated = reorder_labels(centroids,
                                    labels,
                                    None, [2, 5])

    npt.assert_equal(labels_updated,
                     np.array([0, 0, 1, 2, 3, 4, 4, 4]))


def test_2d_data(verbose=False):

    # n_local_samples = 400
    n_local_samples = 1000

    dims = 2

    blob_dis = 5
    cluster_x = 40
    cluster_y = 10

    normal_distribution = True
    uniform_distribution = False

    normal_sigma = 1.0
    uniform_low = -2.
    uniform_high = 2.

    theta_value = 3.6 # 3.0 still okay for uniform distributions with inf_norm
    sg1_value = 3.6 #3.6
    sg2_value = 3.6
    g1_value = 3.6
    g1h_value = 3.6
    g1d_value = 3.6


    xx = np.arange(-1 * blob_dis*cluster_x // 2, blob_dis * cluster_x // 2, blob_dis)
    yy = np.arange(-1 * blob_dis*cluster_y // 2, blob_dis * cluster_y // 2, blob_dis)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], dims))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)

    for i in range(n_centres.shape[0]):
        cpos = n_centres[i]

        if normal_distribution:
            local_X = cpos + np.random.normal(0, normal_sigma, (n_local_samples, dims))

        if uniform_distribution:
            local_X = cpos + np.random.uniform(uniform_low,
                                               uniform_high,
                                               (n_local_samples, dims))

        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    gt_centroids = np.array(gt_centroids)

    points = X
    labels = y
    sz = len(n_centres)

    centroids = gt_centroids

    # Randomize the data
    ind = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
    points = points[ind].copy()
    labels = labels[ind].copy()



def generate_circles(n_samples=15, inner_radius=60, outer_radius=15,
                     density=500, size=(200, 200), labellize=False,
                     return_coord=True):
    """Generate small circle on a bigger one.

    Parameters
    ----------
    n_samples: int
        number of outer circle

    inner_radius: int
        radius of the bigger circle

    outer_radius: int
        radius of the small circle

    density: int
        number of point on each circle uniformly distributed

    size: tuple
        2d image size

    labellize: bool
        Switch on/off if you want to labellize each circle or not

    Returns
    -------
    arr: 2d array
        numpy array containing the drawing

    """
    arr = np.zeros(size)
    label_value = 1
    coord_list = []
    for theta in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
        x = int(size[1]//2 + inner_radius * np.cos(theta))
        y = int(size[0]//2 + inner_radius * np.sin(theta))
        # arr[y, x] = 1
        if labellize:
            label_value += 1
        for _ in range(density):
            r = outer_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = x + r * np.cos(theta)
            y1 = y + r * np.sin(theta)
            if return_coord:
                coord_list.append([y1, x1])
            arr[int(y1), int(x1)] = label_value

    if return_coord:
        return arr, coord_list
    return arr


def test_grid_of_disks():

    # np.random.seed(0)
    sz = (50, 6)

    coords = []
    centroids = []

    for x, y in np.ndindex(sz):

        for d in range(100):
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = 5 * x + r * np.cos(theta)

            y1 = 5 * y + r * np.sin(theta)

            coords.append([x1, y1])

        centroids.append([5 * x, 5 * y])

    points = np.array(coords)
    print(points.shape[0])

    centroids = np.array(centroids, dtype='f8')

    print(centroids.shape[0])

    theta_value = 2.0

    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker_dbg(points, theta_value, all_out=True, verbose=True)

    print(even_centroids_ex2.shape[0])

    plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

    t = 1


def test_ring_of_disks():


    # Note: After outer_radius 13 the problem becomes much harder. An iterative version of berserker could resolve this issue

    _, points = generate_circles(n_samples=15, inner_radius=60, outer_radius=12,
                                 density=500, size=(200, 200), labellize=False,
                                 return_coord=True)
    points = np.asarray(points)

    #print(points.shape)
    range_of_thr = np.arange(21, 23, 0.5)

    for theta_value in range_of_thr:

        print('theta value', theta_value)

        even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker_dbg(points, theta_value, all_out=True, verbose=True)

        print(even_centroids_ex2.shape[0])


        plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
        plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

        t = 1

def points_on_triangle(v, n):
    """ Give n random points uniformly on a triangle.

    The vertices of the triangle are given by the shape
    (2, 3) array *v*: one vertex per row.
    """
    x = np.sort(np.random.rand(2, n), axis=0)

    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v


def test_two_triangles(verbose=False):
    v = np.array([(-0.5, 0.5), (-0.5, -0.5), (0.5, 0)])
    points = points_on_triangle(v, 2000)

    inbetween_distance = 0.3

    v = np.array([(inbetween_distance + 0.5, 0.5), (inbetween_distance + 0.5, -0.5), (inbetween_distance - 0.5, 0)])
    points2 = points_on_triangle(v, 2000)

    points = np.concatenate((points, points2), axis=0)

    theta_value = .6

    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2 = berserker_dbg(points, theta_value, all_out=True, verbose=True)

    if verbose:
        plot_centroids_clusters(points, initial_centroids, None, title='ic' + str(initial_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids, None, title='ec' + str(even_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex, None, title='ecx' + str(even_centroids_ex.shape), show=False)
        plot_centroids_clusters(points, p_centroids, p_labels, title='pc' + str(p_centroids.shape), show=False)
        plot_centroids_clusters(points, even_centroids2, None, title='ec2' + str(even_centroids2.shape), show=False)
        plot_centroids_clusters(points, even_centroids_ex2, updated_labels_ex2, title='ecx2' + str(even_centroids_ex2.shape), show=True)

    npt.assert_almost_equal(even_centroids_ex2.shape[0], 2)

def get_points_trapeze(points1):

    m1 = (points1[2]-points1[0])
    m1= m1[1]/m1[0]
    m2 = -m1

    c1 =  points1[0][1]-m1*points1[0][0]
    c2 = points1[1][1] - m2*points1[1][0]
    x = np.arange(points1[0][0],points1[2][0],0.01)
    y_ulims = m1*x+c1
    y_dlims = m2*x+c2

    stack_arr =[]
    for i,(dlim,ulim) in enumerate(zip(y_dlims,y_ulims)):
        arr = np.random.uniform(ulim,dlim,size=10)
        #mean = (ulim+dlim)//2

        #arr = np.random.normal(mean,scale=0.1,size=10)
        #print(arr)
        points = [(x[i],j) for j in arr]

        stack_arr.extend(points)

    stack_arr = np.array(stack_arr)
    return stack_arr

def euc(p1,p2):
    return np.linalg.norm(p1-p2)

def get_centroid_trap(points):
    p = euc(points[0],points[1])
    print("P: ",p)
    h = points[2][0]-points[0][0]
    q = euc(points[3],points[2])

    X = h/2
    Y = max(p/2,q/2)

    return points[0][0]+X,Y

def get_points_circle(centre,r,num_points=10):
    circle_points =[]
    theta = 0
    while (theta <2*np.pi):
        #print("T: ",r*np.sin(theta))
        pointsX = np.random.uniform(centre[0],centre[0]+r*np.cos(theta),size =num_points).reshape(-1,1)
        pointsY = np.random.uniform(0,r*np.sin(theta),size=num_points).reshape(-1,1)

        circle_points.append(np.hstack([pointsX,pointsY]))

        theta = theta+0.1

    return circle_points

def test_2_trapeze():
    points1 = np.array([[-0.5,-0.5],[-0.5,0.5],[2,-0.25],[2,0.25]])
    points2 = np.array([[2.5,-0.25],[2.5,0.25],[3.5,-0.5],[3.5,0.5]])


    stack_arr1 = get_points_trapeze(points1)
    mean1 = np.mean(stack_arr1,axis=0)

    stack_arr2 = get_points_trapeze(points2)
    mean2 = np.mean(stack_arr2,axis=0)

    points_arr = np.vstack([stack_arr1,stack_arr2])
    gt_centroids = np.array([mean1,mean2])

def test_neighb_compactness():

    shape = (4, 3, 5, 6, 2)

    multipliers = np.random.randint(1000, 2000, len(shape))
    multipliers_diff = np.random.randint(1000, 2000, len(shape) - 1)

    proximal_hashes = []
    proximal_diff_hashes = []

    for index in np.ndindex(shape):

        proximal_hash = np.sum(np.asarray(index) * multipliers)

        diff_index = np.diff(index)
        proximal_diff_hash = np.sum(diff_index * multipliers_diff)

        print(index, diff_index, proximal_hash, proximal_diff_hash, np.sum(diff_index**2))
        proximal_hashes.append(proximal_hash)
        proximal_diff_hashes.append(proximal_diff_hash)

    print(len(proximal_hashes))
    print(len(np.unique(proximal_hashes)))

    print(len(proximal_diff_hashes))
    print(len(np.unique(proximal_diff_hashes)))

    t = 1

def _centroids_from_labels(points, labels):

    new_centroids = []
    for i, l in enumerate(range(labels.max() + 1)):
        # print(i)
        avg = np.average(points[labels==l], axis=0)
        new_centroids.append(avg)

    t = 0
    return np.ascontiguousarray(new_centroids)

def test_centroids_from_labels():

    dims = 3
    theta_value = 50

    points = 1000 * (np.random.rand(1000, dims) - 0.5)

    tb1 = time()
    centroids1, labels1 = sequential(points, theta_value)
    tb2 = time()


    t0 = time()
    _centroids = _centroids_from_labels(points, labels1)
    t1 = time()
    centroids2 = centroids_from_labels(points, labels1)
    t2 = time()

    print('G1 version')
    print(centroids1.shape)
    print(centroids1[:4])

    print('Pythonic version', t1 - t0)
    print(_centroids[:4])
    print(_centroids.shape)

    print('Cythonic version', t2 - t1)
    print(centroids2[:4])
    print(centroids2.shape)

    # plot_centroids_clusters(points, _centroids, labels1, title='Yo before', show=False)
    # plot_centroids_clusters(points, centroids2, labels1, title='Yo after', show=True)

    npt.assert_array_almost_equal(centroids2, centroids1)
    npt.assert_array_almost_equal(centroids2, _centroids)

if __name__ == '__main__':

    pass
