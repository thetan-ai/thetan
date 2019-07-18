import numpy as np
from dipy.segment.clustering import QuickBundlesX
from dipy.segment.metric import IdentityFeature, Feature
from dipy.segment.metric import AveragePointwiseEuclideanMetric, EuclideanMetric, SumPointwiseEuclideanMetric

from dipy.io.image import load_nifti, save_nifti
from dipy.viz.app import horizon
from dipy.segment.metric import Metric
from time import time
from fury import actor, window, ui
from itertools import cycle, islice



def sklearn_data():
    from sklearn import cluster, datasets, mixture
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                        noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                        'quantile': .2, 'n_clusters': 2,
                        'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,
                'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
        (aniso, {'eps': .15, 'n_neighbors': 2,
                'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
        (blobs, {}),
        (no_structure, {})]

    all_data = {}

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset
        from pdb import set_trace
        # set_trace()
        my_data = np.zeros((X.shape[0], 3))
        my_data[:, :2] = X.copy()
        print(my_data.shape)

        all_data[i_dataset] = my_data

    return all_data

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


def vectors_3d(run=0, mode='skdata', sampling='uniform', skdata_id=0, thresholds=[250.], sph_radius=0.1,
               fstealth=None, init_viz=False, shuffling=True, chain_maker=True, i_r = 60, o_r = 15, 
               iterations = 1):
    """ skdata or ecubes """
    # added new parameters
    # i_r : inner radius
    # o_r : outer radius
    # iterations : number of times to run qbx


    if mode == 'ecubes':
        if sampling == 'normal':
            my_data_1 = 10 * np.random.normal(size=(1000, 3))
            my_data_2 = 10 * np.random.normal(size=(1000, 3)) + np.array([250, 0, 0])
            my_data_3 = 10 * np.random.normal(size=(1000, 3)) - np.array([250, 0, 0])

        step = 0

        if sampling == 'uniform':
            my_data_1 = 200 * np.random.rand(1000, 3)
            my_data_2 = 200 * np.random.rand(1000, 3) + np.array([200 + step, 0, 0])
            my_data_3 = 200 * np.random.rand(1000, 3) - np.array([200 + step, 0, 0])

        my_data = np.concatenate((my_data_1, my_data_2, my_data_3))
        my_data = np.ascontiguousarray(my_data)

    if mode == 'skdata':

        all_data = sklearn_data()
        my_data = all_data[skdata_id]
        # TODO: Attension only using a small dataset
        my_data = my_data[:200]

    if mode == 'packing':

        #from robotikos.docs.examples.pix_coord import generate_circles
        my_data = generate_circles(inner_radius=i_r, outer_radius=o_r) # Try 60.
        my_data = np.asarray(my_data[1])
        tmp = np.zeros((len(my_data), 3))
        tmp[:, :2] = my_data
        my_data = np.ascontiguousarray(tmp)

        #from pdb import set_trace
        #set_trace()

    #print(my_data.shape)

    class DataSequence:
        def __init__(self, data):

            self.data = data

        def __getitem__(self, i):
            return self.data[i][None, :]

        def __len__(self):
            return self.data.shape[0]
    
    dataseq = DataSequence(my_data)

    vector_feature = IdentityFeature()
    metric = EuclideanMetric(feature=vector_feature)
    
    #qbx = QuickBundlesX(thresholds=thresholds, metric=metric)

    internal_indices = np.arange(my_data.shape[0])
    all_clusters = []
    
    for it in range(iterations): # run qbx with different ordering of data points in each iteration
        
        qbx = QuickBundlesX(thresholds=thresholds, metric=metric)

        if shuffling:
            np.random.seed(run)
            np.random.shuffle(internal_indices)

        internal_indices = np.ascontiguousarray(internal_indices)
        #print(internal_indices)

        tree = qbx.cluster(dataseq, ordering=internal_indices)
        clusters = tree.get_clusters(1)
        print(len(clusters))
        all_clusters.append(clusters) # add clusters to the list of all clusters

        from fury import actor, window, ui

        #init_viz = True
        if init_viz:
            scene = window.Renderer()
            scene.background((1,1,1))
            spheres = actor.sphere(centers=my_data, colors=(1,0,0.7), radii=sph_radius)
            scene.add(spheres)
            window.show(scene)


            new_colors = np.zeros((len(my_data), 3))
            no_clusters = len(clusters)
            cluster_colors = np.random.rand(no_clusters, 3)
            for c, cluster in enumerate(clusters):
                for ind in cluster.indices:
                    new_colors[ind] = cluster_colors[c]

            scene = window.Renderer()
            scene.background((1,1,1))
            spheres = actor.sphere(centers=my_data, colors=new_colors, radii=sph_radius)
            scene.add(spheres)
            if fstealth is None:
                window.show(scene)
            else:
                window.snapshot(scene, fstealth)
                

        
    # return all clusters, which is list of clusters
    # dataseq and my_data
    return all_clusters, dataseq, my_data

import time
def dist(a, b):
    
    #np.sqrt(np.sum((a-b)**2))
    return np.linalg.norm(a-b)
# ------------------------------------------------------------------------------

thr = 18
skdata_id = 0
fstealth = None
chain_maker = False

    
all_clusters, dataseq, my_data = vectors_3d(run=0, mode='packing', skdata_id=0, thresholds=[thr], sph_radius=1, 
                                 init_viz=False, fstealth=fstealth, shuffling=True, chain_maker=chain_maker, 
                                 i_r=60, o_r=13, iterations = 25)

print(my_data)
from fury import actor, window, ui

init_viz = True
if init_viz: # visualize initial data
    scene = window.Renderer()
    scene.background((1,1,1))
    spheres = actor.sphere(centers=my_data, colors=(1,0,0.7), radii=1)
    scene.add(spheres)
    window.show(scene)
    
m = len(all_clusters) - 1
all_centroids = all_clusters[m].centroids # get centroids of last qbx run

for i in range(m): # merge/extend it with all centroids from other runs of qbx
    
    all_centroids.extend(all_clusters[i].centroids)
    
only_c = np.array(all_centroids.copy())
only_c = only_c.reshape((len(all_centroids),3)) # reshape for visualization

print("no of centroids = ", len(all_centroids))

# Run qbx on the merged centroids to get new centroids

thresholds = [20]
vector_feature = IdentityFeature()
metric = EuclideanMetric(feature=vector_feature)

qbx = QuickBundlesX(thresholds=thresholds, metric=metric)

tree = qbx.cluster(all_centroids)
new_clusters = tree.get_clusters(1)  
only_nc = np.array(new_clusters.centroids.copy()).reshape((len(new_clusters.centroids), 3))

print("no of clusters = ", len(only_nc))

if init_viz: 
    # visualize new centroids on data
    scene = window.Renderer()
    scene.background((1,1,1))
    spheres = actor.sphere(centers=my_data, colors=(1,0,0.8), radii=1)
    scene.add(spheres)
    spheres_c = actor.sphere(centers=only_nc, colors=(1,.9,.3), radii=1.8)
    scene.add(spheres_c)
    window.show(scene)
    
# Re-assignment of points to closest centroid

all_new_centroids = only_nc.copy()

ct_st = []

for st in range(len(my_data)):
    
    dist_min = 1000000000000000
    ct_number = 0
    
    for ct in range(len(all_new_centroids)):
        
        dist_val = dist(all_new_centroids[ct], my_data[st])
        
        if dist_min > dist_val:
            
            dist_min = dist_val
            ct_number = ct

    ct_st.append(ct_number)  # assign cluster number to the points that belong in that cluster
    # 0th index means 0th point in data
    # index tells the data point number
    # value on that index means cluster number it belongs to





if init_viz:
    # visualize final result 
    
    ct_st = np.array(ct_st)
    new_colors = np.zeros((len(my_data), 3))
    no_clusters = len(all_new_centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for c in range(len(all_new_centroids)):
        ind = np.argwhere(ct_st == c)
        new_colors[ind] = cluster_colors[c]

    scene = window.Renderer()
    scene.background((1,1,1))
    spheres = actor.sphere(centers=my_data, colors=new_colors, radii=1)
    scene.add(spheres)
    window.show(scene)
