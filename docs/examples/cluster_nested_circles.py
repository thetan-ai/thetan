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
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
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



all_data = sklearn_data()
my_data = all_data[0]
# TODO: Attension only using a small dataset
my_data = my_data

print(my_data.shape)

class DataSequence:
    def __init__(self, data):

        self.data = data

    def __getitem__(self, i):
        return self.data[i][None, :]

    def __len__(self):
        return self.data.shape[0]


import time
def dist(a, b):
    
    #np.sqrt(np.sum((a-b)**2))
    return np.linalg.norm(a-b)


dataseq = DataSequence(my_data)

vector_feature = IdentityFeature()
metric = EuclideanMetric(feature=vector_feature)

qbx = QuickBundlesX(thresholds=[0.25], metric=metric)

clusters = qbx.cluster(dataseq)

# get clusters of layer 1 (we have only one layer)
total_clusters = clusters.get_clusters(1)
# centroids of to clusters
total_centroids = np.array(total_clusters.centroids)
# reshape data for visualization
data = np.array(dataseq)
data = data.reshape(1500,3)
total_centroids = total_centroids.reshape(total_centroids.shape[0],3)


cluster_no = np.zeros(len(data)) # array that has cluster numbers for each data point

for i in range(len(total_clusters)):
    cluster_no[total_clusters[i]] = i # assign cluster number to the points that belong in that cluster
    # 0th index means 0th point in data
    # index tells the data point number
    # value on that index means cluster number it belongs to
    

viz = True      
if viz : 

    colors = [np.random.rand(3) for streamline in range(len(total_clusters)) ]

    p_colors = [[[0,0,0]]]
    for i in range(len(data)): # give colors to clusters for visualization
        p_colors.append(colors[int(cluster_no[i])])

    p_colors.pop(0)

    # show initial clustered data from running qbx
    ren = window.Renderer()

    ren.background((1,1,1))

    ren.add(actor.sphere(data, colors=p_colors, radii=0.013))
    #ren.add(actor.sphere(total_centroids, colors=(1,0.7,0.2), radii=0.016))

    window.show(ren)

# chaining clusters

'''
cluster_chaining is list of lists
each list on ith index corresponds to the ith data point 
ith list will have all the cluster numbers ith data point belongs to

'''
iteration = 36 # number of iterations you want to run

for ite in range(iteration):
    
    cluster_chaining = [[1]]
    for p in range(len(cluster_no)):
        #initially just assigning it the original cluster numbers we got from qbx
        cluster_chaining.append([cluster_no[p]])
    cluster_chaining.pop(0)

    start_time = time.time()
    theta = 0.15
    
    
    for ck in range(len(total_clusters)): # cluster number

        cc = total_clusters[ck]
        all_points_ck = np.argwhere(cluster_no == cc) #cluster_no[cluster_no == ck]
        remaining_points = np.argwhere(cluster_no != cc) #cluster_no[cluster_no != ck]
        #print(all_points_ck)
        for j in all_points_ck: # j has point number that belongs to cluster ck
            j = j[0]
            pointx = data[j] # point values

            for k in remaining_points: # k has point number that belongs to complement of cluster ck
                k = k[0]
                #print(k[0])
                pointy = data[k] # point value
                #print(pointy)
                if dist(pointx, pointy) <= theta: # if within threshold, add to cluster

                    cluster_chaining[k].append(cc)
                    # cluster_chaining[k] = np.append(cluster_chaining[k], ck)

    print("--- %s seconds ---" % (time.time() - start_time))

    relabled = cluster_chaining.copy()
    # relabel it to the smallest cluster number
    for r in range(len(cluster_chaining)):

        relabled[r] = min(cluster_chaining[r])
        
    cluster_no = relabled.copy()
    total_clusters = list(set(relabled)) # number of clusters left. like cluster 4, 7 9 ,10
        
        
viz = True      
if viz : 
    # visualize final result
    m = int(max(set(relabled)) + 1)

    colors = [np.random.rand(3) for streamline in range(m) ]

    p_colors = [[[0,0,0]]]
    for i in range(len(data)):
        p_colors.append(colors[int(relabled[i])])

    p_colors.pop(0)

    ren.background((1,1,1))
    #ren = window.Renderer()
    ren.add(actor.sphere(data, colors=p_colors, radii=0.013))
    #ren.add(actor.sphere(total_centroids, colors=(1,0.7,0.2), radii=0.01612))

    window.show(ren)

