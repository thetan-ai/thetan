import numpy as np
from thetan.theta import centroids_from_labels
from thetan.chaining import chainer, chainer2
from thetan.berserker import berserker2
from thetan.viz import plot_centroids_clusters
from sklearn.cluster import DBSCAN, HDBSCAN
from time import time

# fname = "/home/elef/tmp/SpiralData_Clean_Spread.csv"
fname = 'C:/Users/Eleftherios/tmp/SpiralData_Clean_Spread.csv'

spiral_data = np.genfromtxt(fname, delimiter=',')
# spiral_data = np.genfromtxt("C:/Users/Eleftherios/tmp/SpiralData_Clean_Spread.csv", delimiter=',')

# first entry is nan
spiral_data = np.delete(spiral_data, 0, 0)
X = spiral_data.copy()

theta_value = 1.0
epsilon_value = 2.0

times = {}

t = time()
initial_centroids, initial_labels = berserker2(X, theta_value, all_out=False)
duration = time() - t
times['TB'] = duration

plot_centroids_clusters(X, initial_centroids, initial_labels, title='Berserker2', show=True)

t = time()
new_labels = chainer(X, initial_centroids, initial_labels, epsilon_value)
new_labels = np.ascontiguousarray(new_labels, dtype='i8')
duration = time() - t
times['chainer'] = duration

# new_labels are not guaranteed to be 0, 1, 2, 3, ... you can have 0, 5, 8

# indices_arg = np.argsort(new_labels)
# sorted_labels = new_labels[indices_arg]

new_centroids = centroids_from_labels(X, new_labels)
plot_centroids_clusters(X, centroids=new_centroids, labels=new_labels, title='Chainer', show=True)



eps = 0.5 
min_samples = 1

t = time()
model = DBSCAN(eps=eps, min_samples=min_samples)
model.fit(X)

unq = np.unique(model.labels_)
db_nclusters = np.sum(unq>=0)

dbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(db_nclusters)])
dblabels = model.labels_
duration = time() - t
times['dbscan'] = duration


plot_centroids_clusters(X, centroids=dbcentroids, labels=dblabels, title='DBSCAN', show=True)

# TB-HDBSCAN


eps = 2.
min_samples = 1

t = time()
model = DBSCAN(eps=eps, min_samples=min_samples)
model.fit(initial_centroids)

unq = np.unique(model.labels_)
db_nclusters = np.sum(unq>=0)

dbcentroids = np.array([np.mean(initial_centroids[model.labels_==l], axis=0) for l in range(db_nclusters)])
dblabels = model.labels_
duration = time() - t
times['tb-dbscan'] = duration

plot_centroids_clusters(initial_centroids, centroids=dbcentroids, labels=dblabels, title='TB-DBSCAN', show=True)

print(times)

t = 1


