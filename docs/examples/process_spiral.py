import numpy as np
from thetan.theta import centroids_from_labels, assign
from thetan.chaining import chainer, chainer2
from thetan.berserker import berserker2
from thetan.viz import plot_centroids_clusters
from sklearn.cluster import DBSCAN, HDBSCAN
from time import time

# fname = "/home/elef/tmp/SpiralData_Clean_Spread.csv"
fname = 'C:/Users/Eleftherios/Data/spiral/SpiralData_Clean_Spread.csv'
dname_save = 'C:/Users/Eleftherios/Data/tb_output_spiral_v2/' #_small_set/'

spiral_data = np.genfromtxt(fname, delimiter=',')


# spiral_data = np.genfromtxt("C:/Users/Eleftherios/tmp/SpiralData_Clean_Spread.csv", delimiter=',')

# first entry is nan
spiral_data = np.delete(spiral_data, 0, 0)
X = spiral_data.copy()
X = np.ascontiguousarray(np.tile(X, (50, 1)))

print('>>> # points ', X.shape[0])

show = False

print('\n >> Starting TB ...')

theta_value = 1.0
epsilon_value = 2.0

times = {}

t = time()
initial_centroids, initial_labels = berserker2(X, theta_value, all_out=False, verbose=False)
duration = time() - t
times['TB'] = duration

plot_centroids_clusters(X, initial_centroids, initial_labels, title='TB', show=show, savefn=dname_save + 'TB.png')

t = time()
new_labels = chainer(X, initial_centroids, initial_labels, epsilon_value)
new_labels = np.ascontiguousarray(new_labels, dtype='i8')
duration = time() - t
times['chainer'] = duration

# new_labels are not guaranteed to be 0, 1, 2, 3, ... you can have 0, 5, 8

# indices_arg = np.argsort(new_labels)
# sorted_labels = new_labels[indices_arg]

new_centroids = centroids_from_labels(X, new_labels)
plot_centroids_clusters(X, centroids=None, labels=new_labels, title='Chainer', show=show, savefn=dname_save + 'Chainer.png')


print('\n >> Starting DBSCAN ...')


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


plot_centroids_clusters(X, centroids=None, labels=dblabels, title='DBSCAN', show=show, savefn=dname_save + 'DBSCAN.png')


print('\n >> Starting TB-DBSCAN ...')

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

plot_centroids_clusters(initial_centroids, centroids=None, labels=dblabels, title='TB-DBSCAN (init)', show=show, savefn=dname_save + 'TB-DBSCAN (init).png')


final_labels = initial_labels.copy()

for i, l in enumerate(initial_labels):
    final_labels[i] = dblabels[l]

plot_centroids_clusters(X, None, labels=final_labels, title='TB-DBSCAN', show=show, savefn=dname_save + 'TB-DBSCAN.png')



print('\n >> Starting HDBSCAN ...')
t = time()
model = HDBSCAN(min_samples=min_samples)
model.fit(X)

unq = np.unique(model.labels_)
hdb_nclusters = np.sum(unq>=0)

hdbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(hdb_nclusters)])
hdblabels = model.labels_
duration = time() - t

times['hdbscan'] = duration
print('Duration', duration)

plot_centroids_clusters(X, None, labels=hdblabels, title='HDBSCAN', show=show, savefn=dname_save + 'HDBSCAN.png')


print('\n >> Starting TB-HDBSCAN ...')

min_samples = 1

t = time()
model = HDBSCAN(min_samples=min_samples)
model.fit(initial_centroids)

unq = np.unique(model.labels_)
hdb_nclusters = np.sum(unq>=0)

hdbcentroids = np.array([np.mean(initial_centroids[model.labels_==l], axis=0) for l in range(hdb_nclusters)])
hdblabels = model.labels_


plot_centroids_clusters(initial_centroids, centroids=None, labels=hdblabels, title='TB-DBSCAN (init)', show=show, savefn=dname_save + 'TB-DBSCAN (init).png')

final_labels = initial_labels.copy()

for i, l in enumerate(initial_labels):
    final_labels[i] = hdblabels[l]

duration = time() - t
times['tb-hdbscan'] = duration

plot_centroids_clusters(X, None, labels=final_labels, title='TB-HDBSCAN', show=show, savefn=dname_save + 'TB-HDBSCAN.png')


print(times)

t = 1



