from time import time
import numpy as np
from robotikos.theta import theta_g1
from robotikos.fgm import fgn_shuffles, fgn
from robotikos.metric import IdentityFeature
from robotikos.metric import (NormalizedSumSquaredDiffMetric,
                              NormalizedSumAbsDiffMetric)

# summs, clusters_sizes, indices = theta_g1(X, theta=1000.0)


# print(X.mean(axis=0))
# print(summs)
# print(clusters_sizes)
# print(summs/clusters_sizes.astype('f8'))


X = np.random.normal(size=(500000, 2))

X[:250000] += np.array([10, 0.])


# X = X[:10].copy()
# print(X)

from pylab import figure, plot, show , axis

figure()
plot(X[:, 0], X[:, 1], '.')
axis('equal')
#show()

theta = 5.0

t1 = time()
summs, clusters_sizes, centroids, indices = theta_g1(X, theta=theta*theta, metric='ssd')
t2 = time()
tg1_duration = t2 - t1
print('TG1 time', t2-t1)
print('sizes', clusters_sizes)
print('centroids', centroids)
print('size of indices', len(indices))

plot(centroids[:, 0], centroids[:, 1], 'rx')
axis('equal')
#show()


X2 = X.copy()[:, None, :]

metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
t1 = time()
all_levels = fgn(X2, thresholds=[theta*theta], metric=metric)
t2 = time()
fgn_duration = t2 - t1
print('FGN time', fgn_duration)
print('centroids', all_levels[1].centroids)
print('Speedup ', fgn_duration/tg1_duration)
test = 1
test = 2

fgn_centroids = np.asarray(all_levels[1].centroids)
plot(fgn_centroids[:, 0, 0], fgn_centroids[:, 0, 1], 'bo')
axis('equal')
show()
