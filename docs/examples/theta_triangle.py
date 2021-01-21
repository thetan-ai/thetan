import numpy as np
from robotikos.theta import theta_g1

X = np.random.rand(5, 3)
X = np.ones((5, 3)) + 0.01 * X

print(X)

summs, clusters_sizes = theta_g1(X, theta=1000.0)


print(X.mean(axis=0))
print(summs)
print(clusters_sizes)
print(summs/clusters_sizes.astype('f8'))
tmp = 1