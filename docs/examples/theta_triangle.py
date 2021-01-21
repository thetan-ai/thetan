import numpy as np
from robotikos.theta import theta_g1

X = np.random.rand(5, 3)
X = np.ones((5, 3)) + 0.01 * X

print(X)

# summs, clusters_sizes, indices = theta_g1(X, theta=1000.0)


# print(X.mean(axis=0))
# print(summs)
# print(clusters_sizes)
# print(summs/clusters_sizes.astype('f8'))


X = np.random.normal(size=(5000, 2))

X[:2500] += np.array([10, 0.])

# from pylab import figure, plot, show #, axis

# figure()
# plot(X[:, 0], X[:, 1], '.')
# # axis('equal')
# show()

summs, clusters_sizes, indices = theta_g1(X, theta=2000.0)

print(summs/clusters_sizes.astype('f8')[:, None])
