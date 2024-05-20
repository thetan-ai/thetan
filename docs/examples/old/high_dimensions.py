
import numpy as np
from pylab import hist, plot, show

dims = 2
scale = 0.1
size = 10000

mu = np.zeros(dims)

A = np.random.multivariate_normal(mu, scale * np.eye(dims), size=size)

mA = np.mean(A, axis=0)

dists = np.sqrt(np.sum((A - mu)**2, axis=1))

hist(dists, bins=50)

show()

t =1
