import numpy as np
from scipy.stats import gmean, hmean



def gen_mean(x, p):
    cksz = x.shape[0]
    xp = x ** p
    if p == 0:
        return (1. / cksz) * xp.sum()
    return ((1. / cksz) * xp.sum()) ** (1 / p)

x = np.random.randn(1000) + 10
arithmetic = x.mean()

# geometric = gen_mean(x, 0)

geometric = gmean(x)

hyperbolic = gen_mean(x, -1)
#hyperbolic = hmean(x)

print('Arithmetic {} Geometric {} Hyperbolic {}'.format(arithmetic, geometric, hyperbolic))




tmp = 1