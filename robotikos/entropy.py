import numpy as np
from scipy.stats import norm


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def kl_divergence_normed(p, q):
    return 1 - 1 * np.exp(- kl_divergence(p, q))


if __name__ == "__main__":

    x = np.arange(-10, 10, 0.001)
    # add compare two normal distributions with two different means
    p = norm.pdf(x, 0, 2)
    q = norm.pdf(x, 5, 2)

    print('kl divergence', kl_divergence(p, q))
    print('normalized kl', kl_divergence_normed(p, q))