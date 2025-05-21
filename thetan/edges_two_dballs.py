import numpy as np


def sample_uniform_ball(center, radius, n_samples, dim):
    """Samples n_samples points uniformly from the interior of a d-dimensional ball.

    Parameters:
        center (array-like): Center of the ball (length d)
        radius (float): Radius of the ball
        n_samples (int): Number of points to sample
        dim (int): Dimensionality of the space

    Returns:
        np.ndarray: (n_samples, dim) array of sampled points
    """
    center = np.asarray(center)
    assert center.shape[0] == dim, "Center must be a vector of length 'dim'."

    # Step 1: Sample directions from standard normal and normalize
    directions = np.random.normal(size=(n_samples, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Step 2: Sample radii with correct volume distribution
    radii = np.random.rand(n_samples) ** (1.0 / dim)

    # Step 3: Scale and translate
    samples = center + radius * (radii[:, None] * directions)

    return samples




def sample_edges(d, n, dim):

    # Centers
    c1 = np.array([0,] * dim)
    c2 = np.array([d,] + [0,] * (dim - 1))

    # Sample points
    B1 = sample_uniform_ball(c1, r, n, dim)
    B2 = sample_uniform_ball(c2, r, n, dim)

    # Count edges with distance < theta
    count = 0
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(B1[i] - B2[j])
            if dist < theta:
                count += 1
    return count


# Parameters
theta = 1.0
r = theta / 2
d = 1.5  # center distance
n = 1000

dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
counts = []
for dim in dims:
    count = sample_edges(d, n, dim)
    counts.append(count/n**2)
    print(f"P of edges with distance < theta in {dim} dimensions: {count/n**2}")


t = 0

# fit an exponential decay function to the data
from scipy.optimize import curve_fit
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

popt, pcov = curve_fit(exp_decay, dims, counts)



import matplotlib.pyplot as plt
plt.plot(dims, counts, marker='o')
plt.plot(dims, exp_decay(np.array(dims), *popt), color='red')
plt.xlabel('Dimensions')
plt.ylabel('Probability of edges with distance < theta')
plt.title('Probability of edges with distance < theta vs Dimensions ' + str(popt))
plt.xticks(dims)
plt.grid()
plt.show()


t = 1


