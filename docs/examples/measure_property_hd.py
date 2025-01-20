import numpy as np
import matplotlib.pyplot as plt

# High-dimensional sphere: Distance from center
dimensions = [2, 10, 50, 100]
radii = []
for dim in dimensions:
    points = np.random.randn(10000, dim)  # Generate random points
    distances = np.linalg.norm(points, axis=1) / np.sqrt(dim)  # Normalize distance
    radii.append(distances)

# Plot the distributions
plt.figure(figsize=(10, 6))
for i, dim in enumerate(dimensions):
    plt.hist(radii[i], bins=50, alpha=0.6, label=f'{dim} Dimensions', density=True)
plt.title("Concentration of Measure in High Dimensions")
plt.xlabel("Normalized Distance from Center")
plt.ylabel("Density")
plt.legend()
plt.show()
