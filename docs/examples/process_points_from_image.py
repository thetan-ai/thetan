from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from thetan.berserker import berserker2_clean as tb

# Set random seed for reproducibility
state = np.random.get_state()
print("Random state before setting seed:", state)

# Load the image and convert to grayscale
image_path = "/home/elef/Downloads/test_4.png"  # Change this path as needed
img = Image.open(image_path).convert("L")  # Convert to grayscale

# Convert to NumPy array and apply threshold to binarize
pixel_values = np.array(img)
binary_img = pixel_values > 50  # Threshold at 20

# Extract (x, y) coordinates of white pixels
white_pixel_coords = np.column_stack(np.where(binary_img.T))  # Transpose so that we get (x, y)


for i in range(100):

    print(f"Iteration {i+1}")
    # Randomly sample 5000 points (without replacement)
    N = min(20000, len(white_pixel_coords))  # Ensure we don't exceed available points
    # np.random.seed(42)
    sample_indices = np.random.choice(len(white_pixel_coords), size=N, replace=False)
    sampled_coords = np.ascontiguousarray(white_pixel_coords[sample_indices], dtype='f8')

    for theta_value in [150]: # range(180, 280, 10):

        centroids, labels = tb(sampled_coords, theta_value)

        # Plot the centroids and labels
        plt.figure(figsize=(20, 12))
        plt.scatter(sampled_coords[:, 0], sampled_coords[:, 1], s=5, c=labels, cmap='tab20', label='Sampled Points')
        plt.scatter(centroids[:, 0], centroids[:, 1], s=50, color='red', label='Centroids')
        plt.gca().invert_yaxis()
        plt.title("TB Clustering with $\\theta$ " + str(theta_value))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        # plt.show()
        plt.savefig(f"tb_clustering_i_{i}.png")


    # Using KMeans for comparison
    # kmeans = KMeans(n_clusters=15, init='random', n_init=1)
    kmeans = KMeans(n_clusters=15)

    kmeans.fit(sampled_coords)
    # Plot KMeans results
    plt.figure(figsize=(20, 12))
    plt.scatter(sampled_coords[:, 0], sampled_coords[:, 1], s=5, c=kmeans.labels_, cmap='tab20', label='Sampled Points')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, color='red', label='KMeans Centroids')
    plt.gca().invert_yaxis()
    plt.title("KMeans Clustering with K" + str(kmeans.n_clusters))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    # plt.show()
    plt.savefig(f"kmeanspp_clustering_i_{i}.png")
