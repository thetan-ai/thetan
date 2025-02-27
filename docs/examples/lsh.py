import numpy as np

# Generate two similar vectors
dim = 128
x = np.random.randn(dim)
y = x + 0.5 * np.random.randn(dim)  # Slight perturbation

# Normalize vectors (for cosine similarity)
x /= np.linalg.norm(x)
y /= np.linalg.norm(y)

# Generate random hyperplanes
num_hashes = 10
random_planes = np.random.randn(num_hashes, dim)

# Compute binary hashes
hash_x = (x @ random_planes.T) > 0
hash_y = (y @ random_planes.T) > 0

# Compute Hamming similarity
hamming_similarity = np.mean(hash_x == hash_y)

print(f"Hamming similarity: {hamming_similarity:.2f}")
