from unittest import signals
import numpy as np
from thetan.hashing import hash_rows_64, hash64, simhash_xxhash_64, simhash_xxhash_tables_64
from time import time
from dipy.io.image import load_nifti



# qname = "/home/elef/Data/HCP_3T/136126/denoised_arr_p2s.nii.gz"
# mname = "/home/elef/Data/HCP_3T/136126/nodif_brain_mask.nii.gz"
# sims_path = "/home/elef/Devel/FORCE_hash/simulated_signals/simulated_data.npz"

qname = "C:/Users/Eleftherios/Data/136126/data.nii.gz"
mname = "C:/Users/Eleftherios/Data/136126/nodif_brain_mask.nii.gz"
sims_path = "C:/Users/Eleftherios/Downloads/simulated_data.npz"  # expects this file in the same directory\


mask, mask_affine = load_nifti(mname)
mask = mask.astype(bool)
print("Mask shape:", mask.shape)


qata, qaffine = load_nifti(qname)
print("Query shape:", qata.shape)

qata_shape = qata.shape

qata = qata * mask[..., np.newaxis]  # Apply mask to each channel

qata_flattened = qata.reshape(-1, qata.shape[-1])
print("Reshaped query shape:", qata_flattened.shape)

qata_flattened[qata_flattened <= 0] = 0

# Crucial normalization step
mask_norm = np.linalg.norm(qata_flattened, axis=1, keepdims=True)

mask_norm[mask_norm == 0] = 1.0

qata_norm = 100 * (qata_flattened / mask_norm)
qata_norm = np.ascontiguousarray(qata_norm)

flat_index = np.ravel_multi_index((72, 110, 72), qata_shape[:-1])
print(qata[72, 110, 72])
print(qata_flattened[flat_index])
print(qata_norm[flat_index])


# Load simulations
data = np.load(sims_path, allow_pickle=False)

X = data["signals"]
labels = data["labels"]

print("Loaded:")
print("  X shape:", X.shape)
print("  labels shape:", labels.shape)

dbg = True

print("Ranges after normalization:")
print(X.min(), X.max())
print(qata_norm.min(), qata_norm.max())



# Hashing measurements & simulations

dbg = True

qatad = qata_norm.astype('f8') / 100.0

# Enable for testing highly structured areas
# qatad = qatad[flat_index:flat_index + 10000].copy()
# Enable for testing background areas
# qatad = qatad[:10000].copy()
print("Subset shape:", qatad.shape)

# Simhash

# Normalize once
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
Xn = np.ascontiguousarray(Xn, dtype='f8')

# Fixed random planes (store them!)
rng = np.random.default_rng(0)
R = rng.standard_normal((32, qatad.shape[1]))

start = time()
h = simhash_xxhash_64(Xn, R)
end = time()
print("Sims Simhashing took:", end - start, "seconds")

shunq, shidx, shcnts = np.unique(h, return_index=True, return_counts=True)
print("Sims Unique simhashes:", len(shunq))

start = time()
q = simhash_xxhash_64(qatad, R)
end = time()
print("Data Simhashing took:", end - start, "seconds")

hunq, hidx, hcnts = np.unique(q, return_index=True, return_counts=True)
print("Data Unique simhashes:", len(hunq))

common = np.intersect1d(shunq, hunq)
print("Common simhashes between sims and data:", len(common))

from scipy.spatial.distance import cosine

# for cs in common:
#     sim_indices = np.where(shunq == cs)[0]
#     data_indices = np.where(h == cs)[0]

#     for si in sim_indices:
#         for di in data_indices:
#             dist = cosine(Xn[si], qatad[di])
#             if dist < 0.2:
#                 print(f"Close match found for simhash {cs}: sim index {si}, data index {di}, cosine distance {dist}")

# start = time()
# h2 = simhash_xxhash_tables_64(qatad, R, L=4)
# end = time()
# print("Simhashing tables took:", end - start, "seconds")

# hunq2, hidx2, hcnts2 = np.unique(h2, return_index=True, return_counts=True)
# print("Unique simhashes:", len(hunq2))


use_thetan = True

if use_thetan:

    from thetan import berserker, sequential, sequential_pexact

    # centroids, labels = berserker(X, n_clusters=1000, random_state=42)

    theta = 0.025

    start = time()
    centroids, labels = sequential(qatad, theta, metric='cosine')
    end = time()
    print("Sequential took:", end - start, "seconds")
    print("Centroids shape:", centroids.shape)

    dbg = True

    theta = 0.025

    start = time()
    centroids, labels = sequential_pexact(qatad, theta, metric='cosine')
    end = time()
    print("Sequential Pexact took:", end - start, "seconds")
    print("Centroids shape:", centroids.shape)

# from berspar2 import berserker_papprox

# theta = 0.025

# start = time()
# centroids, labels = berserker_papprox(qatad, theta, metric='cosine', chunks_per_cpu=2, target_chunk_mb=64)
# end = time()
# print("Berserker Parallel Approx took:", end - start, "seconds")
# print("Centroids shape:", centroids.shape)

dbg = True