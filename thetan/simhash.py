from unittest import signals
import numpy as np
from thetan.hashing import hash_rows_64, hash64, simhash_xxhash_64
from time import time
from dipy.io.image import load_nifti



qname = "/home/elef/Data/HCP_3T/136126/denoised_arr_p2s.nii.gz"
mname = "/home/elef/Data/HCP_3T/136126/nodif_brain_mask.nii.gz"

# qname = "C:/Users/Eleftherios/Data/136126/data.nii.gz"
# mname = "C:/Users/Eleftherios/Data/136126/nodif_brain_mask.nii.gz"

sims_path = "C:/Users/Eleftherios/Downloads/simulated_data.npz"  # expects this file in the same directory\
sims_path = "/home/elef/Devel/FORCE_hash/simulated_signals/simulated_data.npz"

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


# Hashing simulations

divider = 15

# start = time()
# Xh = np.ascontiguousarray(np.round(np.asarray(X) / divider, 0), dtype=np.uint64)
# row_Xhashes = hash_rows_64(Xh)
# end = time()
# print("Hashing sims took:", end - start, "seconds")

# start = time()
# unq, idx, cnts = np.unique(row_Xhashes, return_index=True, return_counts=True)
# end = time()
# print("Unique hashes:", len(unq))
# print("Counting unique hashes took:", end - start, "seconds")


# Hashing measurements

# start = time()
# qh = np.ascontiguousarray(np.round(np.asarray(qata_norm) / divider, 0), dtype=np.uint64)
# row_qhashes = hash_rows_64(qh)
# end = time()
# print("Hashing measurements took:", end - start, "seconds")

# start = time()
# qunq, qidx, qcnts = np.unique(row_qhashes, return_index=True, return_counts=True)
# end = time()
# print("Unique hashes:", len(qunq))
# print("Counting unique hashes took:", end - start, "seconds")


# common = np.intersect1d(qunq, unq)
# print(common.size, "common hashes between sims and measurements")

# for u in unq:
#     if query == u:
#         print("Match!")
#         print(u)
#         # Here do cosine for idxs

dbg = True

qatad = qata_norm.astype('f8')
# Enable for testing highly structured areas
# qatad = qatad[flat_index:flat_index + 10000].copy()
# Enable for testing background areas
# qatad = qatad[:10000].copy()
print("Subset shape:", qatad.shape)

# Simhash

# Normalize once
# Xn = X / np.linalg.norm(X, axis=1, keepdims=True)

# Fixed random planes (store them!)
rng = np.random.default_rng(0)
R = rng.standard_normal((128, qatad.shape[1]))


start = time()
h = simhash_xxhash_64(qatad, R)
end = time()
print("Simhashing took:", end - start, "seconds")

hunq, hidx, hcnts = np.unique(h, return_index=True, return_counts=True)
print("Unique simhashes:", len(hunq))



use_thetan = False

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