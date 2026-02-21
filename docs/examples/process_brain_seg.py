import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, uniform_filter
from scipy.linalg import logm
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import horizon
import time
from thetan import berserker as tb

def load_image(path):
    data, affine = load_nifti(path)
    return data, affine 

def compute_gradients(volume, sigma=1.0):
    smoothed = gaussian_filter(volume, sigma=sigma)
    grad = np.stack(np.gradient(smoothed), axis=-1)
    grad_mag = np.linalg.norm(grad, axis=-1)
    return grad, grad_mag

def horizon_support(affine, *args):
    images = []
    for arg in args:

        _data = arg[0]
        _data_name = arg[1]

        images.append((_data, affine, _data_name))
        # print(_data_name)
    return images

def normalized_intensity(volume):
    mask = volume > 0
    if mask.sum() == 0:
        return volume.astype(np.float64)
    p1, p99 = np.percentile(volume[mask], [1, 99])
    out = (volume.astype(np.float64) - p1) / (p99 - p1 + 1e-12)
    return np.clip(out, 0.0, 1.0)

def zscore_columns(features):
    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True) + 1e-12
    return (features - mu) / sd

def build_spatial_coords(shape):
    grids = np.meshgrid(np.linspace(0,1, shape[0]), np.linspace(0,1, shape[1]), np.linspace(0,1, shape[2]), indexing='ij')
    return np.stack(grids, axis=-1)

def structure_tensor_features(grad, sigma_st=2.0, eps=1e-6):
    X, Y, Z, _ = grad.shape
    J = np.zeros((X, Y, Z, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(i, 3):
            outer_ij = grad[..., i] * grad[..., j]
            smoothed = gaussian_filter(outer_ij, sigma=sigma_st)
            J[..., i, j] = smoothed
            if i != j:
                J[..., j, i] = smoothed
    
    J[..., 0,0] += eps
    J[..., 1,1] += eps
    J[..., 2,2] += eps
    
    eigs = np.linalg.eigvalsh(J.reshape(-1, 3, 3))
    eigs = np.maximum(eigs, eps)
    log_eigs = np.log(eigs)
    log_eigs = log_eigs[:, ::-1].reshape(X, Y, Z, 3)

    return log_eigs

# Patch Covariance
def patch_covariance_features(volume, grad, patch_radius=2, eps=1e-6, use_log_matrix=False):
    X, Y, Z = volume.shape
    r = patch_radius

    local_feat = np.concatenate([volume[..., None], grad], axis=-1)
    d = local_feat.shape[-1]

    patch_size = 2 * r + 1
    mean_f = np.zeros((X, Y, Z, d), dtype=np.float64)
    for k in range(d):
        mean_f[..., k] = uniform_filter(local_feat[..., k], size=patch_size, mode='reflect')
    
    mean_ff = np.zeros((X, Y, Z, d, d), dtype=np.float64)
    for i in range(d):
        for j in range(i, d):
            product = local_feat[..., i] * local_feat[..., j]
            filtered = uniform_filter(product, size=patch_size, mode='reflect')
            mean_ff[..., i, j] = filtered
            if i != j:
                mean_ff[..., j, i] = filtered
    
    cov = mean_ff - np.einsum('...i,...j->...ij', mean_f, mean_f)

    for i in range(d):
        cov[..., i, i] += eps

    cov_flat = cov.reshape(-1, d, d)
    N = cov_flat.shape[0]

    if use_log_matrix:
        n_upper = d * (d + 1) // 2
        feats_flat = np.zeros((N, n_upper), dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(cov_flat)
        eigvals = np.maximum(eigvals, eps)
        log_eigvals = np.log(eigvals)
        log_cov = np.einsum('...ij,...j,...kj->...ik', eigvecs, log_eigvals, eigvecs)
        idx = 0
        for i in range(d):
            for j in range(i,d):
                scale = 1.0 if i==j else np.sqrt(2.0)
                feats_flat[:, idx] = log_cov[:, i, j] * scale
                idx += 1
            feats = feats_flat.reshape(X, Y, Z, n_upper)
    else:
        eigvals = np.linalg.eigvalsh(cov_flat)
        eigvals = np.maximum(eigvals, eps)
        log_eigs = np.log(eigvals)[:, ::-1]
        feats = log_eigs.reshape(X, Y, Z, d)
    
    return feats

def build_mri_features(volume, mask=None, spd_mode='structure_tensor',gradient_sigma=1.0, structure_tensor_sigma=2.0, patch_radius=2, use_log_matrix=False, spatial_weight=1.0, normalize=True, eps=1e-6):
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f'Expected 3D Volume, got shape {volume.shape}')
    
    X, Y, Z = volume.shape
    print(f'[FeatureBuilder] Volume shape: ({X}, {Y}, {Z})')

    # Normalize Intensity
    I_norm = normalized_intensity(volume)
    print('[FeatureBuilder] Intensity normalized to [0,1]')

    # Gradients
    grad, grad_mag = compute_gradients(I_norm, sigma=gradient_sigma)
    print('[FeatureBuilder] Gradients computed with sigma={}'.format(gradient_sigma))

    # Spatial Coordinates
    coords = build_spatial_coords((X,Y,Z)) * spatial_weight

    # SPD Features
    if spd_mode == "none":
        spd_feats = None
        print('[FeatureBuilder] SPD mode: none')
    elif spd_mode == 'structure_tensor':
        spd_feats = structure_tensor_features(grad, sigma_st = structure_tensor_sigma, eps = eps)
        print('[FeatureBuilder] Structure tensor features computed with sigma={}'.format(structure_tensor_sigma))
    elif spd_mode in ('patch_covariance', 'patch_covariance_logm'):
        use_logm = (spd_mode == 'patch_covariance_logm')
        spd_feats = patch_covariance_features(I_norm, grad, patch_radius=patch_radius, eps=eps, use_log_matrix=use_logm)
        d_spd = spd_feats.shape[-1]
        print(f'[FeatureBuilder] Patch covariance features computed with radius={patch_radius}, log_matrix={use_logm}, d_spd={d_spd}')
    else:
        raise ValueError(f'Unknown spd_mode: {spd_mode}')


    channels = [I_norm[..., None], grad_mag[..., None]]
    if spd_feats is not None:
        channels.append(spd_feats)
    channels.append(coords)

    feat_volume = np.concatenate(channels, axis=-1)
    D = feat_volume.shape[-1]
    print(f'[FeatureBuilder] Total channels: {D}')
    
    if mask is None:
        mask = volume > 0
        print(f'[FeatureBuilder] Auto-mask: volume > 0')
    
    mask = np.asarray(mask, dtype=bool)
    idx = np.argwhere(mask)
    features = feat_volume[mask]
    print(f'[FeatureBuilder] Masked voxels: {features.shape[0]:,} / {X*Y*Z:,} total ({100*features.shape[0]/(X*Y*Z):.1f}%)')

    if normalize:
        features = zscore_columns(features)
        print('[FeatureBuilder] Features z-score normalized')
    
    return features, idx

def labels_to_volume(labels, indices, shape):
    label_vol = np.full(shape, fill_value=-1, dtype=int)
    label_vol[indices[:, 0], indices[:, 1], indices[:, 2]] = labels
    return label_vol

def extract_largest_cluster_mask(data_labels, indices, shape):
    unique, counts = np.unique(data_labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]

    mask = np.zeros(shape, dtype=np.uint8)
    largest_voxels = indices[data_labels == largest_label]
    mask[largest_voxels[:, 0], largest_voxels[:, 1], largest_voxels[:, 2]] = 1

    return mask, largest_label

if __name__ == "__main__":
    input_path = r'/Users/prajwal/Desktop/thetan_seg/IXI002-Guys-0828-T1.nii.gz'
    
    spd_mode = 'structure_tensor'
    # spd_mode = 'patch_covariance'
    # spd_mode = 'patch_covariance_logm'
    spatial_weight = 1.0
    gradient_sigma = 1.0
    st_sigma = 2.0
    theta_value = 2.5

    print(f'Loading t1')
    vol, affine = load_nifti(input_path)

    if vol.ndim == 4:
        vol = vol[..., 0]

    print("=" * 60)
    print("MRI Feature Builder + Thetan Berserker Pipeline")
    print("=" * 60)
    print(f"Volume shape    : {vol.shape}")
    print(f"Voxel dtype     : {vol.dtype}")
    print(f"Intensity range : [{vol.min():.1f}, {vol.max():.1f}]")
    print(f"Nonzero voxels  : {(vol > 0).sum():,} / {vol.size:,}")

    nonzero_vals = vol[vol > 0]
    threshold = np.percentile(nonzero_vals, 5)
    mask = vol > threshold 
    print(f"Mask threshold  : {threshold:.1f} (5th percentile of nonzero)")
    print(f"Mask voxels     : {mask.sum():,} ({100 * mask.sum() / vol.size:.1f}%)")

    # ---- Step 1: Build feature space ----
    print(f"\n{'─' * 60}")
    print(f"  Step 1: Building features (spd_mode='{spd_mode}')")
    print(f"{'─' * 60}")

    t0 = time.time()
    features, indices = build_mri_features(
        vol,
        mask=mask,
        spd_mode=spd_mode,
        gradient_sigma=gradient_sigma,
        structure_tensor_sigma=st_sigma,
        spatial_weight=spatial_weight,
    )
    t_feat = time.time() - t0

    print(f"\nFeature matrix  : {features.shape}  "
          f"({features.nbytes / 1e6:.1f} MB)")
    print(f"Time            : {t_feat:.1f}s")
    assert np.all(np.isfinite(features)), "ERROR: NaN or Inf in features!"
    print("✓ Features clean (no NaN/Inf)")

    # ---- Step 2: Cluster with Thetan Berserker ----
    print(f"\n{'─' * 60}")
    print(f"  Step 2: Thetan Berserker (θ = {theta_value})")
    print(f"{'─' * 60}")

    t0 = time.time()
    centroids, data_labels = tb(features, theta_value)
    t_tb = time.time() - t0

    n_clusters = len(centroids)
    print(f"\nClusters found  : {n_clusters}")
    print(f"Time            : {t_tb:.1f}s")
    print(f"Label range     : [{data_labels.min()}, {data_labels.max()}]")

    # ---- Step 3: Map labels back to volume ----
    print(f"\n{'─' * 60}")
    print(f"  Step 3: Reconstructing label volume")
    print(f"{'─' * 60}")

    label_vol = labels_to_volume(data_labels, indices, vol.shape)
    assert label_vol.shape == vol.shape
    print(f"Label volume shape: {label_vol.shape}")
    print(f"Background voxels (label=-1): {(label_vol == -1).sum():,}")

    brain_mask, largest_label = extract_largest_cluster_mask(
        data_labels, indices, vol.shape
    )

    unique, counts = np.unique(data_labels, return_counts=True)
    largest_count = counts[unique == largest_label][0]
    print(f"Largest cluster : label {largest_label}")
    print(f"Voxel count     : {largest_count:,} / {len(data_labels):,} "
          f"({100 * largest_count / len(data_labels):.1f}%)")
    print(f"Mask volume     : {brain_mask.sum():,} voxels")

    # Apply mask to T1 to get skull-stripped volume
    brain_vol = vol * brain_mask

    # ---- Visualize ----
    images = horizon_support(
        affine,
        (vol, "t1"),
        (label_vol, "berserker_labels"),
        (brain_mask, "brain_mask"),
        (brain_vol, "brain_extracted"),
    )
    horizon(images=images, bg_color=(1, 0.5, 0.5))