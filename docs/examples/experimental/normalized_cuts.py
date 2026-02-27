import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.ndimage import gaussian_filter
from skimage import io, color, transform
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def compute_weight_matrix(image, r=5, sigma_I = 0.1, sigma_X = 4.0, features ='brightness'):
    """
    Construct the sparse weight matrix for image graphs.
    """
    if image.ndim == 3 and features == 'brightness':
        image = color.rgb2gray(image)
    
    H, W_img = image.shape[:2]
    N = H * W_img

    F = extract_features(image, features)
    rows, cols, vals = [], [], []

    offsets = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dy == 0 and dx == 0:
                continue
            if dy * dy + dx * dx <= r * r:
                offsets.append((dy, dx))
    
    for y in range(H):
        for x in range(W_img):
            i = y * W_img + x
            fi = F[y,x]

            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W_img:
                    j = ny * W_img + nx

                    if j<=i:
                        continue

                    fj = F[ny, nx]

                    feat_dist = np.sum((fi - fj) ** 2)
                    spat_dist = dy * dy + dx * dx

                    w = np.exp(-feat_dist / sigma_I) * np.exp(-spat_dist / sigma_X)

                    if w > 1e-7:
                        rows.append(i)
                        cols.append(j)
                        vals.append(w)
    
    rows, cols, vals = np.array(rows), np.array(cols), np.array(vals)
    
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    W = W + W.T
    return W

def extract_features(image, features='brightness'):
    """
    Extract per pixel feature vectors.
    """
    if features == 'brightness':
        if image.ndim == 3:
            F = color.rgb2gray(image)
        else:
            F = image.copy()
        return F[..., np.newaxis]
    
    elif features == 'color':
        if image.ndim == 2:
            raise ValueError('Color features require a color image')
        hsv = color.rgb2hsv(image)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        F = np.stack([v, v * s * np.sin(2 * np.pi * h), v * s * np.cos(2 * np.pi * h)], axis=-1)
        return F
    
    elif features == 'texture':
        if image.ndim == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image.copy()
        
        features_list = []
        for sigma in [1, 2, 4]:
            gx = gaussian_filter(gray, sigma=sigma, order=[0,1])
            gy = gaussian_filter(gray, sigma=sigma, order=[1,0])
            features_list.extend([np.abs(gx), np.abs(gy)])

        F = np.stack(features_list, axis=-1)
        return F
    
    else:
        raise ValueError(f'Unknown feature type: {features}')

def solve_ncut_eigensystem(W, num_eigenvectors=10):
    """
    Solve the generalized eigenvalue system: (D-W)y = λDy
    """
    N = W.shape[0]

    d = np.array(W.sum(axis=1)).flatten()
    d = np.maximum(d, 1e-10)

    D = sparse.diags(d)
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(d))

    L = D - W

    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    L_norm = (L_norm + L_norm.T) / 2.0

    k = min(num_eigenvectors, N-2)
    eigenvalues, eigenvectors_z = eigsh(L_norm, k=k, which='SM', tol=1e-6)

    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors_z = eigenvectors_z[:, idx]

    eigenvectors_y = D_inv_sqrt @ eigenvectors_z

    return eigenvalues, eigenvectors_y


def bipartition_from_eigenvector(W, eigenvector, num_splits=20):
    """
    Find the best splitting point for the second eigenvector to minimize Ncut(A,B)
    """

    d = np.array(W.sum(axis=1)).flatten()

    v_min, v_max = eigenvector.min(), eigenvector.max()
    split_points = np.linspace(v_min, v_max, num_splits + 2)[1:-1]

    split_points = np.concatenate([split_points, [0.0, np.median(eigenvector)]])

    best_ncut = np.inf
    best_partition = None

    for sp in split_points:
        mask_A = eigenvector > sp
        ncut_val = compute_ncut(W, d, mask_A)

        if ncut_val < best_ncut:
            best_ncut = ncut_val
            best_partition = mask_A
        
    return best_partition, best_ncut

def compute_ncut(W, d, mask_A):
    """
    Compute Ncut(A, B) = cut(A, B) / assoc(A, V) + cut(A, B) / assoc(B, V)
    """

    mask_B = ~mask_A

    if mask_A.sum() == 0 or mask_B.sum() == 0:
        return np.inf
    
    assoc_A_V = d[mask_A].sum()
    assoc_B_V = d[mask_B].sum()

    if assoc_A_V < 1e-10 or assoc_B_V < 1e-10:
        return np.inf
    
    idx_A = np.where(mask_A)[0]
    idx_B = np.where(mask_B)[0]

    W_sub = W[idx_A][:, idx_B]
    cut_val = W_sub.sum()

    ncut_val = cut_val / assoc_A_V + cut_val / assoc_B_V

    return ncut_val

def check_eigenvector_stability(eigenvector, threshold=0.06):
    """
    Check if an eigenvector is a good discrete indicator or a smooth continous function.
    """

    n_bins = max(10, len(eigenvector) // 50)
    hist, _ = np.histogram(eigenvector, bins=n_bins)
    hist = hist[hist > 0]

    if len(hist) < 2:
        return True

    ratio = hist.min() / hist.max()

    return ratio < threshold

def recursive_ncut(W, indices=None, ncut_threshold=0.2, min_segment_size=50, num_eigenvectors=6,num_splits=20, stability_threshold=0.06, max_depth=10, depth=0):
    N = W.shape[0]

    if indices is None:
        indices = np.arange(N)
    
    if N < 2 * min_segment_size or depth >= max_depth:
        return [indices]
    
    k = min(num_eigenvectors, N - 2)
    if k < 2:
        return [indices]
    
    try:
        eigenvalues, eigenvectors = solve_ncut_eigensystem(W, num_eigenvectors=k)
    except Exception:
        return [indices]
    
    best_partition = None
    best_ncut = np.inf

    for i in range(1, k):
        ev = eigenvectors[:, i]

        if not check_eigenvector_stability(ev, stability_threshold):
            continue

        partition, ncut_bval = bipartition_from_eigenvector(W, ev, num_splits)
        
        if ncut_bval < best_ncut:
            best_ncut = ncut_bval
            best_partition = partition
        
    if best_partition is None or best_ncut > ncut_threshold:
        return [indices]

    idx_A = np.where(best_partition)[0]
    idx_B = np.where(~best_partition)[0]
    
    if len(idx_A) < min_segment_size or len(idx_B) < min_segment_size:
        return [indices]
    
    W_A = W[idx_A][:, idx_A]
    W_B = W[idx_B][:, idx_B]

    segments_A = recursive_ncut(W_A, indices[idx_A], ncut_threshold, min_segment_size, num_eigenvectors, num_splits, stability_threshold, max_depth, depth + 1)
    segments_B = recursive_ncut(W_B, indices[idx_B], ncut_threshold, min_segment_size, num_eigenvectors, num_splits,stability_threshold, max_depth, depth + 1)

    return segments_A + segments_B

def kway_ncut(W, k=5, num_eigenvectors=10):
    """
    Simultaneous k-way partitioning using multiple eigenvectors.
    """
    n_eig = min(num_eigenvectors, W.shape[0] - 2)
    eigenvalues, eigenvectors = solve_ncut_eigensystem(W, num_eigenvectors=n_eig)

    embedding = eigenvectors[:, 1:n_eig]

    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embedding = embedding / norms

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return labels

def normalized_cut_segmentation(image, method='recursive', max_size=100, r=5, sigma_I = 0.1, sigma_X = 4.0, features='brightness', ncut_threshold=0.2, min_segment_size=20, k=5, num_eigenvectors=10):
    """
    Full Normalized cut image segmentation pipeline.

    * method : 'recursive' or 'kway'
    * features : 'brightness', 'color' or 'texture'
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    
    H, W_img = img.shape[:2]
    scale = max_size / max(H, W_img)
    if scale < 1.0:
        new_H, new_W = int(H * scale), int(W_img * scale)
        img = transform.resize(img, (new_H, new_W), anti_aliasing=True)
    else:
        new_H, new_W = H, W_img
    
    print(f'Image Size: {new_H} x {new_W} ({new_H * new_W} pixels)')
    print(f'Building weight matrix (r={r}, sigma_I={sigma_I}, sigma_X={sigma_X})')

    W = compute_weight_matrix(img, r=r, sigma_I=sigma_I, sigma_X=sigma_X, features=features)
    print(f'Weight matrix: {W.nnz} nonzero entries ({W.nnz / (new_H * new_W):.4f} avg connections per pixel)')

    N = new_H * new_W
    labels = np.zeros(N, dtype=int)

    if method == 'recursive':
        print(f'Running recursive 2-way Ncut (threshold={ncut_threshold})')
        segments = recursive_ncut(W, ncut_threshold=ncut_threshold, min_segment_size=min_segment_size, num_eigenvectors=num_eigenvectors)
        for seg_id, seg_indices in enumerate(segments):
            labels[seg_indices] = seg_id
        print(f'Found {len(segments)} segments')
    
    elif method == 'kway':
        print(f'Running K-way Ncut (k={k})')
        labels = kway_ncut(W, k=k, num_eigenvectors=num_eigenvectors)
        print(f'Found {len(np.unique(labels))} segments')
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    labels = labels.reshape(new_H, new_W)
    return labels, img

def visualize_segmentation(image, labels, save_path=None):
    n_segments = len(np.unique(labels))
    H, W = labels.shape

    fig, axes = plt.subplots(2,3, figsize=(18,12))
    fig.suptitle('Normalized Cuts Segmentation', fontsize=16, fontweight='bold')

    if image.ndim == 2:
        axes[0,0].imshow(image, cmap='gray')
    else:
        axes[0,0].imshow(image)
    axes[0,0].set_title('Original Image', fontsize=14)
    axes[0,0].axis('off')

    cmap = plt.cm.get_cmap('tab20', n_segments)
    axes[0,1].imshow(labels, cmap=cmap, interpolation='nearest')
    axes[0,1].set_title(f'Segmentation ({n_segments} segments)', fontsize=14)
    axes[0,1].axis('off')

    if image.ndim == 2:
        overlay = np.stack([image] * 3, axis=-1)
    else:
        overlay = image.copy()
    
    boundary = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(labels, dy, axis=0), dx, axis=1)
        boundary |= (labels != shifted)
    
    overlay_with_bounds = overlay.copy()
    overlay_with_bounds[boundary] = [1, 0, 0]
    
    axes[0,2].imshow(overlay_with_bounds)
    axes[0,2].set_title('Segmentation Overlay', fontsize=14)
    axes[0,2].axis('off')

    unique_labels = np.unique(labels)
    for i in range(3):
        ax = axes[1, i]
        if i < len(unique_labels):
            mask = (labels == unique_labels[i])
            if image.ndim == 2:
                seg_vis = np.where(mask, image, 0.0)
                ax.imshow(seg_vis, cmap='gray')
            else:
                seg_vis = np.where(mask[..., None], image, 0.0)
                ax.imshow(seg_vis)
            ax.set_title(f'Segment {unique_labels[i]}', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()

    plt.close()

def visualize_eigenvectors(image, W, save_path=None, num_eigenvectors=9):
    H, W_img  = image.shape[:2]

    eigenvalues, eigenvectors = solve_ncut_eigensystem(W, num_eigenvectors=num_eigenvectors)

    fig, axes = plt.subplots(3, 3, figsize=(15,15))
    fig.suptitle('Eigenvectors of Normalized Cuts', fontsize=16)

    for i in range(min(9, eigenvectors.shape[1])):
        ax = axes[i // 3, i % 3]
        ev = eigenvectors[:, i].reshape(H, W_img)
        ax.imshow(ev, cmap='RdBu_r')
        ax.set_title(f'Eigenvector {i+1} (λ={eigenvalues[i]:.4f})')
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved eigenvector visualization to {save_path}')
    else:
        plt.show()
    
    plt.close()


def main(image_path, output_path=None, method='kway',k=5, max_size=100, r=5, sigma_I=0.1, sigma_X=4.0, features='brightness', ncut_threshold=0.2, min_segment_size=20, num_eigenvectors=10, show_eigenvectors=False):
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = f"{base}_ncut.png"
    
    image = io.imread(image_path)
    

    labels, resized_img = normalized_cut_segmentation(image, method=method, max_size=max_size, r=r, sigma_I=sigma_I, sigma_X=sigma_X, features=features, ncut_threshold=ncut_threshold, min_segment_size=min_segment_size, k=k, num_eigenvectors=num_eigenvectors)

    visualize_segmentation(resized_img, labels, save_path=output_path)
    print(f'Saved segmentation to {output_path}')

    if show_eigenvectors:
        eig_path = output_path.replace('.png', '_eigenvectors.png')
        W = compute_weight_matrix(resized_img, r=r, sigma_I=sigma_I, sigma_X=sigma_X, features=features)
        visualize_eigenvectors(resized_img, W, save_path=eig_path, num_eigenvectors = min(9, num_eigenvectors))
        print(f'Saved eigenvector visualization to {eig_path}')
    
    return labels, resized_img

if __name__ == '__main__':
    main('/Users/prajwal/Downloads/mountain.jpg', k = 4, method='kway', features='color', show_eigenvectors=True)
