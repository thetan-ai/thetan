from time import time

import numpy as np
from dipy.viz import skyline
from scipy.ndimage import binary_dilation

from thetan import scf_3d


def make_sphere_fragment(
    n=52, radius=19.0, cap_lat=1.15, lat_gap=(0.62, 0.14), az_gap=(np.pi, 0.30)
):
    center = n / 2.0
    grid = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    dr, dc, dz = grid[0] - center, grid[1] - center, grid[2] - center

    dist = np.sqrt(dr * dr + dc * dc + dz * dz)
    shell = np.abs(dist - radius) <= 1.0
    lat = np.arccos(np.clip(dz / np.maximum(dist, 1e-9), -1.0, 1.0))
    az = np.mod(np.arctan2(dr, dc), 2.0 * np.pi)

    cap = shell & (lat < cap_lat)

    lat_c, lat_w = lat_gap
    az_c, az_w = az_gap
    gap = cap & ((np.abs(lat - lat_c) < lat_w) | (np.abs(az - az_c) < az_w))
    mask = cap.astype(np.float64)
    mask[gap] = 0.0
    return mask, gap


if __name__ == "__main__":
    n = 52
    radius = 19.0
    mask, gap = make_sphere_fragment(n=n, radius=radius)
    print(
        f"Fragment: {mask.shape} surviving voxels: {int((mask > 0).sum())} | removed gap voxels: {int(gap.sum())}"
    )

    t0 = time()
    completion = scf_3d(mask, n_theta=20, n_phi=10, sigma=0.30, tau=8.0, dt=1.5)

    print(f"3D SCF finished in {time() - t0:.1f}s max={completion.max():.3f}")

    support = binary_dilation(mask > 0, iterations=2)
    completed = completion.copy()
    completed[support] = 0.0
    cmax = completed.max()
    completed_norm = completed / cmax if cmax > 0 else completed
    comp_disp = np.power(completed_norm, 0.5)

    center = n / 2.0
    grid = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
    dist = np.sqrt(
        (grid[0] - center) ** 2 + (grid[1] - center) ** 2 + (grid[2] - center) ** 2
    )
    roi = np.abs(dist - radius) <= 2.5
    gap_thresh = 0.15

    composite = np.zeros_like(mask)
    composite[(comp_disp > gap_thresh) & roi] = 2.0
    composite[mask > 0] = 1.0

    recovered = int(((comp_disp > gap_thresh) & roi & gap).sum())
    print(
        f"Gap reconstructed: {recovered}/{int(gap.sum())} ({100 * recovered / int(gap.sum()):.0f}%) of removed voxels bridged"
    )

    affine = np.eye(4)
    images = [
        (mask, affine, "Sphere Fragment"),
        (completion, affine, "Full SCF field"),
        (completed_norm, affine, "Completion only"),
        (composite, affine, "Composite: 1=Fragment 2=Completed"),
    ]
    skyline(images=images)
