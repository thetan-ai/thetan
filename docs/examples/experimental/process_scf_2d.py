import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation

from thetan import scf_2d


# Generating synthetic data
def _stamp(mask, rs, cs):
    H, W = mask.shape
    rs = np.round(rs).astype(int)
    cs = np.round(cs).astype(int)
    ok = (rs >= 0) & (rs < H) & (cs >= 0) & (cs < W)
    mask[rs[ok], cs[ok]] = 1.0


def make_fragmented_scene(H=170, W=170, gap_frac=0.40, line_gap=(70, 98)):
    mask = np.zeros((H, W), dtype=np.float64)

    cy, cx, R = 78, 88, 46
    n_arcs = 8
    for i in range(n_arcs):
        a0 = 2 * np.pi * i / n_arcs
        a1 = a0 + (2 * np.pi / n_arcs) * (1.0 - gap_frac)
        ang = np.linspace(a0, a1, 80)
        _stamp(mask, cy + R * np.sin(ang), cx + R * np.cos(ang))

    row = 150
    g0, g1 = line_gap
    mask[row, 20:g0] = 1.0
    mask[row, g1:150] = 1.0

    return mask


if __name__ == "__main__":
    mask = make_fragmented_scene()

    C, theta_map = scf_2d(
        mask, n_theta=36, sigma=0.20, tau=12.0, dt=0.8, return_orientation=True
    )

    support_radius = 2
    orig_support = binary_dilation(mask > 0, iterations=support_radius)

    completed = C.copy()
    completed[orig_support] = 0.0
    cmax = completed.max()
    completed_norm = completed / cmax if cmax > 0 else completed

    comp_disp = np.power(completed_norm, 0.5)
    heat = colormaps["inferno"]
    composite = heat(comp_disp)[..., :3]
    composite[comp_disp < 0.12] = 0.0
    composite[mask > 0] = np.array([0.20, 1.0, 0.20])

    norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))

    axes[0, 0].imshow(mask, cmap="gray_r", interpolation="nearest")
    axes[0, 0].set_title("(a) Original edge fragments (input)")

    im = axes[0, 1].imshow(C, cmap="magma", norm=norm, interpolation="bilinear")
    axes[0, 1].set_title("(b) Full SCF field (originals + completion)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(
        completed_norm, cmap="inferno", norm=norm, interpolation="bilinear"
    )
    axes[1, 0].set_title("(c) SCF completion only (input support removed)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(composite, interpolation="bilinear")
    axes[1, 1].set_title("(d) Composite: original vs completed")
    axes[1, 1].legend(
        handles=[
            Patch(color="#33FF33", label="original contour"),
            Patch(color="#F98C0A", label="SCF-completed gap"),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.85,
    )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "2D Stochastic Completion Field - original vs. completed contours",
        fontsize=15,
        fontweight="bold",
    )

    fig.tight_layout()
    plt.show()
