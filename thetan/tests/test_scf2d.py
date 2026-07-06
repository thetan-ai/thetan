import numpy as np
import numpy.testing as npt
from scipy.ndimage import gaussian_filter1d, shift

from thetan.berserker import (
    _advect,
    _inject_field,
    _propagate,
    edges_from_mask,
    scf_2d,
)


def ang_diff_modpi(a, b):
    """Smallest absolute difference between two orientations living in [0, pi)."""
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


def inject_ground(points_rc, thetas_pt, weights, shape, thetas_grid):
    H, W = shape
    K = len(thetas_grid)
    dtheta = thetas_grid[1] - thetas_grid[0]
    field = np.zeros((H, W, K))
    for (r, c), th, w in zip(points_rc, thetas_pt, weights):
        for d in (th, th + np.pi):
            k = int(np.round((d % (2 * np.pi)) / dtheta)) % K
            field[r, c, k] += w
    return field


def test_inject_field_hand_computed():
    # K=4 grid: [0, pi/2, pi, 3pi/2], dtheta = pi/2
    thetas = np.arange(4) * (2 * np.pi / 4)
    shape = (5, 6)

    # single point at (2,3), theta=0, weight=1 -> channels 0 and 2
    field = _inject_field([(2, 3)], [0.0], [1.0], shape, thetas)
    expected = np.zeros((5, 6, 4))
    expected[2, 3, 0] = 1.0
    expected[2, 3, 2] = 1.0
    npt.assert_allclose(field, expected)

    # theta = pi/2 -> channels 1 and 3
    field = _inject_field([(2, 3)], [np.pi / 2], [1.0], shape, thetas)
    expected = np.zeros((5, 6, 4))
    expected[2, 3, 1] = 1.0
    expected[2, 3, 3] = 1.0
    npt.assert_allclose(field, expected)


def test_inject_field_weight_accumulation():
    thetas = np.arange(8) * (2 * np.pi / 8)
    shape = (4, 4)
    # two identical points at same pixel/orientation accumulate
    field = _inject_field([(1, 1), (1, 1)], [0.3, 0.3], [2.0, 5.0], shape, thetas)
    assert field[1, 1].sum() == 14.0  # (2+5) into each of 2 channels
    # every point injects into exactly two opposite channels
    assert np.count_nonzero(field[1, 1]) == 2


def test_inject_field_opposite_channels():
    # th and th+pi must land K/2 apart for even K
    K = 36
    thetas = np.arange(K) * (2 * np.pi / K)
    field = _inject_field([(0, 0)], [0.9], [1.0], (2, 2), thetas)
    ks = np.nonzero(field[0, 0])[0]
    assert len(ks) == 2
    assert (ks[1] - ks[0]) % K == K // 2


def test_inject_field_wrap_near_2pi():
    # a direction just under 2*pi should round-wrap back to channel 0
    K = 36
    thetas = np.arange(K) * (2 * np.pi / K)
    field = _inject_field([(0, 0)], [2 * np.pi - 1e-9], [1.0], (1, 1), thetas)
    assert field[0, 0, 0] == 1.0


def test_inject_field_random_vs_ground():
    rng = np.random.default_rng(0)
    K = 36
    thetas = np.arange(K) * (2 * np.pi / K)
    shape = (12, 15)
    n = 40
    pts = [
        (int(rng.integers(0, shape[0])), int(rng.integers(0, shape[1])))
        for _ in range(n)
    ]
    ths = rng.uniform(0, np.pi, n)
    ws = rng.uniform(0.1, 3.0, n)
    actual = _inject_field(pts, ths, ws, shape, thetas)
    expected = inject_ground(pts, ths, ws, shape, thetas)
    npt.assert_allclose(actual, expected)


def advect_ground(field, thetas, dt):
    out = np.empty_like(field)
    for k, th in enumerate(thetas):
        out[:, :, k] = shift(
            field[:, :, k],
            (dt * np.sin(th), dt * np.cos(th)),
            order=1,
            mode="constant",
            cval=0.0,
        )
    return out


def test_advect_impulse_horizontal():
    # theta = 0 -> velocity (sin,cos) = (0,1): impulse moves +1 column
    thetas = np.array([0.0])
    field = np.zeros((7, 7, 1))
    field[3, 3, 0] = 1.0
    out = _advect(field, thetas, dt=1.0)
    expected = np.zeros((7, 7, 1))
    expected[3, 4, 0] = 1.0
    npt.assert_allclose(out, expected, atol=1e-9)


def test_advect_impulse_vertical():
    # theta = pi/2 -> velocity (1,0): impulse moves +1 row
    thetas = np.array([np.pi / 2])
    field = np.zeros((7, 7, 1))
    field[3, 3, 0] = 1.0
    out = _advect(field, thetas, dt=1.0)
    expected = np.zeros((7, 7, 1))
    expected[4, 3, 0] = 1.0
    npt.assert_allclose(out, expected, atol=1e-9)


def test_advect_random_vs_ground():
    rng = np.random.default_rng(1)
    K = 12
    thetas = np.arange(K) * (2 * np.pi / K)
    field = rng.random((10, 11, K))
    out = _advect(field, thetas, dt=0.7)
    expected = advect_ground(field, thetas, 0.7)
    npt.assert_allclose(out, expected)


def test_advect_mass_conserved_interior():
    # A blob far from the border keeps its total mass under linear-interp shift.
    thetas = np.array([0.37])
    field = np.zeros((40, 40, 1))
    field[18:22, 18:22, 0] = 1.0
    out = _advect(field, thetas, dt=0.6)
    npt.assert_allclose(out.sum(), field.sum(), rtol=1e-6)


def propagate_ground(inject, thetas, sigma, tau, dt, n_steps):
    K = len(thetas)
    dtheta = 2 * np.pi / K
    sig_ch = sigma * np.sqrt(dt) / dtheta
    decay = np.exp(-dt / tau)
    P = np.zeros_like(inject)
    for _ in range(n_steps):
        P = P + inject
        P = advect_ground(P, thetas, dt)
        P = gaussian_filter1d(P, sig_ch, axis=2, mode="wrap")
        P = P * decay
    return P


def test_propagate_zero_steps():
    K = 8
    thetas = np.arange(K) * (2 * np.pi / K)
    inject = np.random.default_rng(2).random((6, 6, K))
    P = _propagate(inject, thetas, sigma=0.2, tau=12.0, dt=0.8, n_steps=0)
    npt.assert_allclose(P, np.zeros_like(inject))


def test_propagate_linearity():
    rng = np.random.default_rng(3)
    K = 12
    thetas = np.arange(K) * (2 * np.pi / K)
    inject = rng.random((8, 9, K))
    kw = dict(sigma=0.2, tau=12.0, dt=0.8, n_steps=5)
    P1 = _propagate(inject, thetas, **kw)
    P2 = _propagate(3.5 * inject, thetas, **kw)
    npt.assert_allclose(P2, 3.5 * P1, rtol=1e-9, atol=1e-12)


def test_propagate_vs_ground():
    rng = np.random.default_rng(4)
    K = 36
    thetas = np.arange(K) * (2 * np.pi / K)
    inject = rng.random((10, 10, K))
    kw = dict(sigma=0.2, tau=12.0, dt=0.8, n_steps=8)
    actual = _propagate(inject, thetas, **kw)
    expected = propagate_ground(inject, thetas, **kw)
    npt.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_edges_horizontal_line():
    mask = np.zeros((64, 64))
    mask[32, 12:52] = 1.0
    theta = edges_from_mask(mask)
    interior = theta[32, 20:44]  # avoid endpoints
    # horizontal line runs along columns -> tangent angle 0
    err = ang_diff_modpi(interior, 0.0)
    assert err.mean() < 0.1, err.mean()


def test_edges_vertical_line():
    mask = np.zeros((64, 64))
    mask[12:52, 32] = 1.0
    theta = edges_from_mask(mask)
    interior = theta[20:44, 32]
    # vertical line runs along rows -> tangent angle pi/2
    err = ang_diff_modpi(interior, np.pi / 2)
    assert err.mean() < 0.1, err.mean()


def test_edges_diagonal_line():
    mask = np.zeros((64, 64))
    idx = np.arange(12, 52)
    mask[idx, idx] = 1.0
    theta = edges_from_mask(mask)
    interior = theta[idx[8:-8], idx[8:-8]]
    err = ang_diff_modpi(interior, np.pi / 4)
    assert err.mean() < 0.15, err.mean()


def test_edges_output_range():
    rng = np.random.default_rng(5)
    mask = (rng.random((40, 40)) > 0.7).astype(float)
    theta = edges_from_mask(mask)
    assert np.all(theta >= 0)
    assert np.all(theta < np.pi)


def test_scf_empty_mask():
    mask = np.zeros((20, 25))
    C = scf_2d(mask)
    assert C.shape == (20, 25)
    npt.assert_array_equal(C, 0.0)

    C, theta = scf_2d(mask, return_orientation=True)
    assert C.shape == (20, 25)
    assert theta.shape == (20, 25)
    npt.assert_array_equal(C, 0.0)
    npt.assert_array_equal(theta, 0.0)


def test_scf_shapes_and_range():
    mask = np.zeros((48, 48))
    mask[24, 10:20] = 1.0
    mask[24, 28:38] = 1.0
    C = scf_2d(mask, n_steps=20)
    assert C.shape == mask.shape
    assert np.all(C >= 0.0)
    assert np.all(C <= 1.0 + 1e-12)
    npt.assert_allclose(C.max(), 1.0)


def test_scf_orientation_range():
    mask = np.zeros((48, 48))
    mask[24, 10:38] = 1.0
    C, theta = scf_2d(mask, n_steps=20, return_orientation=True)
    assert theta.shape == mask.shape
    assert np.all(theta >= 0)
    assert np.all(theta < np.pi)


def test_scf_deterministic():
    mask = np.zeros((40, 40))
    mask[20, 8:16] = 1.0
    mask[20, 24:32] = 1.0
    C1 = scf_2d(mask, n_steps=15)
    C2 = scf_2d(mask, n_steps=15)
    npt.assert_array_equal(C1, C2)


def test_scf_bridges_collinear_gap():
    # two collinear horizontal fragments with a gap; the completion field
    # inside the gap (on the line) should exceed an off-line control region.
    mask = np.zeros((60, 60))
    row = 30
    mask[row, 10:25] = 1.0
    mask[row, 35:50] = 1.0
    C = scf_2d(mask, n_steps=60)

    gap_on_line = C[row, 26:34].mean()
    off_line = C[row + 8, 26:34].mean()
    assert gap_on_line > off_line, (gap_on_line, off_line)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("All Tests Passed!")
