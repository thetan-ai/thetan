import numpy as np
import numpy.testing as npt
from scipy.ndimage import gaussian_filter1d, shift

from thetan.berserker import (
    _advect_3d,
    _inject_field_3d,
    _propagate_3d,
    edges_from_mask_3d,
    scf_3d,
)


def sph_to_vec(theta, phi):
    """(theta, phi) -> unit direction in (r, c, z), matching the code's convention."""
    return np.stack(
        [np.sin(phi) * np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi)],
        axis=-1,
    )


def axis_alignment(theta, phi, direction):
    """|v . direction| per point; ~1 when the recovered axis matches (sign-agnostic)."""
    v = sph_to_vec(np.asarray(theta), np.asarray(phi))
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    return np.abs(v @ direction)


def inject_ground_3d(coords, thetas_pt, phis_pt, weights, shape, thetas, phis):
    h, w, d = shape
    n_theta, n_phi = thetas.shape[0], phis.shape[0]
    dtheta, dphi = 2.0 * np.pi / n_theta, np.pi / n_phi
    field = np.zeros((h, w, d, n_theta, n_phi), dtype=np.float64)
    rr, cc, zz = coords
    for i in range(len(rr)):
        kt = int(round(thetas_pt[i] / dtheta)) % n_theta
        kp = int(np.clip(np.floor(phis_pt[i] / dphi), 0, n_phi - 1))
        field[rr[i], cc[i], zz[i], kt, kp] += weights[i]
        field[rr[i], cc[i], zz[i], (kt + n_theta // 2) % n_theta, n_phi - 1 - kp] += (
            weights[i]
        )
    return field


def test_inject3d_hand_computed():
    n_theta, n_phi = 4, 4
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    coords = (np.array([1]), np.array([2]), np.array([3]))

    # theta=0 -> kt=0, kt_opp=2 ; phi=0.1 -> kp=0, kp_opp=3
    field = _inject_field_3d(
        coords,
        np.array([0.0]),
        np.array([0.1]),
        np.array([1.0]),
        (5, 6, 7),
        thetas,
        phis,
    )
    expected = np.zeros((5, 6, 7, n_theta, n_phi), dtype=np.float32)
    expected[1, 2, 3, 0, 0] = 1.0
    expected[1, 2, 3, 2, 3] = 1.0
    npt.assert_allclose(field, expected)


def test_inject3d_phi_clip_at_pole():
    # phi = pi lands on floor == n_phi and must clip to n_phi - 1
    n_theta, n_phi = 4, 8
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    field = _inject_field_3d(
        (np.array([0]), np.array([0]), np.array([0])),
        np.array([np.pi / 2]),
        np.array([np.pi]),
        np.array([1.0]),
        (1, 1, 1),
        thetas,
        phis,
    )
    kp = np.nonzero(field[0, 0, 0].sum(axis=0))[0]
    assert set(kp.tolist()) == {0, n_phi - 1}  # kp clipped to 7, kp_opp = 0


def test_inject3d_two_opposite_bins_per_point():
    n_theta, n_phi = 16, 8
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    field = _inject_field_3d(
        (np.array([0]), np.array([0]), np.array([0])),
        np.array([1.3]),
        np.array([0.7]),
        np.array([2.0]),
        (1, 1, 1),
        thetas,
        phis,
    )
    assert np.count_nonzero(field[0, 0, 0]) == 2
    assert field[0, 0, 0].sum() == 4.0  # weight 2.0 into each of two bins


def test_inject3d_weight_accumulation():
    n_theta, n_phi = 8, 4
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    coords = (np.array([1, 1]), np.array([1, 1]), np.array([1, 1]))
    field = _inject_field_3d(
        coords,
        np.array([0.3, 0.3]),
        np.array([0.5, 0.5]),
        np.array([2.0, 5.0]),
        (3, 3, 3),
        thetas,
        phis,
    )
    assert field[1, 1, 1].sum() == 14.0  # (2+5) into each of two bins


def test_inject3d_random_vs_ground():
    rng = np.random.default_rng(0)
    n_theta, n_phi = 16, 8
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    shape = (8, 9, 7)
    n = 60
    rr = rng.integers(0, shape[0], n)
    cc = rng.integers(0, shape[1], n)
    zz = rng.integers(0, shape[2], n)
    ths = rng.uniform(0, 2 * np.pi, n)
    phs = rng.uniform(0, np.pi, n)
    ws = rng.uniform(0.1, 3.0, n)
    actual = _inject_field_3d((rr, cc, zz), ths, phs, ws, shape, thetas, phis)
    expected = inject_ground_3d((rr, cc, zz), ths, phs, ws, shape, thetas, phis)
    npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def advect_ground_3d(field, thetas, phis, dt):
    out = np.empty_like(field)
    for it in range(thetas.shape[0]):
        for ip in range(phis.shape[0]):
            dr = dt * np.sin(phis[ip]) * np.sin(thetas[it])
            dc = dt * np.sin(phis[ip]) * np.cos(thetas[it])
            dz = dt * np.cos(phis[ip])
            out[:, :, :, it, ip] = shift(
                field[:, :, :, it, ip], (dr, dc, dz), order=1, mode="constant", cval=0.0
            )
    return out


def _impulse_advect(theta, phi, dt):
    field = np.zeros((7, 7, 7, 1, 1), dtype=np.float32)
    field[3, 3, 3, 0, 0] = 1.0
    out = _advect_3d(field, np.array([theta]), np.array([phi]), dt)
    return out[:, :, :, 0, 0]


def test_advect3d_impulse_z():
    # phi=0 -> pure +z motion
    out = _impulse_advect(0.0, 0.0, dt=1.0)
    exp = np.zeros((7, 7, 7))
    exp[3, 3, 4] = 1.0
    npt.assert_allclose(out, exp, atol=1e-6)


def test_advect3d_impulse_c():
    # phi=pi/2, theta=0 -> pure +c motion
    out = _impulse_advect(0.0, np.pi / 2, dt=1.0)
    exp = np.zeros((7, 7, 7))
    exp[3, 4, 3] = 1.0
    npt.assert_allclose(out, exp, atol=1e-6)


def test_advect3d_impulse_r():
    # phi=pi/2, theta=pi/2 -> pure +r motion
    out = _impulse_advect(np.pi / 2, np.pi / 2, dt=1.0)
    exp = np.zeros((7, 7, 7))
    exp[4, 3, 3] = 1.0
    npt.assert_allclose(out, exp, atol=1e-6)


def test_advect3d_random_vs_ground():
    rng = np.random.default_rng(1)
    n_theta, n_phi = 6, 4
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    field = rng.random((8, 9, 7, n_theta, n_phi)).astype(np.float32)
    out = _advect_3d(field, thetas, phis, dt=0.7)
    expected = advect_ground_3d(field, thetas, phis, 0.7)
    npt.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def test_advect3d_mass_conserved_interior():
    thetas = np.array([0.6])
    phis = np.array([1.0])
    field = np.zeros((20, 20, 20, 1, 1), dtype=np.float64)
    field[8:12, 8:12, 8:12, 0, 0] = 1.0
    out = _advect_3d(field, thetas, phis, dt=0.5)
    npt.assert_allclose(out.sum(), field.sum(), rtol=1e-6)


def propagate_ground_3d(inject, thetas, phis, sigma, tau, dt, n_steps):
    n_theta = thetas.shape[0]
    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / phis.shape[0]
    sig_t = sigma * np.sqrt(dt) / dtheta
    sig_p = sigma * np.sqrt(dt) / dphi
    decay = float(np.exp(-dt / tau))
    field = np.zeros_like(inject)
    for _ in range(n_steps):
        field = field + inject
        field = advect_ground_3d(field, thetas, phis, dt)
        field = gaussian_filter1d(field, sig_t, axis=3, mode="wrap")
        field = gaussian_filter1d(field, sig_p, axis=4, mode="mirror")
        field = field * decay
    return field


def _small_grids(n_theta=8, n_phi=4):
    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    return thetas, phis


def test_propagate3d_zero_steps():
    thetas, phis = _small_grids()
    inject = np.random.default_rng(2).random((4, 4, 4, 8, 4)).astype(np.float32)
    P = _propagate_3d(inject, thetas, phis, sigma=0.25, tau=12.0, dt=1.0, n_steps=0)
    npt.assert_allclose(P, np.zeros_like(inject))


def test_propagate3d_linearity():
    rng = np.random.default_rng(3)
    thetas, phis = _small_grids()
    inject = rng.random((5, 5, 5, 8, 4)).astype(np.float32)
    kw = dict(sigma=0.25, tau=12.0, dt=1.0, n_steps=4)
    P1 = _propagate_3d(inject, thetas, phis, **kw)
    P2 = _propagate_3d(3.5 * inject, thetas, phis, **kw)
    npt.assert_allclose(P2, 3.5 * P1, rtol=1e-4, atol=1e-6)


def test_propagate3d_vs_ground():
    rng = np.random.default_rng(4)
    thetas, phis = _small_grids()
    inject = rng.random((5, 6, 4, 8, 4)).astype(np.float32)
    kw = dict(sigma=0.25, tau=12.0, dt=1.0, n_steps=5)
    actual = _propagate_3d(inject, thetas, phis, **kw)
    expected = propagate_ground_3d(inject, thetas, phis, **kw)
    npt.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


def _line_interior(axis, length=40, lo=8, hi=8, grid=64):
    """Build a straight line of edge voxels along an axis; return mask + interior coords."""
    mask = np.zeros((grid, grid, grid))
    mid = grid // 2
    idx = np.arange(lo, length + lo)
    coord = [np.full(idx.shape, mid), np.full(idx.shape, mid), np.full(idx.shape, mid)]
    coord[axis] = idx
    mask[coord[0], coord[1], coord[2]] = 1.0
    keep = slice(hi, len(idx) - hi)
    interior = (coord[0][keep], coord[1][keep], coord[2][keep])
    return mask, interior


def test_edges3d_line_along_r():
    mask, coords = _line_interior(axis=0)
    theta, phi = edges_from_mask_3d(mask, coords)
    align = axis_alignment(theta, phi, [1, 0, 0])
    assert align.mean() > 0.98, align.mean()


def test_edges3d_line_along_c():
    mask, coords = _line_interior(axis=1)
    theta, phi = edges_from_mask_3d(mask, coords)
    align = axis_alignment(theta, phi, [0, 1, 0])
    assert align.mean() > 0.98, align.mean()


def test_edges3d_line_along_z():
    mask, coords = _line_interior(axis=2)
    theta, phi = edges_from_mask_3d(mask, coords)
    align = axis_alignment(theta, phi, [0, 0, 1])
    assert align.mean() > 0.98, align.mean()


def test_edges3d_diagonal_in_rc_plane():
    grid = 64
    mask = np.zeros((grid, grid, grid))
    idx = np.arange(12, 52)
    z = grid // 2
    mask[idx, idx, z] = 1.0
    keep = slice(8, len(idx) - 8)
    coords = (idx[keep], idx[keep], np.full(idx[keep].shape, z))
    theta, phi = edges_from_mask_3d(mask, coords)
    align = axis_alignment(theta, phi, [1, 1, 0])
    assert align.mean() > 0.95, align.mean()


def test_edges3d_output_ranges():
    rng = np.random.default_rng(5)
    mask = (rng.random((24, 24, 24)) > 0.7).astype(float)
    coords = np.nonzero(mask)
    theta, phi = edges_from_mask_3d(mask, coords)
    assert np.all(theta >= 0) and np.all(theta < 2 * np.pi)
    assert np.all(phi >= 0) and np.all(phi <= np.pi)


def test_scf3d_rejects_non_3d():
    npt.assert_raises(ValueError, scf_3d, np.zeros((10, 10)))


def test_scf3d_rejects_odd_n_theta():
    npt.assert_raises(ValueError, scf_3d, np.zeros((8, 8, 8)), n_theta=15)


def test_scf3d_empty_mask():
    mask = np.zeros((12, 10, 8))
    C = scf_3d(mask)
    assert C.shape == (12, 10, 8)
    assert C.dtype == np.float32
    npt.assert_array_equal(C, 0.0)

    C, th, ph = scf_3d(mask, return_orientation=True)
    for arr in (C, th, ph):
        assert arr.shape == (12, 10, 8)
        npt.assert_array_equal(arr, 0.0)


def test_scf3d_shapes_and_range():
    mask = np.zeros((24, 24, 24))
    mask[12, 6:11, 12] = 1.0
    mask[12, 14:19, 12] = 1.0
    C = scf_3d(mask, n_theta=8, n_phi=4, n_steps=15)
    assert C.shape == mask.shape
    assert C.dtype == np.float32
    assert np.all(C >= 0.0) and np.all(C <= 1.0 + 1e-6)
    npt.assert_allclose(C.max(), 1.0, rtol=1e-5)


def test_scf3d_orientation_ranges():
    mask = np.zeros((24, 24, 24))
    mask[12, 6:19, 12] = 1.0
    n_phi = 4
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)
    C, theta_map, phi_map = scf_3d(
        mask, n_theta=8, n_phi=n_phi, n_steps=15, return_orientation=True
    )
    assert np.all(theta_map >= 0) and np.all(theta_map < np.pi)
    assert np.all(phi_map >= phis.min() - 1e-6) and np.all(phi_map <= phis.max() + 1e-6)


def test_scf3d_deterministic():
    mask = np.zeros((20, 20, 20))
    mask[10, 4:9, 10] = 1.0
    mask[10, 12:17, 10] = 1.0
    C1 = scf_3d(mask, n_theta=8, n_phi=4, n_steps=12)
    C2 = scf_3d(mask, n_theta=8, n_phi=4, n_steps=12)
    npt.assert_array_equal(C1, C2)


def test_scf3d_bridges_collinear_gap():
    # two collinear fragments along the c-axis with a gap between them;
    # the completion field in the gap (on the line) should beat an off-line control.
    mask = np.zeros((32, 40, 32))
    r = z = 16
    mask[r, 8:18, z] = 1.0
    mask[r, 22:32, z] = 1.0
    C = scf_3d(mask, n_theta=8, n_phi=4, n_steps=40)
    gap_on_line = C[r, 18:22, z].mean()
    off_line = C[r + 6, 18:22, z].mean()
    assert gap_on_line > off_line, (gap_on_line, off_line)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("All Tests Passed!")
