import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    shift,
    sobel,
)


# SCF 2D function
def edges_from_mask(edge_mask, tensor_sigma=1.5):
    m = gaussian_filter(edge_mask.astype(float), tensor_sigma)
    gy = sobel(m, axis=0)
    gx = sobel(m, axis=1)
    Jxx = gaussian_filter(gx * gx, tensor_sigma)
    Jyy = gaussian_filter(gy * gy, tensor_sigma)
    Jxy = gaussian_filter(gx * gy, tensor_sigma)
    theta_grad = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    theta_tangent = theta_grad + np.pi / 2.0
    return np.mod(theta_tangent, np.pi)


def _inject_field(points_rc, thetas_pt, weights, shape, thetas_grid):
    H, W = shape
    K = len(thetas_grid)
    field = np.zeros((H, W, K), dtype=np.float64)
    dtheta = thetas_grid[1] - thetas_grid[0]
    for (r, c), th, w in zip(points_rc, thetas_pt, weights):
        for direction in (th, th + np.pi):
            k = int(np.round((direction % (2 * np.pi)) / dtheta)) % K
            field[r, c, k] += w

    return field


def _advect(field, thetas, dt):
    out = np.empty_like(field)
    for k, th in enumerate(thetas):
        dr = dt * np.sin(th)
        dc = dt * np.cos(th)
        out[:, :, k] = shift(
            field[:, :, k], (dr, dc), order=1, mode="constant", cval=0.0
        )
    return out


def _propagate(inject, thetas, sigma, tau, dt, n_steps):
    K = len(thetas)
    dtheta = 2 * np.pi / K
    sig_ch = sigma * np.sqrt(dt) / dtheta
    decay = np.exp(-dt / tau)

    P = np.zeros_like(inject)
    for _ in range(n_steps):
        P += inject
        P = _advect(P, thetas, dt)
        P = gaussian_filter1d(P, sig_ch, axis=2, mode="wrap")
        P *= decay
    return P


def scf_2d(
    edge_mask,
    n_theta=36,
    sigma=0.20,
    tau=12.0,
    dt=0.8,
    n_steps=None,
    tensor_sigma=1.5,
    return_orientation=False,
):
    edge_mask = np.asarray(edge_mask)
    H, W = edge_mask.shape
    if n_steps is None:
        n_steps = int(np.ceil(4 * tau / dt))

    thetas = np.arange(n_theta) * (2 * np.pi / n_theta)
    rr, cc = np.nonzero(edge_mask)

    if len(rr) == 0:
        C = np.zeros((H, W))
        return (C, np.zeros((H, W))) if return_orientation else C

    theta_pix = edges_from_mask(edge_mask, tensor_sigma)
    points_rc = list(zip(rr, cc))
    thetas_pt = theta_pix[rr, cc]
    weights = edge_mask[rr, cc].astype(float)

    inject = _inject_field(points_rc, thetas_pt, weights, (H, W), thetas)
    P = _propagate(inject, thetas, sigma, tau, dt, n_steps)

    half = n_theta // 2
    P_opp = np.roll(P, half, axis=2)
    c = P * P_opp

    C = c.sum(axis=2)
    m = C.max()
    if m > 0:
        C /= m

    if return_orientation:
        k_star = np.argmax(c, axis=2)
        theta_map = np.mod(thetas[k_star], np.pi)
        return C, theta_map
    return C


# SCF 3D function
def edges_from_mask_3d(edge_mask, coords, tensor_sigma=1.5):
    m = gaussian_filter(edge_mask.astype(np.float64), tensor_sigma)

    gr = sobel(m, axis=0)
    gc = sobel(m, axis=1)
    gz = sobel(m, axis=2)

    jrr = gaussian_filter(gr * gr, tensor_sigma)
    jcc = gaussian_filter(gc * gc, tensor_sigma)
    jzz = gaussian_filter(gz * gz, tensor_sigma)
    jrc = gaussian_filter(gr * gc, tensor_sigma)
    jrz = gaussian_filter(gr * gz, tensor_sigma)
    jcz = gaussian_filter(gc * gz, tensor_sigma)

    rr, cc, zz = coords
    n = rr.shape[0]

    tensors = np.empty((n, 3, 3), dtype=np.float64)
    tensors[:, 0, 0] = jrr[rr, cc, zz]
    tensors[:, 1, 1] = jcc[rr, cc, zz]
    tensors[:, 2, 2] = jzz[rr, cc, zz]
    tensors[:, 0, 1] = tensors[:, 1, 0] = jrc[rr, cc, zz]
    tensors[:, 0, 2] = tensors[:, 2, 0] = jrz[rr, cc, zz]
    tensors[:, 1, 2] = tensors[:, 2, 1] = jcz[rr, cc, zz]

    _, vecs = np.linalg.eigh(tensors)
    tangent = vecs[:, :, 0]

    vr, vc, vz = tangent[:, 0], tangent[:, 1], tangent[:, 2]

    theta = np.mod(np.arctan2(vr, vc), 2.0 * np.pi)
    phi = np.arccos(np.clip(vz, -1.0, 1.0))

    return theta, phi


def _inject_field_3d(coords, thetas_pt, phis_pt, weights, shape, thetas, phis):
    h, w, d = shape
    n_theta = thetas.shape[0]
    n_phi = phis.shape[0]
    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / n_phi

    field = np.zeros((h, w, d, n_theta, n_phi), dtype=np.float32)

    rr, cc, zz = coords

    kt = np.mod(np.round(thetas_pt / dtheta).astype(np.intp), n_theta)
    kp = np.clip(np.floor(phis_pt / dphi).astype(np.intp), 0, n_phi - 1)

    kt_opp = np.mod(kt + n_theta // 2, n_theta)
    kp_opp = n_phi - 1 - kp

    w32 = np.asarray(weights, dtype=np.float32)
    np.add.at(field, (rr, cc, zz, kt, kp), w32)
    np.add.at(field, (rr, cc, zz, kt_opp, kp_opp), w32)
    return field


def _advect_3d(field, thetas, phis, dt):
    out = np.empty_like(field)
    sin_t, cos_t = np.sin(thetas), np.cos(thetas)
    sin_p, cos_p = np.sin(phis), np.cos(phis)

    for it in range(thetas.shape[0]):
        st, ct = sin_t[it], cos_t[it]
        for ip in range(phis.shape[0]):
            dr = dt * sin_p[ip] * st
            dc = dt * sin_p[ip] * ct
            dz = dt * cos_p[ip]
            out[:, :, :, it, ip] = shift(
                field[:, :, :, it, ip], (dr, dc, dz), order=1, mode="constant", cval=0.0
            )

    return out


def _propagate_3d(inject, thetas, phis, sigma, tau, dt, n_steps):
    n_theta = thetas.shape[0]
    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / phis.shape[0]

    sig_t = sigma * np.sqrt(dt) / dtheta
    sig_p = sigma * np.sqrt(dt) / dphi
    decay = float(np.exp(-dt / tau))

    field = np.zeros_like(inject)
    for _ in range(n_steps):
        field += inject
        field = _advect_3d(field, thetas, phis, dt)
        field = gaussian_filter1d(field, sig_t, axis=3, mode="wrap")
        field = gaussian_filter1d(field, sig_p, axis=4, mode="mirror")
        field *= decay
    return field


def scf_3d(
    edge_mask,
    n_theta=16,
    n_phi=8,
    sigma=0.25,
    tau=12.0,
    dt=1.0,
    n_steps=None,
    tensor_sigma=1.5,
    return_orientation=False,
):
    edge_mask = np.asarray(edge_mask)
    if edge_mask.ndim != 3:
        raise ValueError("edge_mask must be a 3D volume (H, W, D)")
    if n_theta % 2 != 0:
        raise ValueError("n_theta must be even for the antipodal azimuth roll")

    h, w, d = edge_mask.shape
    if n_steps is None:
        n_steps = int(np.ceil(4.0 * tau / dt))

    thetas = np.arange(n_theta) * (2.0 * np.pi / n_theta)
    phis = (np.arange(n_phi) + 0.5) * (np.pi / n_phi)

    coords = np.nonzero(edge_mask)
    if coords[0].size == 0:
        zero = np.zeros((h, w, d), dtype=np.float32)
        if return_orientation:
            return zero, zero.copy(), zero.copy()
        return zero

    theta_pt, phi_pt = edges_from_mask_3d(edge_mask, coords, tensor_sigma)
    weights = edge_mask[coords].astype(np.float32)

    inject = _inject_field_3d(
        coords, theta_pt, phi_pt, weights, (h, w, d), thetas, phis
    )
    field = _propagate_3d(inject, thetas, phis, sigma, tau, dt, n_steps)

    half = n_theta // 2
    c_field = np.zeros((h, w, d), dtype=np.float32)
    if return_orientation:
        best = np.full((h, w, d), -np.inf, dtype=np.float32)
        best_t = np.zeros((h, w, d), dtype=np.intp)
        best_p = np.zeros((h, w, d), dtype=np.intp)

    for ip in range(n_phi):
        opp = np.roll(field[:, :, :, :, n_phi - 1 - ip], half, axis=3)
        prod = field[:, :, :, :, ip] * opp
        c_field += prod.sum(axis=3)
        if return_orientation:
            local_k = np.argmax(prod, axis=3)
            local_max = np.take_along_axis(prod, local_k[..., None], axis=3)[..., 0]
            better = local_max > best
            best = np.where(better, local_max, best)
            best_t = np.where(better, local_k, best_t)
            best_p = np.where(better, ip, best_p)

    peak = float(c_field.max())
    if peak > 0.0:
        c_field /= peak

    if return_orientation:
        theta_map = np.mod(thetas[best_t], np.pi)
        phi_map = phis[best_p]
        return c_field, theta_map, phi_map
    return c_field
