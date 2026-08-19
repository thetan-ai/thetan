import numpy as np
from scipy.fft import irfft, rfft
from scipy.ndimage import gaussian_filter, sobel

from thetan.complete import (
    add_sparse_injection2d,
    add_sparse_injection3d,
    advect_2d_into,
    advect_3d_into,
    angular_filter2d_into,
    angular_filter_phi_into,
    angular_filter_theta_into,
    antipodal_reduce2d,
    antipodal_reduce3d,
)


def edges_from_mask(edge_mask, tensor_sigma=1.5):
    m = gaussian_filter(edge_mask.astype(float), tensor_sigma)
    gy = sobel(m, axis = 0)
    gx = sobel(m, axis = 1)
    Jxx = gaussian_filter(gx * gx, tensor_sigma)
    Jyy = gaussian_filter(gy * gy, tensor_sigma)
    Jxy = gaussian_filter(gx * gy, tensor_sigma)
    theta_grad = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    theta_tangent = theta_grad + np.pi / 2.0
    return np.mod(theta_tangent, np.pi)

def make_advection_plan_2d(shape, thetas, dt):
    height, width = shape
    channels = thetas.shape[0]
    shifts = np.ascontiguousarray(np.column_stack((dt * np.sin(thetas), dt * np.cos(thetas))),
        dtype=np.float64)

    source_delta = -shifts
    lower = np.floor(source_delta).astype(np.intp)
    fraction = source_delta - lower
    offsets = np.empty((channels, 4), dtype=np.intp)
    interpolation_weights = np.empty((channels, 4), dtype=np.float64)

    corner = 0
    for br in (0,1):
        wr = fraction[:, 0] if br else 1.0 - fraction[:, 0]
        for bc in (0,1):
            wc = fraction[:, 1] if bc else 1.0 - fraction[:, 1]
            spatial_offset = (lower[:, 0] + br) * width + lower[:, 1] + bc
            offsets[:, corner] = spatial_offset * channels
            interpolation_weights[:, corner] = wr * wc
            corner += 1

    margins = np.ceil(np.max(np.abs(shifts), axis = 0)).astype(np.intp) + 1
    bounds = []
    for size, margin in zip(shape, margins):
        start = min(int(margin), size)
        stop = max(start, size - int(margin))
        bounds.extend((start, stop))

    return (
        np.ascontiguousarray(offsets),
        np.ascontiguousarray(interpolation_weights),
        shifts,
        np.ascontiguousarray(bounds, dtype=np.intp),
    )

def make_angular_plan_2d(n_theta, sigma_channels):
    kernel, offsets = gaussian_kernel_1d(sigma_channels)
    lookup = np.mod(
        np.arange(n_theta, dtype=np.intp)[:, None] + offsets[None, :],
        n_theta,
    )
    return kernel, np.ascontiguousarray(lookup, dtype=np.intp)

def prepare_sparse_injection_2d(coords, thetas_pt, weights, shape, n_theta):
    _, width = shape
    dtheta = 2.0 * np.pi / n_theta
    rr, cc = coords

    spatial = np.ascontiguousarray(rr, dtype=np.intp).copy()
    spatial *= width
    spatial += cc
    channel0 = np.ascontiguousarray(
        np.mod(np.round(thetas_pt / dtheta).astype(np.intp), n_theta),
        dtype = np.intp,
    )
    channel1 = np.ascontiguousarray(
        np.mod(
            np.round(np.mod(thetas_pt + np.pi, 2.0 * np.pi) / dtheta).astype(
                np.intp
            ),
            n_theta,
        ),
        dtype=np.intp,
    )

    return (
        spatial,
        channel0,
        channel1,
        np.ascontiguousarray(weights, dtype=np.float64),
    )

def advect_2d(field, thetas, dt, num_threads=0):
    if num_threads < 0:
        raise ValueError('num_threads must be non-negative')

    field = np.ascontiguousarray(field, dtype=np.float64)
    thetas = np.ascontiguousarray(thetas, dtype=np.float64)
    output = np.empty_like(field)
    offsets, interpolation_weights, shifts, bounds = make_advection_plan_2d(
        field.shape[:2], thetas, dt
    )
    advect_2d_into(
        field.reshape(-1),
        output.reshape(-1),
        offsets,
        interpolation_weights,
        shifts,
        bounds,
        field.shape[0],
        field.shape[1],
        num_threads,
    )

    return output

def angular_gaussian_fft_response(n_channels, sigma_channels, dtype):
    kernel, offsets = gaussian_kernel_1d(sigma_channels)
    circular_kernel = np.zeros(n_channels, dtype=dtype)
    np.add.at(circular_kernel, np.mod(offsets, n_channels), kernel)
    return rfft(circular_kernel)

def propagate_2d(
    shape,
    sparse_injection,
    thetas,
    sigma,
    tau,
    dt,
    n_steps,
    num_threads=0,
):
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    height, width = shape
    n_theta = thetas.shape[0]
    n_spatial = height * width
    current = np.zeros((height, width, n_theta), dtype=np.float64)
    if n_steps <= 0:
        return current
    scratch = np.empty_like(current)

    offsets, interpolation_weights, shifts, bounds = make_advection_plan_2d(shape, thetas, dt)
    dtheta = 2.0 * np.pi / n_theta
    sigma_channels = sigma * np.sqrt(dt) / dtheta
    decay = float(np.exp(-dt / tau))
    spatial, channel0, channel1, values = sparse_injection

    use_fft = int(4.0 * sigma_channels + 0.5) >= 16
    if use_fft:
        angular_response = angular_gaussian_fft_response(
            n_theta, sigma_channels, current.dtype
        )
        angular_response *= decay
        fft_workers = num_threads if num_threads > 0 else - 1
    else:
        angular_kernel, angular_lookup = make_angular_plan_2d(
            n_theta, sigma_channels
        )

    for _ in range(n_steps):
        current_flat = current.reshape(-1)
        scratch_flat = scratch.reshape(-1)
        add_sparse_injection2d(
            current_flat,
            spatial,
            channel0,
            channel1,
            values,
            n_theta,
            num_threads,
        )
        advect_2d_into(
            current_flat,
            scratch_flat,
            offsets,
            interpolation_weights,
            shifts,
            bounds,
            height,
            width,
            num_threads,
        )
        if use_fft:
            spectrum = rfft(
                scratch, axis=2, workers=fft_workers, overwrite_x = True
            )
            spectrum *= angular_response
            current = irfft(
                spectrum,
                n=n_theta,
                axis=2,
                workers=fft_workers,
                overwrite_x=True,
            )
            np.maximum(current, 0.0, out=current)
        else:
            angular_filter2d_into(
                scratch_flat,
                current_flat,
                angular_kernel,
                angular_lookup,
                n_spatial,
                n_theta,
                decay,
                num_threads,
            )
    return current

def scf_2d(
    edge_mask,
    n_theta=36,
    sigma=0.20,
    tau=12.0,
    dt=0.8,
    n_steps=None,
    tensor_sigma=1.5,
    return_orientation=False,
    num_threads=0,
):
    edge_mask = np.asarray(edge_mask)
    if edge_mask.ndim != 2:
        raise ValueError("edge_mask must be a 2D image (H,W)")
    if not isinstance(n_theta, (int, np.integer)) or n_theta <= 0 or n_theta % 2:
        raise ValueError("n_theta must be a positive even integer")
    if sigma < 0 or tau <= 0 or dt <= 0:
        raise ValueError("sigma must be non-negative and tau/dt must be positive")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    H, W = edge_mask.shape
    if n_steps is None:
        n_steps = int(np.ceil(4 * tau / dt))
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")

    thetas = np.ascontiguousarray(
        np.arange(n_theta) * (2.0 * np.pi / n_theta), dtype=np.float64
    )
    rr, cc = np.nonzero(edge_mask)

    if len(rr) == 0:
       zero = np.zeros((H, W), dtype=np.float64)
       return (zero, zero.copy()) if return_orientation else zero

    theta_pix = edges_from_mask(edge_mask, tensor_sigma)
    thetas_pt = theta_pix[rr, cc]
    weights = edge_mask[rr, cc].astype(np.float64)

    sparse_injection = prepare_sparse_injection_2d(
        (rr,cc), thetas_pt, weights, (H, W), n_theta
    )

    del rr, cc, theta_pix, thetas_pt, weights
    field = propagate_2d(
        (H,W),
        sparse_injection,
        thetas,
        sigma,
        tau,
        dt,
        n_steps,
        num_threads=num_threads,
    )

    reduced = antipodal_reduce2d(
        field.reshape(-1),
        H * W,
        n_theta,
        return_orientation,
        num_threads,
    )

    if return_orientation:
        completion_flat, best_theta_flat = reduced
        completion = completion_flat.reshape(H, W)
        best_theta = best_theta_flat.reshape(H, W)
    else:
        completion = reduced.reshape(H, W)

    maximum = float(completion.max())
    if maximum > 0.0:
        completion /= maximum

    if return_orientation:
        theta_map = np.mod(thetas[best_theta], np.pi)
        return completion, theta_map
    return completion


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


def make_advection_plan_3d(shape, thetas, phis, dt):
    h, w, d = shape
    n_theta = thetas.shape[0]
    n_phi = phis.shape[0]
    channels = n_theta * n_phi

    sin_t = np.sin(thetas)[:, None]
    cos_t = np.cos(thetas)[:, None]
    sin_p = np.sin(phis)[None, :]
    cos_p = np.cos(phis)[None, :]
    dr = (dt * sin_t * sin_p).reshape(-1)
    dc = (dt * cos_t * sin_p).reshape(-1)
    dz = np.broadcast_to(dt * cos_p, (n_theta, n_phi)).reshape(-1)
    shifts = np.ascontiguousarray(np.column_stack((dr, dc, dz)), dtype=np.float64)

    source_delta = -shifts
    lower = np.floor(source_delta).astype(np.intp)
    fraction = source_delta - lower
    offsets = np.empty((channels, 8), dtype=np.intp)
    interpolation_weights = np.empty((channels, 8), dtype=np.float64)

    corner = 0
    for br in (0, 1):
        wr = fraction[:, 0] if br else 1.0 - fraction[:, 0]
        for bc in (0, 1):
            wc = fraction[:, 1] if bc else 1.0 - fraction[:, 1]
            for bz in (0, 1):
                wz = fraction[:, 2] if bz else 1.0 - fraction[:, 2]
                spatial_offset = (
                    (lower[:, 0] + br) * w * d
                    + (lower[:, 1] + bc) * d
                    + lower[:, 2]
                    + bz
                )
                offsets[:, corner] = spatial_offset * channels
                interpolation_weights[:, corner] = wr * wc * wz
                corner += 1

    margins = np.ceil(np.max(np.abs(shifts), axis=0)).astype(np.intp) + 1
    bounds = []
    for size, margin in zip(shape, margins):
        start = min(int(margin), size)
        stop = max(start, size - int(margin))
        bounds.extend((start, stop))

    return (
        np.ascontiguousarray(offsets),
        np.ascontiguousarray(interpolation_weights),
        shifts,
        np.ascontiguousarray(bounds, dtype=np.intp),
    )

def gaussian_kernel_1d(sigma_channels):
    if sigma_channels < 0:
        raise ValueError("Gaussian sigma must be non-negative")
    radius = int(4.0 * sigma_channels + 0.5)
    if radius == 0:
        return np.ones(1, dtype=np.float64), np.zeros(1, dtype=np.intp)
    offsets = np.arange(-radius, radius + 1, dtype=np.intp)
    kernel = np.exp(-0.5 * (offsets / sigma_channels) ** 2)
    kernel /= kernel.sum()
    return np.ascontiguousarray(kernel), offsets

def make_angular_plan_3d(n_theta, n_phi, sig_t, sig_p):
    theta_kernel, theta_offsets = gaussian_kernel_1d(sig_t)
    phi_kernel, phi_offsets = gaussian_kernel_1d(sig_p)

    theta_lookup = np.mod(
        np.arange(n_theta, dtype=np.intp)[:, None] + theta_offsets[None, :],
        n_theta,
    )

    if n_phi == 1:
        phi_lookup = np.zeros((1, phi_offsets.size), dtype=np.intp)
    else:
        period = 2 * n_phi - 2
        reflected = np.mod(
            np.arange(n_phi, dtype=np.intp)[:, None] + phi_offsets[None, :],
            period,
        )
        phi_lookup = np.where(reflected < n_phi, reflected, period - reflected)

    return (
        theta_kernel,
        np.ascontiguousarray(theta_lookup, dtype=np.intp),
        phi_kernel,
        np.ascontiguousarray(phi_lookup, dtype=np.intp),
    )

def prepare_sparse_injection_3d(
    coords, thetas_pt, phis_pt, weights, shape, n_theta, n_phi
):
    h, w, d = shape
    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / n_phi
    rr, cc, zz = coords

    spatial = np.ascontiguousarray(rr, dtype=np.intp).copy()
    spatial *= w
    spatial += cc
    spatial *= d
    spatial += zz

    kt = np.mod(np.round(thetas_pt / dtheta).astype(np.intp), n_theta)
    kp = np.clip(np.floor(phis_pt / dphi).astype(np.intp), 0, n_phi - 1)
    channel0 = np.ascontiguousarray(kt * n_phi + kp, dtype=np.intp)
    channel1 = np.ascontiguousarray(
        np.mod(kt + n_theta // 2, n_theta) * n_phi + (n_phi - 1 - kp),
        dtype=np.intp,
    )
    return (
        spatial,
        channel0,
        channel1,
        np.ascontiguousarray(weights, dtype=np.float32),
    )

def advect_3d(field, thetas, phis, dt, num_threads=0):
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    field = np.ascontiguousarray(field, dtype=np.float32)
    thetas = np.ascontiguousarray(thetas, dtype=np.float64)
    phis = np.ascontiguousarray(phis, dtype=np.float64)
    output = np.empty_like(field)
    offsets, interpolation_weights, shifts, bounds = make_advection_plan_3d(
        field.shape[:3], thetas, phis, dt
    )
    advect_3d_into(
        field.reshape(-1),
        output.reshape(-1),
        offsets,
        interpolation_weights,
        shifts,
        bounds,
        field.shape[0],
        field.shape[1],
        field.shape[2],
        num_threads,
    )
    return output

def propagate_3d(
    shape,
    sparse_injection,
    thetas,
    phis,
    sigma,
    tau,
    dt,
    n_steps,
    num_threads=0,
):

    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    h, w, d = shape
    n_theta = thetas.shape[0]
    n_phi = phis.shape[0]
    channels = n_theta * n_phi
    n_spatial = h * w * d
    field_shape = (h, w, d, n_theta, n_phi)
    current = np.zeros(field_shape, dtype=np.float32)
    if n_steps <= 0:
        return current
    scratch = np.empty_like(current)

    offsets, interpolation_weights, shifts, bounds = make_advection_plan_3d(
        shape, thetas, phis, dt
    )
    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / n_phi
    sig_t = sigma * np.sqrt(dt) / dtheta
    sig_p = sigma * np.sqrt(dt) / dphi
    decay = float(np.exp(-dt / tau))
    theta_kernel, theta_lookup, phi_kernel, phi_lookup = make_angular_plan_3d(
        n_theta, n_phi, sig_t, sig_p
    )
    spatial, channel0, channel1, values = sparse_injection

    for _ in range(n_steps):
        current_flat = current.reshape(-1)
        scratch_flat = scratch.reshape(-1)
        add_sparse_injection3d(
            current_flat,
            spatial,
            channel0,
            channel1,
            values,
            channels,
            num_threads,
        )
        advect_3d_into(
            current_flat,
            scratch_flat,
            offsets,
            interpolation_weights,
            shifts,
            bounds,
            h,
            w,
            d,
            num_threads,
        )
        angular_filter_theta_into(
            scratch_flat,
            current_flat,
            theta_kernel,
            theta_lookup,
            n_spatial,
            n_theta,
            n_phi,
            num_threads,
        )
        angular_filter_phi_into(
            current_flat,
            scratch_flat,
            phi_kernel,
            phi_lookup,
            n_spatial,
            n_theta,
            n_phi,
            decay,
            num_threads,
        )
        current, scratch = scratch, current

    return current


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
    num_threads=0,
):
    edge_mask = np.asarray(edge_mask)
    if edge_mask.ndim != 3:
        raise ValueError("edge_mask must be a 3D volume (H, W, D)")
    if not isinstance(n_theta, (int, np.integer)) or n_theta <= 0 or n_theta % 2:
        raise ValueError("n_theta must be a positive even integer")
    if not isinstance(n_phi, (int, np.integer)) or n_phi <= 0:
        raise ValueError("n_phi must be a positive integer")
    if sigma < 0 or tau <= 0 or dt <= 0:
        raise ValueError("sigma must be non-negative and tau/dt must be positive")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    h, w, d = edge_mask.shape
    if n_steps is None:
        n_steps = int(np.ceil(4.0 * tau / dt))

    thetas = np.ascontiguousarray(
        np.arange(n_theta) * (2.0 * np.pi / n_theta), dtype=np.float64
    )
    phis = np.ascontiguousarray(
        (np.arange(n_phi) + 0.5) * (np.pi / n_phi), dtype=np.float64
    )

    coords = np.nonzero(edge_mask)
    if coords[0].size == 0:
        zero = np.zeros((h, w, d), dtype=np.float32)
        if return_orientation:
            return zero, zero.copy(), zero.copy()
        return zero

    theta_pt, phi_pt = edges_from_mask_3d(edge_mask, coords, tensor_sigma)
    weights = edge_mask[coords].astype(np.float32)

    sparse_injection = prepare_sparse_injection_3d(
        coords,
        theta_pt,
        phi_pt,
        weights,
        (h, w, d),
        n_theta,
        n_phi,
    )
    # These full-size coordinate/angle arrays are no longer needed once the
    # compact sparse plan has been created.
    del coords, theta_pt, phi_pt, weights
    field = propagate_3d(
        (h, w, d),
        sparse_injection,
        thetas,
        phis,
        sigma,
        tau,
        dt,
        n_steps,
        num_threads=num_threads,
    )
    reduced = antipodal_reduce3d(
        field.reshape(-1),
        h * w * d,
        n_theta,
        n_phi,
        return_orientation,
        num_threads,
    )
    if return_orientation:
        c_flat, best_t_flat, best_p_flat = reduced
        c_field = c_flat.reshape(h, w, d)
        best_t = best_t_flat.reshape(h, w, d)
        best_p = best_p_flat.reshape(h, w, d)
    else:
        c_field = reduced.reshape(h, w, d)

    peak = float(c_field.max())
    if peak > 0.0:
        c_field /= peak

    if return_orientation:
        theta_map = np.mod(thetas[best_t], np.pi)
        phi_map = phis[best_p]
        return c_field, theta_map, phi_map
    return c_field
