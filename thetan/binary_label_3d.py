from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - optional dependency
    njit = None
    _HAS_NUMBA = False


ArrayLike = np.ndarray


def make_ellipsoid_kernel(radii: Sequence[int]) -> np.ndarray:
    """
    Create a 3D ellipsoidal connectivity kernel.

    Parameters
    ----------
    radii:
        (rz, ry, rx) radii in voxels. The output shape is
        (2*rz + 1, 2*ry + 1, 2*rx + 1).

    Returns
    -------
    np.ndarray
        Boolean 3D kernel whose center voxel is included.

    Examples
    --------
    >>> k = make_ellipsoid_kernel((5, 5, 5))
    >>> k.shape
    (11, 11, 11)
    """
    r = np.asarray(radii, dtype=np.int64)
    if r.shape != (3,):
        raise ValueError("radii must be a sequence of length 3: (rz, ry, rx)")
    if np.any(r < 0):
        raise ValueError("radii must be non-negative")

    rz, ry, rx = (int(v) for v in r)
    z, y, x = np.ogrid[-rz : rz + 1, -ry : ry + 1, -rx : rx + 1]

    # max(radius, 1) avoids division by zero when a radius is 0.
    dz = z / max(rz, 1)
    dy = y / max(ry, 1)
    dx = x / max(rx, 1)
    return (dz * dz + dy * dy + dx * dx) <= 1.0



def make_ball_kernel(radius: int) -> np.ndarray:
    """
    Create a solid spherical connectivity kernel.

    Parameters
    ----------
    radius:
        Sphere radius in voxels. The returned shape is
        (2*radius + 1, 2*radius + 1, 2*radius + 1).

    Returns
    -------
    np.ndarray
        Boolean kernel.

    Examples
    --------
    >>> k = make_ball_kernel(5)
    >>> k.shape
    (11, 11, 11)
    """
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return make_ellipsoid_kernel((radius, radius, radius))



def kernel_offsets(
    kernel: ArrayLike,
    origin: Sequence[int] | None = None,
    *,
    symmetrize: bool = True,
    include_center: bool = False,
) -> np.ndarray:
    """
    Convert a 3D kernel into integer offsets.

    Any non-zero kernel entry becomes a valid offset. By default the center
    offset (0, 0, 0) is removed because a voxel is always connected to itself,
    but it can be kept when the full kernel footprint is needed for local
    coverage tests.

    Parameters
    ----------
    kernel:
        3D array. Non-zero entries define connectivity / support.
    origin:
        Kernel center in index coordinates. By default the integer center
        (shape // 2) is used.
    symmetrize:
        If True, offsets are mirrored so connectivity is undirected even if the
        supplied kernel is not perfectly centrosymmetric.
    include_center:
        If True, keep the center offset when the kernel contains it.

    Returns
    -------
    np.ndarray
        Array of shape (n_offsets, 3) with integer offsets.
    """
    k = np.asarray(kernel) != 0
    if k.ndim != 3:
        raise ValueError("kernel must be a 3D array")

    if origin is None:
        origin_arr = np.asarray(k.shape, dtype=np.int64) // 2
    else:
        origin_arr = np.asarray(origin, dtype=np.int64)
        if origin_arr.shape != (3,):
            raise ValueError("origin must be a sequence of length 3")
        if np.any(origin_arr < 0) or np.any(origin_arr >= np.asarray(k.shape)):
            raise ValueError("origin must lie inside kernel bounds")

    offsets = np.argwhere(k).astype(np.int64) - origin_arr
    if not include_center:
        offsets = offsets[np.any(offsets != 0, axis=1)]  # drop (0, 0, 0)

    if symmetrize and offsets.size:
        base = {tuple(v) for v in offsets.tolist()}
        mirrored = {(-dz, -dy, -dx) for dz, dy, dx in base}
        offsets = np.asarray(sorted(base | mirrored), dtype=np.int64)

    return offsets



def _prepare_labeling(
    volume: ArrayLike,
    kernel: ArrayLike,
    origin: Sequence[int] | None,
    symmetrize: bool,
):
    vol = np.asarray(volume) != 0
    if vol.ndim != 3:
        raise ValueError("volume must be a 3D array")

    neighbor_offsets = kernel_offsets(
        kernel,
        origin=origin,
        symmetrize=symmetrize,
        include_center=False,
    )
    support_offsets = kernel_offsets(
        kernel,
        origin=origin,
        symmetrize=symmetrize,
        include_center=True,
    )

    extent_offsets = support_offsets if support_offsets.size else neighbor_offsets
    if extent_offsets.size:
        min_off = extent_offsets.min(axis=0)
        max_off = extent_offsets.max(axis=0)
        pad_before = np.maximum(-min_off, 0)
        pad_after = np.maximum(max_off, 0)
    else:
        pad_before = np.zeros(3, dtype=np.int64)
        pad_after = np.zeros(3, dtype=np.int64)

    pad_width = tuple((int(a), int(b)) for a, b in zip(pad_before, pad_after))
    padded_fg = np.pad(vol, pad_width, mode="constant", constant_values=False)

    valid = np.ones(vol.shape, dtype=bool)
    padded_valid = np.pad(valid, pad_width, mode="constant", constant_values=False)

    _, sy, sx = padded_fg.shape
    strides = np.asarray([sy * sx, sx, 1], dtype=np.int64)

    neighbor_shifts = (
        (neighbor_offsets @ strides).astype(np.int64)
        if neighbor_offsets.size
        else np.empty(0, dtype=np.int64)
    )
    support_shifts = (
        (support_offsets @ strides).astype(np.int64)
        if support_offsets.size
        else np.empty(0, dtype=np.int64)
    )

    flat_fg = padded_fg.ravel()
    flat_valid = padded_valid.ravel()
    starts = np.flatnonzero(flat_fg).astype(np.int64)

    return (
        vol.shape,
        padded_fg.shape,
        pad_before,
        starts,
        neighbor_shifts,
        support_shifts,
        flat_fg,
        flat_valid,
    )



def _apply_coverage_threshold_python(
    flat_fg: np.ndarray,
    flat_valid: np.ndarray,
    starts: np.ndarray,
    support_shifts: np.ndarray,
    min_coverage: float,
):
    """
    Keep only foreground voxels whose kernel footprint is sufficiently covered.

    Coverage is computed as:
        (# active kernel positions overlapping foreground) /
        (# active kernel positions that fall inside the image)

    Only in-bounds kernel positions count toward the denominator so objects that
    touch the image boundary are not penalized for out-of-bounds support.
    """
    if min_coverage <= 0.0:
        return flat_fg, starts

    supported = np.zeros(flat_fg.shape[0], dtype=bool)
    kept_starts = np.empty(starts.shape[0], dtype=np.int64)
    n_kept = 0

    for s in starts:
        valid_count = 0
        white_count = 0

        for shift in support_shifts:
            idx = s + shift
            if flat_valid[idx]:
                valid_count += 1
                if flat_fg[idx]:
                    white_count += 1

        if valid_count > 0 and white_count >= min_coverage * valid_count:
            supported[s] = True
            kept_starts[n_kept] = s
            n_kept += 1

    return supported, kept_starts[:n_kept]



def _label_flat_python(flat_fg: np.ndarray, starts: np.ndarray, shifts: np.ndarray):
    """Pure-Python fallback using an explicit stack over flattened indices."""
    labels = np.zeros(flat_fg.shape[0], dtype=np.int64)
    sizes = np.empty(starts.shape[0], dtype=np.int64)
    stack = np.empty(starts.shape[0], dtype=np.int64)

    n_labels = 0
    for s in starts:
        if labels[s] != 0:
            continue

        n_labels += 1
        labels[s] = n_labels
        top = 1
        stack[0] = s
        comp_size = 0

        while top > 0:
            top -= 1
            idx = stack[top]
            comp_size += 1

            for shift in shifts:
                nidx = idx + shift
                if flat_fg[nidx] and labels[nidx] == 0:
                    labels[nidx] = n_labels
                    stack[top] = nidx
                    top += 1

        sizes[n_labels - 1] = comp_size

    return labels, sizes[:n_labels]


if _HAS_NUMBA:

    @njit
    def _apply_coverage_threshold_numba(
        flat_fg: np.ndarray,
        flat_valid: np.ndarray,
        starts: np.ndarray,
        support_shifts: np.ndarray,
        min_coverage: float,
    ):
        supported = np.zeros(flat_fg.shape[0], dtype=np.bool_)
        kept_starts = np.empty(starts.shape[0], dtype=np.int64)
        n_kept = 0

        for si in range(starts.shape[0]):
            s = starts[si]
            valid_count = 0
            white_count = 0

            for k in range(support_shifts.shape[0]):
                idx = s + support_shifts[k]
                if flat_valid[idx]:
                    valid_count += 1
                    if flat_fg[idx]:
                        white_count += 1

            if valid_count > 0 and white_count >= min_coverage * valid_count:
                supported[s] = True
                kept_starts[n_kept] = s
                n_kept += 1

        return supported, kept_starts[:n_kept]


    @njit
    def _label_flat_numba(flat_fg: np.ndarray, starts: np.ndarray, shifts: np.ndarray):
        """Numba-accelerated flood fill over flattened indices."""
        labels = np.zeros(flat_fg.shape[0], dtype=np.int64)
        sizes = np.empty(starts.shape[0], dtype=np.int64)
        stack = np.empty(starts.shape[0], dtype=np.int64)

        n_labels = 0
        for si in range(starts.shape[0]):
            s = starts[si]
            if labels[s] != 0:
                continue

            n_labels += 1
            labels[s] = n_labels
            top = 1
            stack[0] = s
            comp_size = 0

            while top > 0:
                top -= 1
                idx = stack[top]
                comp_size += 1

                for k in range(shifts.shape[0]):
                    nidx = idx + shifts[k]
                    if flat_fg[nidx] and labels[nidx] == 0:
                        labels[nidx] = n_labels
                        stack[top] = nidx
                        top += 1

            sizes[n_labels - 1] = comp_size

        return labels, sizes[:n_labels]



def label_binary_3d(
    volume: ArrayLike,
    kernel: ArrayLike,
    *,
    origin: Sequence[int] | None = None,
    symmetrize: bool = True,
    min_coverage: float = 0.0,
    use_numba: bool = True,
    output_dtype=np.int32,
    return_num: bool = False,
    return_sizes: bool = False,
):
    """
    Connected-component labeling for 3D binary data with an arbitrary kernel.

    This does not use `scipy.ndimage.label` or `skimage.measure.label`. The
    kernel may be any 3D support mask, including large supports like an
    11x11x11 solid ball.

    Parameters
    ----------
    volume:
        3D input array. Any non-zero value is treated as foreground.
    kernel:
        3D connectivity mask. Any non-zero value is a neighbor relation.
    origin:
        Kernel center. Default is `np.array(kernel.shape) // 2`.
    symmetrize:
        Mirror all kernel offsets so connectivity is undirected.
    min_coverage:
        Minimum local kernel coverage required for a foreground voxel to take
        part in any component. This must lie in [0, 1].

        The local coverage at a voxel is

            (# active kernel positions overlapping foreground) /
            (# active kernel positions that fall inside the image)

        using the kernel footprint centered on that voxel. A value of 0.0
        disables this extra gating and reproduces the original behavior. A
        value of 1.0 means every in-bounds active kernel position must land on
        foreground for that voxel to remain connectable.
    use_numba:
        If True and numba is available, use the JIT-accelerated backend.
    output_dtype:
        Integer dtype for the label image.
    return_num:
        If True, also return the number of components.
    return_sizes:
        If True, also return component sizes in label order.

    Returns
    -------
    labels : np.ndarray
        Labeled 3D array, same shape as `volume`.
    num : int, optional
        Number of connected components.
    sizes : np.ndarray, optional
        1D array of component sizes, one entry per label.

    Notes
    -----
    Complexity is O(F * K), where F is the number of foreground voxels and K is
    the number of active kernel entries. Large dense volumes with very large
    kernels can still be expensive, but the implementation supports arbitrary
    kernel sizes exactly.
    """
    min_coverage = float(min_coverage)
    if not (0.0 <= min_coverage <= 1.0):
        raise ValueError("min_coverage must lie in the range [0, 1]")

    (
        volume_shape,
        padded_shape,
        pad_before,
        starts,
        neighbor_shifts,
        support_shifts,
        flat_fg,
        flat_valid,
    ) = _prepare_labeling(volume, kernel, origin, symmetrize)

    if min_coverage > 0.0:
        if use_numba and _HAS_NUMBA:
            flat_connectable, starts = _apply_coverage_threshold_numba(
                flat_fg,
                flat_valid,
                starts,
                support_shifts,
                min_coverage,
            )
        else:
            flat_connectable, starts = _apply_coverage_threshold_python(
                flat_fg,
                flat_valid,
                starts,
                support_shifts,
                min_coverage,
            )
    else:
        flat_connectable = flat_fg

    if use_numba and _HAS_NUMBA:
        flat_labels, sizes = _label_flat_numba(flat_connectable, starts, neighbor_shifts)
    else:
        flat_labels, sizes = _label_flat_python(flat_connectable, starts, neighbor_shifts)

    n_components = int(sizes.shape[0])
    dtype = np.dtype(output_dtype)
    if np.issubdtype(dtype, np.integer):
        if n_components > np.iinfo(dtype).max:
            raise OverflowError(
                f"{n_components} components do not fit in dtype {dtype}"
            )
    else:
        raise TypeError("output_dtype must be an integer dtype")

    labels = flat_labels.reshape(padded_shape)
    z0, y0, x0 = (int(v) for v in pad_before)
    z1, y1, x1 = z0 + volume_shape[0], y0 + volume_shape[1], x0 + volume_shape[2]
    labels = labels[z0:z1, y0:y1, x0:x1].astype(dtype, copy=False)

    result = [labels]
    if return_num:
        result.append(n_components)
    if return_sizes:
        result.append(sizes)

    if len(result) == 1:
        return result[0]
    return tuple(result)


if __name__ == "__main__":
    # Example: two solid blocks joined by a one-voxel-thick rod. With
    # min_coverage > 0, the rod can be suppressed if too little of the kernel
    # footprint sees white.
    vol = np.zeros((50, 50, 50), dtype=np.uint8)
    vol[20:30, 20:30, 4:12] = 1
    vol[20:30, 20:30, 28:36] = 1
    vol[24, 24, 12:28] = 1  # thin bridge connecting the blocks

    kernel = make_ball_kernel(2)  # shape = (5, 5, 5)

    labels0, num0 = label_binary_3d(
        vol,
        kernel,
        min_coverage=0.0,
        return_num=True,
    )
    labels1, num1 = label_binary_3d(
        vol,
        kernel,
        min_coverage=0.2,
        return_num=True,
    )

    print(f"Without coverage gating: {num0} component(s)")
    print(f"With min_coverage=0.2: {num1} component(s)")
    print("Unique labels without gating:", np.unique(labels0).tolist())
    print("Unique labels with gating:", np.unique(labels1).tolist())
