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


def make_connectivity_kernel(connectivity: int = 26) -> np.ndarray:
    """
    Create a standard 3x3x3 connectivity kernel.

    Parameters
    ----------
    connectivity:
        One of {6, 18, 26}.

    Returns
    -------
    np.ndarray
        Boolean (3, 3, 3) kernel that includes the center voxel.

    Notes
    -----
    - 6-connectivity: face neighbors only.
    - 18-connectivity: faces + edges.
    - 26-connectivity: faces + edges + corners.
    """
    connectivity = int(connectivity)
    if connectivity not in (6, 18, 26):
        raise ValueError("connectivity must be one of {6, 18, 26}")

    k = np.zeros((3, 3, 3), dtype=bool)
    center = np.array([1, 1, 1], dtype=np.int64)

    for idx in np.ndindex(k.shape):
        d = np.abs(np.asarray(idx, dtype=np.int64) - center)
        s = int(d.sum())
        if connectivity == 6:
            keep = s <= 1
        elif connectivity == 18:
            keep = s <= 2
        else:  # 26
            keep = True
        k[idx] = keep

    return k


# -----------------------------------------------------------------------------
# Offset helpers
# -----------------------------------------------------------------------------


def kernel_offsets(
    kernel: ArrayLike,
    origin: Sequence[int] | None = None,
    *,
    symmetrize: bool = True,
    include_center: bool = False,
) -> np.ndarray:
    """
    Convert a 3D kernel into integer offsets.

    Parameters
    ----------
    kernel:
        3D array. Any non-zero entry becomes an active offset.
    origin:
        Kernel center in index coordinates. By default the integer center
        (`shape // 2`) is used.
    symmetrize:
        If True, mirror all offsets so the resulting relation is undirected.
    include_center:
        If True, keep the (0, 0, 0) offset when the kernel contains it.

    Returns
    -------
    np.ndarray
        Array of shape (n_offsets, 3) with integer offsets.
    """
    k = np.asarray(kernel) != 0
    if k.ndim != 3:
        raise ValueError("kernel must be a 3D array")
    if not np.any(k):
        raise ValueError("kernel must contain at least one non-zero element")

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
        offsets = offsets[np.any(offsets != 0, axis=1)]

    if symmetrize and offsets.size:
        base = {tuple(v) for v in offsets.tolist()}
        mirrored = {(-dz, -dy, -dx) for dz, dy, dx in base}
        offsets = np.asarray(sorted(base | mirrored), dtype=np.int64)

    return offsets



def _pad_and_flatten(
    vol: np.ndarray,
    offsets: np.ndarray,
    *,
    need_valid_mask: bool,
):
    """Pad a boolean 3D volume so all shifted lookups are in-bounds."""
    if vol.ndim != 3:
        raise ValueError("volume must be a 3D array")

    if offsets.size:
        min_off = offsets.min(axis=0)
        max_off = offsets.max(axis=0)
        pad_before = np.maximum(-min_off, 0)
        pad_after = np.maximum(max_off, 0)
    else:
        pad_before = np.zeros(3, dtype=np.int64)
        pad_after = np.zeros(3, dtype=np.int64)

    pad_width = tuple((int(a), int(b)) for a, b in zip(pad_before, pad_after))
    padded_fg = np.pad(vol, pad_width, mode="constant", constant_values=False)

    padded_valid = None
    if need_valid_mask:
        valid = np.ones(vol.shape, dtype=bool)
        padded_valid = np.pad(valid, pad_width, mode="constant", constant_values=False)

    _, sy, sx = padded_fg.shape
    strides = np.asarray([sy * sx, sx, 1], dtype=np.int64)
    shifts = (offsets @ strides).astype(np.int64) if offsets.size else np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(padded_fg.ravel()).astype(np.int64)

    return (
        vol.shape,
        padded_fg.shape,
        pad_before,
        starts,
        shifts,
        padded_fg.ravel(),
        None if padded_valid is None else padded_valid.ravel(),
    )



def _crop_padded(arr: np.ndarray, original_shape: Sequence[int], pad_before: Sequence[int]) -> np.ndarray:
    z0, y0, x0 = (int(v) for v in pad_before)
    z1 = z0 + int(original_shape[0])
    y1 = y0 + int(original_shape[1])
    x1 = x0 + int(original_shape[2])
    return arr[z0:z1, y0:y1, x0:x1]


# -----------------------------------------------------------------------------
# Coverage filtering
# -----------------------------------------------------------------------------


def _coverage_mask_python(
    flat_fg: np.ndarray,
    flat_valid: np.ndarray,
    starts: np.ndarray,
    support_shifts: np.ndarray,
    min_coverage: float,
) -> np.ndarray:
    """Return a flat boolean array marking supported foreground voxels."""
    supported = np.zeros(flat_fg.shape[0], dtype=bool)

    for s in starts:
        valid_count = 0
        white_count = 0

        for shift in support_shifts:
            idx = s + shift
            if flat_valid[idx]:
                valid_count += 1
                if flat_fg[idx]:
                    white_count += 1

        if valid_count > 0 and (white_count / valid_count) >= min_coverage:
            supported[s] = True

    return supported


if _HAS_NUMBA:

    @njit(cache=True)
    def _coverage_mask_numba(
        flat_fg: np.ndarray,
        flat_valid: np.ndarray,
        starts: np.ndarray,
        support_shifts: np.ndarray,
        min_coverage: float,
    ) -> np.ndarray:
        supported = np.zeros(flat_fg.shape[0], dtype=np.bool_)

        for i in range(starts.shape[0]):
            s = starts[i]
            valid_count = 0
            white_count = 0

            for k in range(support_shifts.shape[0]):
                idx = s + support_shifts[k]
                if flat_valid[idx]:
                    valid_count += 1
                    if flat_fg[idx]:
                        white_count += 1

            if valid_count > 0 and (white_count / valid_count) >= min_coverage:
                supported[s] = True

        return supported



def filter_by_local_coverage(
    volume: ArrayLike,
    kernel: ArrayLike,
    *,
    min_coverage: float = 0.0,
    origin: Sequence[int] | None = None,
    symmetrize: bool = True,
    use_numba: bool = True,
) -> np.ndarray:
    """
    Keep only foreground voxels that are locally supported by the kernel.

    For every foreground voxel, this computes:

        coverage = (# active kernel positions overlapping foreground) /
                   (# active kernel positions that are in-bounds)

    A voxel is kept iff `coverage >= min_coverage`.

    Parameters
    ----------
    volume:
        3D binary input. Any non-zero value is treated as foreground.
    kernel:
        3D support kernel. Any non-zero value is part of the support footprint.
    min_coverage:
        Threshold in [0, 1].
        - 0.0 keeps all foreground voxels.
        - 1.0 requires the full in-bounds kernel footprint to be foreground.
    origin:
        Kernel center. Default is `np.array(kernel.shape) // 2`.
    symmetrize:
        Mirror all kernel offsets so the support is centrosymmetric.
    use_numba:
        If True and numba is available, use the accelerated backend.

    Returns
    -------
    np.ndarray
        Boolean 3D mask of supported voxels.
    """
    if not (0.0 <= float(min_coverage) <= 1.0):
        raise ValueError("min_coverage must be between 0 and 1")

    vol = np.asarray(volume) != 0
    if vol.ndim != 3:
        raise ValueError("volume must be a 3D array")
    if min_coverage <= 0.0:
        return vol.copy()

    support_offsets = kernel_offsets(
        kernel,
        origin=origin,
        symmetrize=symmetrize,
        include_center=True,
    )

    (
        volume_shape,
        padded_shape,
        pad_before,
        starts,
        support_shifts,
        flat_fg,
        flat_valid,
    ) = _pad_and_flatten(vol, support_offsets, need_valid_mask=True)

    if use_numba and _HAS_NUMBA:
        flat_supported = _coverage_mask_numba(
            flat_fg,
            flat_valid,
            starts,
            support_shifts,
            float(min_coverage),
        )
    else:
        flat_supported = _coverage_mask_python(
            flat_fg,
            flat_valid,
            starts,
            support_shifts,
            float(min_coverage),
        )

    supported = flat_supported.reshape(padded_shape)
    return _crop_padded(supported, volume_shape, pad_before)


# -----------------------------------------------------------------------------
# Connected components on a filtered volume
# -----------------------------------------------------------------------------


def _label_flat_python(flat_fg: np.ndarray, starts: np.ndarray, shifts: np.ndarray):
    """Pure-Python fallback using an explicit stack over flattened indices."""
    labels = np.zeros(flat_fg.shape[0], dtype=np.int64)
    sizes = np.empty(starts.shape[0], dtype=np.int64)
    stack = np.empty(max(1, starts.shape[0]), dtype=np.int64)

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
                    if top >= stack.shape[0]:
                        stack = np.concatenate([stack, np.empty(stack.shape[0], dtype=np.int64)])
                    stack[top] = nidx
                    top += 1

        sizes[n_labels - 1] = comp_size

    return labels, sizes[:n_labels]


if _HAS_NUMBA:

    @njit(cache=True)
    def _label_flat_numba(flat_fg: np.ndarray, starts: np.ndarray, shifts: np.ndarray):
        labels = np.zeros(flat_fg.shape[0], dtype=np.int64)
        sizes = np.empty(starts.shape[0], dtype=np.int64)
        stack = np.empty(max(1, starts.shape[0]), dtype=np.int64)

        n_labels = 0
        for i in range(starts.shape[0]):
            s = starts[i]
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



def label_binary_3d_v2(
    volume: ArrayLike,
    kernel: ArrayLike,
    *,
    connectivity_kernel: ArrayLike | None = None,
    min_coverage: float = 0.0,
    origin: Sequence[int] | None = None,
    symmetrize: bool = True,
    use_numba: bool = True,
    output_dtype=np.int32,
    return_num: bool = False,
    return_sizes: bool = False,
    return_filtered: bool = False,
):
    """
    Label connected components in a 3D binary volume.

    This does not use `scipy.ndimage.label` or `skimage.measure.label`.

    Important behavior:
    - `kernel` is the support kernel used for the local coverage test.
    - If `connectivity_kernel` is not given, the same `kernel` is also used as
      the connectivity kernel, which preserves the earlier behavior.
    - To avoid long-range merges across narrow gaps, use a large `kernel` for
      support and a small `connectivity_kernel` (typically 6- or 26-connected)
      for the actual graph connectivity.

    Parameters
    ----------
    volume:
        3D input array. Any non-zero value is treated as foreground.
    kernel:
        3D support kernel. Used to compute the local foreground coverage.
    connectivity_kernel:
        Optional 3D kernel that defines which voxels are direct graph neighbors.
        If omitted, `kernel` is also used for connectivity.
    min_coverage:
        Threshold in [0, 1] for the local support test.
        - 0.0 keeps all foreground voxels.
        - 1.0 requires the full in-bounds support footprint to be foreground.
    origin:
        Kernel center. Default is `np.array(kernel.shape) // 2`.
        The same origin is used for both kernels.
    symmetrize:
        Mirror offsets so support and connectivity are undirected.
    use_numba:
        If True and numba is available, use the accelerated backend.
    output_dtype:
        Integer dtype for the label image.
    return_num:
        If True, also return the number of components.
    return_sizes:
        If True, also return component sizes in label order.
    return_filtered:
        If True, also return the post-coverage binary mask that was actually
        labeled.

    Returns
    -------
    labels : np.ndarray
        Labeled 3D array, same shape as `volume`.
    num : int, optional
        Number of connected components.
    sizes : np.ndarray, optional
        1D array of component sizes, one entry per label.
    filtered : np.ndarray, optional
        Boolean 3D mask after the local coverage filter.
    """
    if not (0.0 <= float(min_coverage) <= 1.0):
        raise ValueError("min_coverage must be between 0 and 1")

    filtered = filter_by_local_coverage(
        volume,
        kernel,
        min_coverage=float(min_coverage),
        origin=origin,
        symmetrize=symmetrize,
        use_numba=use_numba,
    )

    conn_kernel = kernel if connectivity_kernel is None else connectivity_kernel
    neighbor_offsets = kernel_offsets(
        conn_kernel,
        origin=origin,
        symmetrize=symmetrize,
        include_center=False,
    )

    (
        volume_shape,
        padded_shape,
        pad_before,
        starts,
        shifts,
        flat_fg,
        _,
    ) = _pad_and_flatten(filtered, neighbor_offsets, need_valid_mask=False)

    if use_numba and _HAS_NUMBA:
        flat_labels, sizes = _label_flat_numba(flat_fg, starts, shifts)
    else:
        flat_labels, sizes = _label_flat_python(flat_fg, starts, shifts)

    n_components = int(sizes.shape[0])
    dtype = np.dtype(output_dtype)
    if not np.issubdtype(dtype, np.integer):
        raise TypeError("output_dtype must be an integer dtype")
    if n_components > np.iinfo(dtype).max:
        raise OverflowError(f"{n_components} components do not fit in dtype {dtype}")

    labels = flat_labels.reshape(padded_shape)
    labels = _crop_padded(labels, volume_shape, pad_before).astype(dtype, copy=False)

    result = [labels]
    if return_num:
        result.append(n_components)
    if return_sizes:
        result.append(sizes)
    if return_filtered:
        result.append(filtered)

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

    # Demo: use a large support kernel, but local connectivity.

    vol = np.zeros((50, 50, 50), dtype=np.uint8)
    vol[8:18, 8:18, 8:18] = 1
    vol[22:32, 22:32, 22:32] = 1

    support_kernel = make_ball_kernel(5)                # 11x11x11 local support
    connectivity_kernel = make_connectivity_kernel(26)  # local graph edges

    labels, num, sizes, filtered = label_binary_3d_v2(
        vol,
        support_kernel,
        connectivity_kernel=connectivity_kernel,
        min_coverage=0.35,
        return_num=True,
        return_sizes=True,
        return_filtered=True,
    )

    print(f"Found {num} components")
    print("Sizes:", sizes.tolist())
    print("Foreground after filtering:", int(filtered.sum()))
