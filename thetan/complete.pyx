# cython: language_level=3, boundscheck=False, wraparound=False
# cython: initializedcheck=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as cnp

from cython.parallel cimport prange
from libc.float cimport DBL_MAX, FLT_MAX

cdef void advect_row(
    const double[:, :, ::1] field,
    double[:, :, ::1] output,
    const double[::1] row_shifts,
    const double[::1] column_shifts,
    Py_ssize_t row,
) noexcept nogil:
    cdef:
        Py_ssize_t height = field.shape[0]
        Py_ssize_t width = field.shape[1]
        Py_ssize_t channels = field.shape[2]
        Py_ssize_t column, channel
        Py_ssize_t row0, row1, column0, column1
        double source_row, source_column
        double row_fraction, column_fraction
        double top, bottom

    for column in range(width):
        for channel in range(channels):
            source_row = row - row_shifts[channel]
            source_column = column - column_shifts[channel]

            if (
                source_row < 0.0
                or source_row > height - 1
                or source_column < 0.0
                or source_column > width - 1
            ):
                output[row, column, channel] = 0.0
                continue

            row0 = <Py_ssize_t>source_row
            column0 = <Py_ssize_t>source_column
            row1 = row0 + 1
            column1 = column0 + 1
            if row1 == height:
                row1 = row0
            if column1 == width:
                column1 = column0

            row_fraction = source_row - row0
            column_fraction = source_column - column0

            top = field[row0, column0, channel] + column_fraction * (
                field[row0, column1, channel] - field[row0, column0, channel]
            )
            bottom = field[row1, column0, channel] + column_fraction * (
                field[row1, column1, channel] - field[row1, column0, channel]
            )
            output[row, column, channel] = top + row_fraction * (bottom - top)


def advect(
    const double[:, :, ::1] field,
    const double[::1] row_shifts,
    const double[::1] column_shifts,
    int num_threads=0,
):
    cdef:
        Py_ssize_t height = field.shape[0]
        Py_ssize_t width = field.shape[1]
        Py_ssize_t channels = field.shape[2]
        Py_ssize_t row
        cnp.ndarray[cnp.float64_t, ndim=3] output_array
        double[:, :, ::1] output

    if row_shifts.shape[0] != channels or column_shifts.shape[0] != channels:
        raise ValueError("shift arrays must match the field's channel dimension")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    output_array = np.empty((height, width, channels), dtype=np.float64)
    output = output_array

    if num_threads == 0:
        with nogil:
            for row in prange(height, schedule="static"):
                advect_row(
                    field, output, row_shifts, column_shifts, row
                )
    else:
        with nogil:
            for row in prange(
                height, schedule="static", num_threads=num_threads
            ):
                advect_row(
                    field, output, row_shifts, column_shifts, row
                )

    return output_array


cdef void advect_row_3d(
    const cnp.float32_t[::1] source,
    cnp.float32_t[::1] destination,
    const cnp.intp_t[:, ::1] source_offsets,
    const double[:, ::1] interpolation_weights,
    const double[:, ::1] shifts,
    const cnp.intp_t[::1] interior_bounds,
    Py_ssize_t height,
    Py_ssize_t width,
    Py_ssize_t depth,
    Py_ssize_t channels,
    Py_ssize_t row,
) noexcept nogil:
    cdef:
        Py_ssize_t column, plane, channel, corner
        Py_ssize_t spatial_base, output_index, source_index
        Py_ssize_t row0, row1, column0, column1, plane0, plane1
        Py_ssize_t base000, base001, base010, base011
        Py_ssize_t base100, base101, base110, base111
        bint in_interior_row, in_interior
        double source_row, source_column, source_plane
        double row_fraction, column_fraction, plane_fraction
        double c00, c01, c10, c11, c0, c1, value

    in_interior_row = (
        row >= interior_bounds[0] and row < interior_bounds[1]
    )

    for column in range(width):
        for plane in range(depth):
            spatial_base = ((row * width + column) * depth + plane) * channels
            in_interior = (
                in_interior_row
                and column >= interior_bounds[2]
                and column < interior_bounds[3]
                and plane >= interior_bounds[4]
                and plane < interior_bounds[5]
            )

            if in_interior:
                for channel in range(channels):
                    value = 0.0
                    for corner in range(8):
                        source_index = (
                            spatial_base
                            + source_offsets[channel, corner]
                            + channel
                        )
                        value += (
                            interpolation_weights[channel, corner]
                            * source[source_index]
                        )
                    destination[spatial_base + channel] = <cnp.float32_t>value
                continue


            for channel in range(channels):
                source_row = row - shifts[channel, 0]
                source_column = column - shifts[channel, 1]
                source_plane = plane - shifts[channel, 2]
                output_index = spatial_base + channel

                if (
                    source_row < 0.0
                    or source_row > height - 1
                    or source_column < 0.0
                    or source_column > width - 1
                    or source_plane < 0.0
                    or source_plane > depth - 1
                ):
                    destination[output_index] = 0.0
                    continue

                row0 = <Py_ssize_t>source_row
                column0 = <Py_ssize_t>source_column
                plane0 = <Py_ssize_t>source_plane
                row1 = row0 + 1
                column1 = column0 + 1
                plane1 = plane0 + 1
                if row1 == height:
                    row1 = row0
                if column1 == width:
                    column1 = column0
                if plane1 == depth:
                    plane1 = plane0

                row_fraction = source_row - row0
                column_fraction = source_column - column0
                plane_fraction = source_plane - plane0

                base000 = ((row0 * width + column0) * depth + plane0) * channels + channel
                base001 = ((row0 * width + column0) * depth + plane1) * channels + channel
                base010 = ((row0 * width + column1) * depth + plane0) * channels + channel
                base011 = ((row0 * width + column1) * depth + plane1) * channels + channel
                base100 = ((row1 * width + column0) * depth + plane0) * channels + channel
                base101 = ((row1 * width + column0) * depth + plane1) * channels + channel
                base110 = ((row1 * width + column1) * depth + plane0) * channels + channel
                base111 = ((row1 * width + column1) * depth + plane1) * channels + channel

                c00 = source[base000] + plane_fraction * (
                    source[base001] - source[base000]
                )
                c01 = source[base010] + plane_fraction * (
                    source[base011] - source[base010]
                )
                c10 = source[base100] + plane_fraction * (
                    source[base101] - source[base100]
                )
                c11 = source[base110] + plane_fraction * (
                    source[base111] - source[base110]
                )
                c0 = c00 + column_fraction * (c01 - c00)
                c1 = c10 + column_fraction * (c11 - c10)
                destination[output_index] = <cnp.float32_t>(
                    c0 + row_fraction * (c1 - c0)
                )

def advect_3d_into(
    const cnp.float32_t[::1] source,
    cnp.float32_t[::1] destination,
    const cnp.intp_t[:, ::1] source_offsets,
    const double[:, ::1] interpolation_weights,
    const double[:, ::1] shifts,
    const cnp.intp_t[::1] interior_bounds,
    Py_ssize_t height,
    Py_ssize_t width,
    Py_ssize_t depth,
    int num_threads=0,
):
    cdef:
        Py_ssize_t channels = source_offsets.shape[0]
        Py_ssize_t expected_size = height * width * depth * channels
        Py_ssize_t row

    if height <= 0 or width <= 0 or depth <= 0 or channels <= 0:
        raise ValueError("all field dimensions must be positive")
    if source.shape[0] != expected_size or destination.shape[0] != expected_size:
        raise ValueError("source and destination sizes do not match the field dimensions")
    if source_offsets.shape[1] != 8:
        raise ValueError("source_offsets must have shape (channels, 8)")
    if interpolation_weights.shape[0] != channels or interpolation_weights.shape[1] != 8:
        raise ValueError("interpolation_weights must have shape (channels, 8)")
    if shifts.shape[0] != channels or shifts.shape[1] != 3:
        raise ValueError("shifts must have shape (channels, 3)")
    if interior_bounds.shape[0] != 6:
        raise ValueError("interior_bounds must contain six values")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for row in prange(height, schedule="static"):
                advect_row_3d(
                    source,
                    destination,
                    source_offsets,
                    interpolation_weights,
                    shifts,
                    interior_bounds,
                    height,
                    width,
                    depth,
                    channels,
                    row,
                )
    else:
        with nogil:
            for row in prange(
                height, schedule="static", num_threads=num_threads
            ):
                advect_row_3d(
                    source,
                    destination,
                    source_offsets,
                    interpolation_weights,
                    shifts,
                    interior_bounds,
                    height,
                    width,
                    depth,
                    channels,
                    row,
                )


cdef inline void add_sparse_point(
    cnp.float32_t[::1] field,
    const cnp.intp_t[::1] spatial_indices,
    const cnp.intp_t[::1] channel0,
    const cnp.intp_t[::1] channel1,
    const cnp.float32_t[::1] values,
    Py_ssize_t channels,
    Py_ssize_t point,
) noexcept nogil:
    cdef Py_ssize_t base = spatial_indices[point] * channels
    field[base + channel0[point]] += values[point]
    field[base + channel1[point]] += values[point]

def add_sparse_injection3d(
    cnp.float32_t[::1] field,
    const cnp.intp_t[::1] spatial_indices,
    const cnp.intp_t[::1] channel0,
    const cnp.intp_t[::1] channel1,
    const cnp.float32_t[::1] values,
    Py_ssize_t channels,
    int num_threads=0,
):
    cdef:
        Py_ssize_t n_points = spatial_indices.shape[0]
        Py_ssize_t point

    if channels <= 0:
        raise ValueError("channels must be positive")
    if (
        channel0.shape[0] != n_points
        or channel1.shape[0] != n_points
        or values.shape[0] != n_points
    ):
        raise ValueError("all sparse injection arrays must have equal length")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for point in prange(n_points, schedule="static"):
                add_sparse_point(
                    field,
                    spatial_indices,
                    channel0,
                    channel1,
                    values,
                    channels,
                    point,
                )
    else:
        with nogil:
            for point in prange(
                n_points, schedule="static", num_threads=num_threads
            ):
                add_sparse_point(
                    field,
                    spatial_indices,
                    channel0,
                    channel1,
                    values,
                    channels,
                    point,
                )

cdef inline void filter_theta_point(
    const cnp.float32_t[::1] source,
    cnp.float32_t[::1] destination,
    const double[::1] kernel,
    const cnp.intp_t[:, ::1] lookup,
    Py_ssize_t n_theta,
    Py_ssize_t n_phi,
    Py_ssize_t spatial,
) noexcept nogil:
    cdef:
        Py_ssize_t theta, phi, tap
        Py_ssize_t base = spatial * n_theta * n_phi
        double value

    for theta in range(n_theta):
        for phi in range(n_phi):
            value = 0.0
            for tap in range(kernel.shape[0]):
                value += kernel[tap] * source[
                    base + lookup[theta, tap] * n_phi + phi
                ]
            destination[base + theta * n_phi + phi] = <cnp.float32_t>value


def angular_filter_theta_into(
    const cnp.float32_t[::1] source,
    cnp.float32_t[::1] destination,
    const double[::1] kernel,
    const cnp.intp_t[:, ::1] lookup,
    Py_ssize_t n_spatial,
    Py_ssize_t n_theta,
    Py_ssize_t n_phi,
    int num_threads=0,
):
    cdef:
        Py_ssize_t expected_size = n_spatial * n_theta * n_phi
        Py_ssize_t spatial

    if source.shape[0] != expected_size or destination.shape[0] != expected_size:
        raise ValueError("source and destination sizes do not match")
    if lookup.shape[0] != n_theta or lookup.shape[1] != kernel.shape[0]:
        raise ValueError("theta lookup shape does not match the kernel")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for spatial in prange(n_spatial, schedule="static"):
                filter_theta_point(
                    source, destination, kernel, lookup, n_theta, n_phi, spatial
                )
    else:
        with nogil:
            for spatial in prange(
                n_spatial, schedule="static", num_threads=num_threads
            ):
                filter_theta_point(
                    source, destination, kernel, lookup, n_theta, n_phi, spatial
                )


cdef inline void filter_phi_point(
    const cnp.float32_t[::1] source,
    cnp.float32_t[::1] destination,
    const double[::1] kernel,
    const cnp.intp_t[:, ::1] lookup,
    Py_ssize_t n_theta,
    Py_ssize_t n_phi,
    Py_ssize_t spatial,
    double scale,
) noexcept nogil:
    cdef:
        Py_ssize_t theta, phi, tap
        Py_ssize_t base = spatial * n_theta * n_phi
        double value

    for theta in range(n_theta):
        for phi in range(n_phi):
            value = 0.0
            for tap in range(kernel.shape[0]):
                value += kernel[tap] * source[
                    base + theta * n_phi + lookup[phi, tap]
                ]
            destination[base + theta * n_phi + phi] = <cnp.float32_t>(
                value * scale
            )

def angular_filter_phi_into(
    const cnp.float32_t[::1] source,
    cnp.float32_t[::1] destination,
    const double[::1] kernel,
    const cnp.intp_t[:, ::1] lookup,
    Py_ssize_t n_spatial,
    Py_ssize_t n_theta,
    Py_ssize_t n_phi,
    double scale=1.0,
    int num_threads=0,
):
    cdef:
        Py_ssize_t expected_size = n_spatial * n_theta * n_phi
        Py_ssize_t spatial

    if source.shape[0] != expected_size or destination.shape[0] != expected_size:
        raise ValueError("source and destination sizes do not match")
    if lookup.shape[0] != n_phi or lookup.shape[1] != kernel.shape[0]:
        raise ValueError("phi lookup shape does not match the kernel")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for spatial in prange(n_spatial, schedule="static"):
                filter_phi_point(
                    source,
                    destination,
                    kernel,
                    lookup,
                    n_theta,
                    n_phi,
                    spatial,
                    scale,
                )
    else:
        with nogil:
            for spatial in prange(
                n_spatial, schedule="static", num_threads=num_threads
            ):
                filter_phi_point(
                    source,
                    destination,
                    kernel,
                    lookup,
                    n_theta,
                    n_phi,
                    spatial,
                    scale,
                )

cdef inline void antipodal_reduce_point(
    const cnp.float32_t[::1] field,
    cnp.float32_t[::1] completion,
    cnp.intp_t[::1] best_theta,
    cnp.intp_t[::1] best_phi,
    Py_ssize_t n_theta,
    Py_ssize_t n_phi,
    Py_ssize_t spatial,
    bint return_orientation,
) noexcept nogil:
    cdef:
        Py_ssize_t theta, phi, opposite_theta, opposite_phi
        Py_ssize_t base = spatial * n_theta * n_phi
        Py_ssize_t half = n_theta // 2
        cnp.float32_t product
        cnp.float32_t total = 0.0
        cnp.float32_t best_product = -FLT_MAX
        Py_ssize_t selected_theta = 0
        Py_ssize_t selected_phi = 0

    for phi in range(n_phi):
        opposite_phi = n_phi - 1 - phi
        for theta in range(n_theta):
            opposite_theta = theta + half
            if opposite_theta >= n_theta:
                opposite_theta -= n_theta
            product = (
                field[base + theta * n_phi + phi]
                * field[base + opposite_theta * n_phi + opposite_phi]
            )
            total += product
            if return_orientation and product > best_product:
                best_product = product
                selected_theta = theta
                selected_phi = phi

    completion[spatial] = total
    if return_orientation:
        best_theta[spatial] = selected_theta
        best_phi[spatial] = selected_phi


def antipodal_reduce3d(
    const cnp.float32_t[::1] field,
    Py_ssize_t n_spatial,
    Py_ssize_t n_theta,
    Py_ssize_t n_phi,
    bint return_orientation=False,
    int num_threads=0,
):
    cdef:
        Py_ssize_t expected_size = n_spatial * n_theta * n_phi
        Py_ssize_t spatial
        cnp.ndarray[cnp.float32_t, ndim=1] completion_array
        cnp.ndarray[cnp.intp_t, ndim=1] best_theta_array
        cnp.ndarray[cnp.intp_t, ndim=1] best_phi_array
        cnp.float32_t[::1] completion
        cnp.intp_t[::1] best_theta
        cnp.intp_t[::1] best_phi

    if n_theta <= 0 or n_theta % 2 != 0 or n_phi <= 0:
        raise ValueError("n_theta must be positive/even and n_phi must be positive")
    if field.shape[0] != expected_size:
        raise ValueError("field size does not match the supplied dimensions")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    completion_array = np.empty(n_spatial, dtype=np.float32)
    if return_orientation:
        best_theta_array = np.empty(n_spatial, dtype=np.intp)
        best_phi_array = np.empty(n_spatial, dtype=np.intp)
    else:
        best_theta_array = np.empty(1, dtype=np.intp)
        best_phi_array = np.empty(1, dtype=np.intp)
    completion = completion_array
    best_theta = best_theta_array
    best_phi = best_phi_array

    if num_threads == 0:
        with nogil:
            for spatial in prange(n_spatial, schedule="static"):
                antipodal_reduce_point(
                    field,
                    completion,
                    best_theta,
                    best_phi,
                    n_theta,
                    n_phi,
                    spatial,
                    return_orientation,
                )
    else:
        with nogil:
            for spatial in prange(
                n_spatial, schedule="static", num_threads=num_threads
            ):
                antipodal_reduce_point(
                    field,
                    completion,
                    best_theta,
                    best_phi,
                    n_theta,
                    n_phi,
                    spatial,
                    return_orientation,
                )

    if return_orientation:
        return completion_array, best_theta_array, best_phi_array
    return completion_array

cdef void advect_row_2d_into(
    const double[::1] source,
    double[::1] destination,
    const cnp.intp_t[:, ::1] source_offsets,
    const double[:, ::1] interpolation_weights,
    const double[:, ::1] shifts,
    const cnp.intp_t[::1] interior_bounds,
    Py_ssize_t height,
    Py_ssize_t width,
    Py_ssize_t channels,
    Py_ssize_t row,
) noexcept nogil:
    cdef:
        Py_ssize_t column, channel, corner
        Py_ssize_t spatial_base, output_index, source_index
        Py_ssize_t row0, row1, column0, column1
        Py_ssize_t base00, base01, base10, base11
        bint in_interior_row, in_interior
        double source_row, source_column
        double row_fraction, column_fraction
        double top, bottom, value

    in_interior_row = (
        row >= interior_bounds[0] and row < interior_bounds[1]
    )

    for column in range(width):
        spatial_base = (row * width + column) * channels
        in_interior = (
            in_interior_row
            and column >= interior_bounds[2]
            and column < interior_bounds[3]
        )

        if in_interior:
            for channel in range(channels):
                value = 0.0
                for corner in range(4):
                    source_index = (
                        spatial_base
                        + source_offsets[channel, corner]
                        + channel
                    )
                    value += (
                        interpolation_weights[channel, corner]
                        * source[source_index]
                    )
                destination[spatial_base + channel] = value
            continue

        for channel in range(channels):
            source_row = row - shifts[channel, 0]
            source_column = column - shifts[channel, 1]
            output_index = spatial_base + channel

            if (
                source_row < 0.0
                or source_row > height - 1
                or source_column < 0.0
                or source_column > width - 1
            ):
                destination[output_index] = 0.0
                continue

            row0 = <Py_ssize_t>source_row
            column0 = <Py_ssize_t>source_column
            row1 = row0 + 1
            column1 = column0 + 1
            if row1 == height:
                row1 = row0
            if column1 == width:
                column1 = column0

            row_fraction = source_row - row0
            column_fraction = source_column - column0

            base00 = (row0 * width + column0) * channels + channel
            base01 = (row0 * width + column1) * channels + channel
            base10 = (row1 * width + column0) * channels + channel
            base11 = (row1 * width + column1) * channels + channel

            top = source[base00] + column_fraction * (
                source[base01] - source[base00]
            )
            bottom = source[base10] + column_fraction * (
                source[base11] - source[base10]
            )
            destination[output_index] = top + row_fraction * (bottom - top)


def advect_2d_into(
    const double[::1] source,
    double[::1] destination,
    const cnp.intp_t[:, ::1] source_offsets,
    const double[:, ::1] interpolation_weights,
    const double[:, ::1] shifts,
    const cnp.intp_t[::1] interior_bounds,
    Py_ssize_t height,
    Py_ssize_t width,
    int num_threads=0,
):
    cdef:
        Py_ssize_t channels = source_offsets.shape[0]
        Py_ssize_t expected_size = height * width * channels
        Py_ssize_t row

    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError("all field dimensions must be positive")
    if source.shape[0] != expected_size or destination.shape[0] != expected_size:
        raise ValueError("source and destination sizes do not match the field dimensions")
    if source_offsets.shape[1] != 4:
        raise ValueError("source_offsets must have shape (channels, 4)")
    if interpolation_weights.shape[0] != channels or interpolation_weights.shape[1] != 4:
        raise ValueError("interpolation_weights must have shape (channels, 4)")
    if shifts.shape[0] != channels or shifts.shape[1] != 2:
        raise ValueError("shifts must have shape (channels, 2)")
    if interior_bounds.shape[0] != 4:
        raise ValueError("interior_bounds must contain four values")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for row in prange(height, schedule="static"):
                advect_row_2d_into(
                    source,
                    destination,
                    source_offsets,
                    interpolation_weights,
                    shifts,
                    interior_bounds,
                    height,
                    width,
                    channels,
                    row,
                )
    else:
        with nogil:
            for row in prange(
                height, schedule="static", num_threads=num_threads
            ):
                advect_row_2d_into(
                    source,
                    destination,
                    source_offsets,
                    interpolation_weights,
                    shifts,
                    interior_bounds,
                    height,
                    width,
                    channels,
                    row,
                )


cdef inline void add_sparse_point_2d(
    double[::1] field,
    const cnp.intp_t[::1] spatial_indices,
    const cnp.intp_t[::1] channel0,
    const cnp.intp_t[::1] channel1,
    const double[::1] values,
    Py_ssize_t channels,
    Py_ssize_t point,
) noexcept nogil:
    cdef Py_ssize_t base = spatial_indices[point] * channels
    field[base + channel0[point]] += values[point]
    field[base + channel1[point]] += values[point]

def add_sparse_injection2d(
    double[::1] field,
    const cnp.intp_t[::1] spatial_indices,
    const cnp.intp_t[::1] channel0,
    const cnp.intp_t[::1] channel1,
    const double[::1] values,
    Py_ssize_t channels,
    int num_threads=0,
):
    cdef:
        Py_ssize_t n_points = spatial_indices.shape[0]
        Py_ssize_t point

    if channels <= 0:
        raise ValueError("channels must be positive")
    if (
        channel0.shape[0] != n_points
        or channel1.shape[0] != n_points
        or values.shape[0] != n_points
    ):
        raise ValueError("all sparse injection arrays must have equal length")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for point in prange(n_points, schedule="static"):
                add_sparse_point_2d(
                    field,
                    spatial_indices,
                    channel0,
                    channel1,
                    values,
                    channels,
                    point,
                )
    else:
        with nogil:
            for point in prange(
                n_points, schedule="static", num_threads=num_threads
            ):
                add_sparse_point_2d(
                    field,
                    spatial_indices,
                    channel0,
                    channel1,
                    values,
                    channels,
                    point,
                )

cdef inline void filter_angular_point_2d(
    const double[::1] source,
    double[::1] destination,
    const double[::1] kernel,
    const cnp.intp_t[:, ::1] lookup,
    Py_ssize_t n_theta,
    Py_ssize_t spatial,
    double scale,
) noexcept nogil:
    cdef:
        Py_ssize_t theta, tap
        Py_ssize_t base = spatial * n_theta
        double value

    for theta in range(n_theta):
        value = 0.0
        for tap in range(kernel.shape[0]):
            value += kernel[tap] * source[base + lookup[theta, tap]]
        destination[base + theta] = value * scale


def angular_filter2d_into(
    const double[::1] source,
    double[::1] destination,
    const double[::1] kernel,
    const cnp.intp_t[:, ::1] lookup,
    Py_ssize_t n_spatial,
    Py_ssize_t n_theta,
    double scale=1.0,
    int num_threads=0,
):
    cdef:
        Py_ssize_t expected_size = n_spatial * n_theta
        Py_ssize_t spatial

    if n_spatial < 0 or n_theta <= 0:
        raise ValueError("n_spatial must be non-negative and n_theta must be positive")
    if source.shape[0] != expected_size or destination.shape[0] != expected_size:
        raise ValueError("source and destination sizes do not match")
    if lookup.shape[0] != n_theta or lookup.shape[1] != kernel.shape[0]:
        raise ValueError("angular lookup shape does not match the kernel")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    if num_threads == 0:
        with nogil:
            for spatial in prange(n_spatial, schedule="static"):
                filter_angular_point_2d(
                    source,
                    destination,
                    kernel,
                    lookup,
                    n_theta,
                    spatial,
                    scale,
                )
    else:
        with nogil:
            for spatial in prange(
                n_spatial, schedule="static", num_threads=num_threads
            ):
                filter_angular_point_2d(
                    source,
                    destination,
                    kernel,
                    lookup,
                    n_theta,
                    spatial,
                    scale,
                )

cdef inline void antipodal_reduce_point_2d(
    const double[::1] field,
    double[::1] completion,
    cnp.intp_t[::1] best_theta,
    Py_ssize_t n_theta,
    Py_ssize_t spatial,
    bint return_orientation,
) noexcept nogil:
    cdef:
        Py_ssize_t theta, opposite_theta
        Py_ssize_t base = spatial * n_theta
        Py_ssize_t half = n_theta // 2
        double product
        double total = 0.0
        double best_product = -DBL_MAX
        Py_ssize_t selected_theta = 0

    for theta in range(n_theta):
        opposite_theta = theta + half
        if opposite_theta >= n_theta:
            opposite_theta -= n_theta
        product = field[base + theta] * field[base + opposite_theta]
        total += product
        if return_orientation and product > best_product:
            best_product = product
            selected_theta = theta

    completion[spatial] = total
    if return_orientation:
        best_theta[spatial] = selected_theta

def antipodal_reduce2d(
    const double[::1] field,
    Py_ssize_t n_spatial,
    Py_ssize_t n_theta,
    bint return_orientation=False,
    int num_threads=0,
):
    cdef:
        Py_ssize_t expected_size = n_spatial * n_theta
        Py_ssize_t spatial
        cnp.ndarray[cnp.float64_t, ndim=1] completion_array
        cnp.ndarray[cnp.intp_t, ndim=1] best_theta_array
        double[::1] completion
        cnp.intp_t[::1] best_theta

    if n_spatial < 0 or n_theta <= 0 or n_theta % 2 != 0:
        raise ValueError("n_spatial must be non-negative and n_theta must be positive/even")
    if field.shape[0] != expected_size:
        raise ValueError("field size does not match the supplied dimensions")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")

    completion_array = np.empty(n_spatial, dtype=np.float64)
    if return_orientation:
        best_theta_array = np.empty(n_spatial, dtype=np.intp)
    else:
        best_theta_array = np.empty(1, dtype=np.intp)
    completion = completion_array
    best_theta = best_theta_array

    if num_threads == 0:
        with nogil:
            for spatial in prange(n_spatial, schedule="static"):
                antipodal_reduce_point_2d(
                    field,
                    completion,
                    best_theta,
                    n_theta,
                    spatial,
                    return_orientation,
                )
    else:
        with nogil:
            for spatial in prange(
                n_spatial, schedule="static", num_threads=num_threads
            ):
                antipodal_reduce_point_2d(
                    field,
                    completion,
                    best_theta,
                    n_theta,
                    spatial,
                    return_orientation,
                )

    if return_orientation:
        return completion_array, best_theta_array
    return completion_array
