import numpy as np


def otsu_threshold(values, bins=256):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer greater than one.")

    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= np.finfo(np.float64).eps:
        return minimum

    counts, edges = np.histogram(values, bins=bins, range=(minimum, maximum))
    centers = 0.5 * (edges[:-1] + edges[1:])

    count_low = np.cumsum(counts)[:-1]
    sum_low = np.cumsum(counts * centers)[:-1]
    count_high = counts.sum() - count_low
    sum_high = np.sum(counts * centers) - sum_low

    usable = (count_low > 0) & (count_high > 0)
    between_class_variance = np.full(bins - 1, -np.inf)
    mean_low = sum_low[usable] / count_low[usable]
    mean_high = sum_high[usable] / count_high[usable]
    between_class_variance[usable] = count_low[usable] * count_high[usable] * (mean_low - mean_high) ** 2
    best_split = int(np.argmax(between_class_variance))

    return float(edges[best_split + 1])

def local_otsu_2d(data, valid_mask=None, lower_threshold=3.0, upper_threshold=7.0, split_fraction=0.5, transition_width=24.0, alpha=0.2, delta=1.0):
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("The data must be a 2d array")
    rows, cols = data.shape
    if min(rows, cols) < 3:
        raise ValueError("The data must be at aleast 3 by 3")
    parameters = [lower_threshold, upper_threshold, split_fraction, transition_width, alpha, delta]
    if not np.isfinite(parameters).all():
        raise ValueError("threshold parameters must be finite")
    if upper_threshold <= lower_threshold:
        raise ValueError("upper_threshold must be greater than lower_threshold")
    if not 0 <= split_fraction <= 1:
        raise ValueError("split_fraction must be between 0 and 1")
    if transition_width < 0:
        raise ValueError("transition_width must be non-negative")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if delta < 0:
        raise ValueError("delta must be non-negative")

    patch_size = 64
    stride = 32
    blend_sigma = 16.0
    global_weight = 0.25
    min_class_fraction = 0.05
    min_valid_fraction = 0.25

    score = np.log1p(np.maximum(data, 0.0))

    valid = np.isfinite(score)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != data.shape:
            raise ValueError("valid_mask must have the same shape as the data")
        valid &= valid_mask

    valid[[0, - 1], :] = False
    valid[:, [0, -1]] = False

    global_values = score[valid]
    if global_values.size == 0:
        raise ValueError("No valid pixels are available for thresholding")

    global_threshold = otsu_threshold(global_values)

    patch_rows = min(patch_size, rows)
    patch_cols = min(patch_size, cols)

    last_row = rows - patch_rows
    row_starts = list(range(0, last_row + 1, stride))
    if row_starts[-1] != last_row:
        row_starts.append(last_row)

    last_col = cols - patch_cols
    col_starts = list(range(0, last_col + 1, stride))
    if col_starts[-1] != last_col:
        col_starts.append(last_col)

    local_numerator = np.zeros_like(score)
    local_denominator = np.zeros_like(score)
    radius = int(np.ceil(3.0 * blend_sigma))

    for row0 in row_starts:
        row1 = row0 + patch_rows
        for col0 in col_starts:
            col1 = col0 + patch_cols

            patch_valid = valid[row0:row1, col0:col1]
            if patch_valid.mean() < min_valid_fraction:
                continue
            values = score[row0:row1, col0:col1][patch_valid]

            local_threshold = otsu_threshold(values)
            low_fraction = float(np.mean(values < local_threshold))
            high_fraction = 1.0 - low_fraction

            if min(low_fraction, high_fraction) < min_class_fraction:
                continue
            reliability = 4.0 * low_fraction * high_fraction

            center_row = 0.5 * (row0 + row1 - 1)
            center_col = 0.5 * (col0 + col1 - 1)
            blend_row0 = max(0, int(np.floor(center_row - radius)))
            blend_row1 = min(rows, int(np.ceil(center_row + radius)) + 1)
            blend_col0 = max(0, int(np.floor(center_col - radius)))
            blend_col1 = min(cols, int(np.ceil(center_col + radius)) + 1)

            rr = np.arange(blend_row0, blend_row1)[:, None]
            cc = np.arange(blend_col0, blend_col1)[None, :]
            distance_squared = (rr - center_row) ** 2 + (cc - center_col) ** 2
            weight = reliability * np.exp(
                -distance_squared / (2.0 * blend_sigma**2)
            )

            local_numerator[
                blend_row0:blend_row1, blend_col0:blend_col1
            ] += weight * local_threshold
            local_denominator[
                blend_row0:blend_row1, blend_col0:blend_col1
            ] += weight

    threshold_log = (
        global_weight * global_threshold + local_numerator
    ) / (global_weight + local_denominator)
    otsu_threshold_map = np.expm1(threshold_log)

    column = np.arange(cols, dtype=np.float64)
    split = split_fraction * (cols - 1)

    if transition_width == 0:
        spatial_weight = (column >= split).astype(np.float64)
    else:
        u = (column - (split - transition_width / 2.0)) / transition_width
        u = np.clip(u, 0.0, 1.0)
        spatial_weight = u * u * (3.0 - 2.0 * u)

    threshold_line = lower_threshold + (
        upper_threshold - lower_threshold
    ) * spatial_weight

    bounded_otsu = np.clip(
        otsu_threshold_map,
        threshold_line - delta,
        threshold_line + delta,
    )
    threshold_map = (
        (1.0 - alpha) * threshold_line
        + alpha * bounded_otsu
    )

    thresholded_mask = (data < threshold_map) & valid
    return thresholded_mask, threshold_map
