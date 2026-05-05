import numpy as np
from dipy.viz import skyline
from numpy.lib.stride_tricks import sliding_window_view

from thetan.berserker import cluster_merge

try:
    from scipy.ndimage import median_filter as _median_filter
except Exception:
    def _median_filter(arr, size=3):
        return arr


_MISSING = object()

def make_case(
    name,
    description,
    data,
    labels,
    affine,
    base_intensities,
    threshold,
    max_intensity_threshold,
    expected_groups,
):
    return {
        "name": name,
        "description": description,
        "data": data,
        "labels": labels,
        "affine": affine,
        "base_intensities": base_intensities,
        "threshold": threshold,
        "max_intensity_threshold": max_intensity_threshold,
        "expected_groups": expected_groups,
    }


def copy_case_with_changes(case, changes):
    new_case = dict(case)

    for key, value in changes.items():
        new_case[key] = value

    return new_case

def make_affine(voxel_size=(1.0, 1.0, 1.0)):
    affine = np.eye(4, dtype=float)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    return affine


def paint_ball(labels, center_zyx, radius, label):
    """
    Paint a small 3D ball into an integer label volume.

    center_zyx:
        z, y, x
    """
    zc, yc, xc = center_zyx
    shape = labels.shape

    z0 = max(0, int(np.floor(zc - radius)))
    z1 = min(shape[0] - 1, int(np.ceil(zc + radius)))

    y0 = max(0, int(np.floor(yc - radius)))
    y1 = min(shape[1] - 1, int(np.ceil(yc + radius)))

    x0 = max(0, int(np.floor(xc - radius)))
    x1 = min(shape[2] - 1, int(np.ceil(xc + radius)))

    if z0 > z1 or y0 > y1 or x0 > x1:
        return

    zz, yy, xx = np.ogrid[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]

    mask = (
        (zz - zc) ** 2
        + (yy - yc) ** 2
        + (xx - xc) ** 2
        <= radius ** 2
    )

    sub = labels[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
    sub[mask] = int(label)


def paint_line_tube(labels, p0, p1, radius, label, points_per_voxel=1.25):
    """
    Paint a cylindrical tube along a line segment.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    distance = np.linalg.norm(p1 - p0)
    n_points = max(2, int(distance * points_per_voxel))

    for t in np.linspace(0.0, 1.0, n_points):
        p = (1.0 - t) * p0 + t * p1
        paint_ball(labels, p, radius, label)


def helix_points(
    center_yx=(40.0, 40.0),
    helix_radius=18.0,
    z_range=(8.0, 72.0),
    turns=4.0,
    phase=0.0,
    n_points=1000,
):
    """
    Return 3D helix points in z, y, x order.
    """
    t = np.linspace(0.0, 2.0 * np.pi * turns, n_points)

    z = np.linspace(z_range[0], z_range[1], n_points)
    y = center_yx[0] + helix_radius * np.sin(t + phase)
    x = center_yx[1] + helix_radius * np.cos(t + phase)

    return np.column_stack([z, y, x])


def paint_tube_segments(labels, points, n_segments, tube_radius, first_label):
    """
    Paint one continuous tube split into consecutive labels.
    """
    points = np.asarray(points, dtype=float)
    breaks = np.linspace(0, len(points) - 1, n_segments + 1).astype(int)

    for seg_idx in range(n_segments):
        label = first_label + seg_idx
        start = breaks[seg_idx]
        end = breaks[seg_idx + 1]

        for p in points[start:end + 1]:
            paint_ball(labels, p, tube_radius, label)


def sphere_mask(shape, center, radius):
    zc, yc, xc = center
    zz, yy, xx = np.indices(shape)

    return (
        (zz - zc) ** 2
        + (yy - yc) ** 2
        + (xx - xc) ** 2
        <= radius ** 2
    )


def cylinder_z_mask(shape, center_yx, radius, z_range):
    yc, xc = center_yx
    z0, z1 = z_range

    zz, yy, xx = np.indices(shape)

    return (
        ((yy - yc) ** 2 + (xx - xc) ** 2 <= radius ** 2)
        & (zz >= z0)
        & (zz < z1)
    )


def cube_mask(shape, start, end):
    z0, y0, x0 = start
    z1, y1, x1 = end

    mask = np.zeros(shape, dtype=bool)
    mask[z0:z1, y0:y1, x0:x1] = True

    return mask


def make_textured_data(labels, base_intensities, seed=0, texture_amplitude=0.35):
    """
    Convert integer labels into floating intensity data.

    Each label keeps its requested mean intensity, but gets tiny zero-mean
    texture. This is important for synthetic data because the neighbor-pair
    function skips patches with extremely low data variance.
    """
    labels = np.asarray(labels, dtype=np.int64)

    zz, yy, xx = np.indices(labels.shape)

    deterministic_pattern = (
        np.sin(0.173 * zz + 0.071 * yy + 0.113 * xx)
        + 0.6 * np.cos(0.097 * zz - 0.191 * yy + 0.053 * xx)
        + 0.4 * np.sin(0.041 * zz + 0.137 * yy - 0.149 * xx)
    )

    rng = np.random.default_rng(seed)
    random_pattern = rng.normal(0.0, 1.0, size=labels.shape)

    pattern = deterministic_pattern + 0.15 * random_pattern
    pattern = pattern / max(pattern.std(), 1e-12)

    data = np.zeros(labels.shape, dtype=np.float64)

    for label, intensity in base_intensities.items():
        label = int(label)
        mask = labels == label

        if not np.any(mask):
            continue

        local_texture = texture_amplitude * pattern[mask]
        local_texture = local_texture - local_texture.mean()

        data[mask] = float(intensity) + local_texture

    np.maximum(data, 0.0, out=data)

    return np.ascontiguousarray(data, dtype=np.float64)

def patchify_numpy(arr, patch_size):
    """
    Test version of patchify.

    For a 3D array, this returns:
        (Z-pz+1, Y-py+1, X-px+1, pz, py, px)
    """
    patches = sliding_window_view(arr, patch_size)
    return np.ascontiguousarray(patches)


def compute_centroids_from_labels(labels, W):
    """
    Return centroids as:
        [mean_z, mean_y, mean_x, mean_intensity]
    """
    labels = np.asarray(labels, dtype=np.int64)
    W = np.asarray(W, dtype=np.float64)

    n_labels = int(labels.max()) + 1
    centroids = np.zeros((n_labels, 4), dtype=np.float64)

    for label in range(n_labels):
        mask = labels == label

        if not np.any(mask):
            continue

        coords = np.argwhere(mask)

        centroids[label, 0] = coords[:, 0].mean()
        centroids[label, 1] = coords[:, 1].mean()
        centroids[label, 2] = coords[:, 2].mean()
        centroids[label, 3] = W[mask].mean()

    return centroids


def centroid_images_numpy(labels, centroids):
    """
    Simple centroid image generator for visual tests.
    """
    labels = np.asarray(labels)
    centroids = np.asarray(centroids, dtype=np.float64)

    out = np.zeros(labels.shape, dtype=np.float64)

    for label in range(centroids.shape[0]):
        z, y, x, intensity = centroids[label]

        zi = int(round(z))
        yi = int(round(y))
        xi = int(round(x))

        if (
            0 <= zi < labels.shape[0]
            and 0 <= yi < labels.shape[1]
            and 0 <= xi < labels.shape[2]
        ):
            out[zi, yi, xi] = intensity

    return out


def patch_cluster_merge_frontend(forced_labels, forced_W):
    """
    Temporarily feed known initial labels into cluster_merge.

    This function returns a backups dictionary. Always call
    restore_cluster_merge_frontend(backups) in a finally block.

    This does not change the cluster_merge source file. It only changes the
    global function references used by the imported cluster_merge function
    during this test call.
    """
    forced_labels = np.ascontiguousarray(forced_labels.copy(), dtype=np.int64)
    forced_W = np.ascontiguousarray(forced_W.copy(), dtype=np.float64)

    g = cluster_merge.__globals__

    real_merge_helpers = [
        "find_neighbor_pairs",
        "compute_cluster_stats",
        "UnionFind",
        "remap_labels",
    ]

    missing_helpers = []

    for name in real_merge_helpers:
        if name not in g:
            missing_helpers.append(name)

    if missing_helpers:
        raise RuntimeError(
            "cluster_merge cannot find these merge helpers in its globals: "
            + ", ".join(missing_helpers)
        )

    def fake_prepare_data_3d(
        data,
        affine,
        perc=(0, 98),
        mult=300.0,
        enh=3.0,
    ):
        return data

    def fake_X_to_many_3D(X, shape):
        Ivol = None
        Jvol = None
        Kvol = None
        W = forced_W
        return Ivol, Jvol, Kvol, W

    def fake_berserker2_to_3D_plus_extra(X, W, theta):
        labels = np.ascontiguousarray(forced_labels.copy(), dtype=np.int64)

        centroids = compute_centroids_from_labels(labels, W)
        centroids_im = centroid_images_numpy(labels, centroids)
        centroids_median_im = _median_filter(centroids_im, size=3)

        return labels, centroids, centroids_im, centroids_median_im

    patch_values = {
        "prepare_data_3d": fake_prepare_data_3d,
        "X_to_many_3D": fake_X_to_many_3D,
        "berserker2_to_3D_plus_extra": fake_berserker2_to_3D_plus_extra,
        "patchify": patchify_numpy,
        "centroid_images": centroid_images_numpy,
    }

    if "median_filter" not in g:
        patch_values["median_filter"] = _median_filter

    if "np" not in g:
        patch_values["np"] = np

    backups = {}

    for name, value in patch_values.items():
        backups[name] = g.get(name, _MISSING)
        g[name] = value

    return backups


def restore_cluster_merge_frontend(backups):
    """
    Restore the globals changed by patch_cluster_merge_frontend.
    """
    g = cluster_merge.__globals__

    for name, old_value in backups.items():
        if old_value is _MISSING:
            del g[name]
        else:
            g[name] = old_value

def get_image(images, name):
    for arr, affine, image_name in images:
        if image_name == name:
            return arr

    raise KeyError("Could not find image named %r" % (name,))


def add_extra_visual_images(images, case):
    """
    Add visual debug images after cluster_merge returns.

    This does not change cluster_merge.
    """
    images = list(images)

    init = get_image(images, "data_labels_initial")
    merged = get_image(images, "data_labels_merged")

    changed = (init != merged).astype(np.int64)

    images.append(
        (
            np.ascontiguousarray(case["labels"], dtype=np.int64),
            case["affine"],
            "forced_initial_labels_expected",
        )
    )

    images.append(
        (
            changed,
            case["affine"],
            "changed_voxels_initial_vs_merged",
        )
    )

    return images


def print_case_header(case):
    print("\n" + "=" * 100)
    print("CASE: %s" % case["name"])
    print("=" * 100)
    print(case["description"])
    print()
    print("threshold: %s" % case["threshold"])
    print("max_intensity_threshold: %s" % case["max_intensity_threshold"])

    print("\nBase intensities:")

    for label in sorted(case["base_intensities"]):
        print("    label %2d: %8.3f" % (label, case["base_intensities"][label]))

    print("\nExpected groups after merging:")

    for group in case["expected_groups"]:
        print("    %s" % (tuple(group),))

    print("=" * 100)


def print_label_transition_report(images, case):
    init = get_image(images, "data_labels_initial")
    merged = get_image(images, "data_labels_merged")

    print("\nMERGE VISUAL CHECK")
    print("-" * 100)
    print(
        "array_equal(data_labels_initial, data_labels_merged):",
        np.array_equal(init, merged),
    )
    print("changed voxels:", int(np.count_nonzero(init != merged)))

    print("\nInitial labels:")
    init_labels, init_counts = np.unique(init, return_counts=True)

    for label, count in zip(init_labels, init_counts):
        print("    label %2d: %8d voxels" % (int(label), int(count)))

    print("\nMerged labels:")
    merged_labels, merged_counts = np.unique(merged, return_counts=True)

    for label, count in zip(merged_labels, merged_counts):
        print("    label %2d: %8d voxels" % (int(label), int(count)))

    print("\nOld label -> new label transition:")

    for old_label in init_labels:
        mask = init == old_label
        new_values, new_counts = np.unique(merged[mask], return_counts=True)

        parts = []

        for new_label, count in zip(new_values, new_counts):
            parts.append("%d(%d)" % (int(new_label), int(count)))

        print("    old label %2d -> %s" % (int(old_label), ", ".join(parts)))

    print("\nExpected group report:")

    used_targets = {}

    for group in case["expected_groups"]:
        merged_targets = set()

        for old_label in group:
            mask = init == old_label

            if not np.any(mask):
                continue

            for x in np.unique(merged[mask]):
                merged_targets.add(int(x))

        if len(merged_targets) == 1:
            status = "OK: group merged into one label"
        else:
            status = "CHECK: group split across multiple merged labels"

        print(
            "    expected group %s -> merged labels %s -> %s"
            % (tuple(group), sorted(merged_targets), status)
        )

        for target in merged_targets:
            if target not in used_targets:
                used_targets[target] = []
            used_targets[target].append(tuple(group))

    print("\nCross-group collision check:")

    any_collision = False

    for target, groups in used_targets.items():
        if len(groups) > 1:
            any_collision = True
            print(
                "    CHECK: merged label %d is shared by expected groups: %s"
                % (target, groups)
            )

    if not any_collision:
        print("    OK: expected groups did not collapse into each other.")


def require_consecutive_labels(case):
    """
    These tests intentionally use labels 0..N with no gaps.
    That keeps them compatible with the current remapping code in cluster_merge.
    """
    unique_labels = np.unique(case["labels"])
    expected = np.arange(int(unique_labels.max()) + 1, dtype=unique_labels.dtype)

    if not np.array_equal(unique_labels, expected):
        raise RuntimeError(
            "Case %s has non-consecutive labels.\nFound:    %s\nExpected: %s"
            % (case["name"], unique_labels, expected)
        )

def make_case_segmented_helix():
    """
    Continuous helix split into 8 labels.

    Expected with threshold=10 and max_intensity_threshold=20:
        labels 1,2,3,4 merge
        labels 5,6 merge
        labels 7 and 8 stay separate
    """
    shape = (80, 80, 80)
    labels = np.zeros(shape, dtype=np.int64)

    points = helix_points(
        center_yx=(40.0, 40.0),
        helix_radius=19.0,
        z_range=(7.0, 73.0),
        turns=4.25,
        n_points=1400,
    )

    paint_tube_segments(
        labels,
        points,
        n_segments=8,
        tube_radius=2.35,
        first_label=1,
    )

    base_intensities = {
        1: 80.0,
        2: 80.0,
        3: 84.0,
        4: 88.0,
        5: 111.0,
        6: 111.0,
        7: 133.0,
        8: 150.0,
    }

    data = make_textured_data(
        labels,
        base_intensities,
        seed=11,
        texture_amplitude=0.35,
    )

    return make_case(
        name="segmented_helix",
        description=(
            "One continuous 3D helix split into 8 labels. It has same-intensity "
            "neighbors, close-intensity neighbors, and far-intensity neighbors."
        ),
        data=data,
        labels=labels,
        affine=make_affine(),
        base_intensities=base_intensities,
        threshold=10.0,
        max_intensity_threshold=20.0,
        expected_groups=[
            (1, 2, 3, 4),
            (5, 6),
            (7,),
            (8,),
        ],
    )


def make_case_double_helix_with_bridges():
    """
    Two intertwined helices plus bridge labels.

    Expected:
        labels 1,2,3,4,9 merge
        labels 5,6,7,8,10 merge
        those two groups stay separate
    """
    shape = (80, 80, 80)
    labels = np.zeros(shape, dtype=np.int64)

    points_a = helix_points(
        center_yx=(40.0, 40.0),
        helix_radius=15.5,
        z_range=(8.0, 72.0),
        turns=3.75,
        phase=0.0,
        n_points=1100,
    )

    points_b = helix_points(
        center_yx=(40.0, 40.0),
        helix_radius=15.5,
        z_range=(8.0, 72.0),
        turns=3.75,
        phase=np.pi,
        n_points=1100,
    )

    paint_tube_segments(
        labels,
        points_a,
        n_segments=4,
        tube_radius=2.15,
        first_label=1,
    )

    paint_tube_segments(
        labels,
        points_b,
        n_segments=4,
        tube_radius=2.15,
        first_label=5,
    )

    # Three bridges. Each bridge is split into two labels so the bridge halves
    # are close to their own helix but far from each other.
    for idx in [230, 550, 850]:
        p_a = points_a[idx]
        p_b = points_b[idx]
        mid = 0.5 * (p_a + p_b)

        paint_line_tube(
            labels,
            p_a,
            mid,
            radius=1.8,
            label=9,
            points_per_voxel=1.4,
        )

        paint_line_tube(
            labels,
            mid,
            p_b,
            radius=1.8,
            label=10,
            points_per_voxel=1.4,
        )

    base_intensities = {
        1: 55.0,
        2: 55.0,
        3: 59.0,
        4: 63.0,
        5: 104.0,
        6: 108.0,
        7: 108.0,
        8: 112.0,
        9: 59.0,
        10: 108.0,
    }

    data = make_textured_data(
        labels,
        base_intensities,
        seed=22,
        texture_amplitude=0.35,
    )

    return make_case(
        name="double_helix_with_bridges",
        description=(
            "Two intertwined helices with split bridge labels. Each helix should "
            "merge with its matching bridge half, but the two helices should not "
            "collapse into each other."
        ),
        data=data,
        labels=labels,
        affine=make_affine(),
        base_intensities=base_intensities,
        threshold=9.0,
        max_intensity_threshold=16.0,
        expected_groups=[
            (1, 2, 3, 4, 9),
            (5, 6, 7, 8, 10),
        ],
    )


def make_case_nested_shells_with_spiral():
    """
    Nested spherical shells with a spiral tube through them.

    Expected:
        labels 1,2 merge
        label 3 stays separate
        labels 4,5,6 merge
        label 7 stays separate
    """
    shape = (80, 80, 80)
    labels = np.zeros(shape, dtype=np.int64)

    zz, yy, xx = np.indices(shape)

    center = np.array([40.0, 40.0, 40.0])

    r = np.sqrt(
        (zz - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (xx - center[2]) ** 2
    )

    labels[(r >= 10.0) & (r < 16.0)] = 1
    labels[(r >= 16.0) & (r < 22.0)] = 2
    labels[(r >= 22.0) & (r < 28.0)] = 3

    spiral = helix_points(
        center_yx=(40.0, 40.0),
        helix_radius=10.0,
        z_range=(8.0, 72.0),
        turns=5.25,
        phase=0.4,
        n_points=1400,
    )

    # Spiral overwrites shell labels where it passes through them.
    paint_tube_segments(
        labels,
        spiral,
        n_segments=4,
        tube_radius=2.25,
        first_label=4,
    )

    base_intensities = {
        1: 70.0,
        2: 72.0,
        3: 91.0,
        4: 130.0,
        5: 130.0,
        6: 134.0,
        7: 148.0,
    }

    data = make_textured_data(
        labels,
        base_intensities,
        seed=33,
        texture_amplitude=0.35,
    )

    return make_case(
        name="nested_shells_with_spiral",
        description=(
            "Nested shell boundaries plus an embedded spiral. This tests curved "
            "surfaces, shell/spiral contact, same-intensity spiral segments, and "
            "intensity barriers."
        ),
        data=data,
        labels=labels,
        affine=make_affine(),
        base_intensities=base_intensities,
        threshold=6.0,
        max_intensity_threshold=12.0,
        expected_groups=[
            (1, 2),
            (3,),
            (4, 5, 6),
            (7,),
        ],
    )


def make_case_branching_crossing_vessels():
    """
    Branching vessel-like tubes plus a high-intensity crossing object.

    Expected:
        labels 1,2,3 merge
        labels 4,5 merge
        labels 6,7 merge
        the three groups stay separate
    """
    shape = (80, 80, 80)
    labels = np.zeros(shape, dtype=np.int64)

    paint_line_tube(
        labels,
        p0=(8, 40, 40),
        p1=(26, 40, 40),
        radius=2.4,
        label=1,
    )

    paint_line_tube(
        labels,
        p0=(26, 40, 40),
        p1=(44, 40, 40),
        radius=2.4,
        label=2,
    )

    paint_line_tube(
        labels,
        p0=(44, 40, 40),
        p1=(68, 24, 40),
        radius=2.4,
        label=3,
    )

    # A second branch touches the main branch spatially, but should not merge
    # because its intensity is much higher.
    paint_line_tube(
        labels,
        p0=(46, 42, 40),
        p1=(62, 56, 40),
        radius=2.4,
        label=4,
    )

    paint_line_tube(
        labels,
        p0=(62, 56, 40),
        p1=(74, 66, 54),
        radius=2.4,
        label=5,
    )

    # High-intensity diagonal object crossing the scene without controlling the
    # low-intensity branch junction.
    paint_line_tube(
        labels,
        p0=(14, 18, 60),
        p1=(72, 62, 18),
        radius=2.15,
        label=6,
    )

    blob = sphere_mask(
        shape,
        center=(48, 44, 36),
        radius=5,
    )

    labels[blob] = 7

    base_intensities = {
        1: 40.0,
        2: 40.0,
        3: 44.0,
        4: 75.0,
        5: 75.0,
        6: 120.0,
        7: 123.0,
    }

    data = make_textured_data(
        labels,
        base_intensities,
        seed=44,
        texture_amplitude=0.35,
    )

    return make_case(
        name="branching_crossing_vessels",
        description=(
            "A Y-shaped branching tube network plus a diagonal high-intensity "
            "crossing object and blob. This is a hard geometry/contact case."
        ),
        data=data,
        labels=labels,
        affine=make_affine(),
        base_intensities=base_intensities,
        threshold=6.0,
        max_intensity_threshold=10.0,
        expected_groups=[
            (1, 2, 3),
            (4, 5),
            (6, 7),
        ],
    )

def run_case(case, individual_labels=True, show=True):
    require_consecutive_labels(case)
    print_case_header(case)

    data_for_call = np.ascontiguousarray(case["data"].copy(), dtype=np.float64)
    forced_W = np.ascontiguousarray(case["data"].copy(), dtype=np.float64)
    forced_labels = np.ascontiguousarray(case["labels"].copy(), dtype=np.int64)

    backups = patch_cluster_merge_frontend(
        forced_labels=forced_labels,
        forced_W=forced_W,
    )

    try:
        images = cluster_merge(
            data=data_for_call,
            affine=case["affine"],
            theta=80,
            threshold=case["threshold"],
            max_intensity_threshold=case["max_intensity_threshold"],
            patch_size=(3, 3, 3),
            individual_labels=individual_labels,
            verbose=True,
        )
    finally:
        restore_cluster_merge_frontend(backups)

    images = add_extra_visual_images(images, case)

    print_label_transition_report(images, case)

    if show:
        skyline(images=images)

    return images


def keep_key_images_with_prefix(images, prefix):
    keep_names = {
        "data",
        "W",
        "data_labels_initial",
        "data_labels_merged",
        "forced_initial_labels_expected",
        "changed_voxels_initial_vs_merged",
    }

    out = []

    for arr, affine, name in images:
        if name in keep_names:
            out.append((arr, affine, "%s_%s" % (prefix, name)))

    return out


def run_threshold_sweep_on_segmented_helix():
    """
    Same helix, two thresholds.

    Low threshold=1:
        same-intensity neighbors should merge only.

    Higher threshold=10:
        same-intensity and close-intensity neighbors should merge.
    """
    base_case = make_case_segmented_helix()

    low_case = copy_case_with_changes(
        base_case,
        {
            "name": "segmented_helix_threshold_1",
            "threshold": 1.0,
            "expected_groups": [
                (1, 2),
                (3,),
                (4,),
                (5, 6),
                (7,),
                (8,),
            ],
        },
    )

    high_case = copy_case_with_changes(
        base_case,
        {
            "name": "segmented_helix_threshold_10",
            "threshold": 10.0,
            "expected_groups": [
                (1, 2, 3, 4),
                (5, 6),
                (7,),
                (8,),
            ],
        },
    )

    print("\n" + "=" * 100)
    print("THRESHOLD SWEEP ON THE SAME SEGMENTED HELIX")
    print("=" * 100)

    images_low = run_case(
        low_case,
        individual_labels=False,
        show=False,
    )

    images_high = run_case(
        high_case,
        individual_labels=False,
        show=False,
    )

    comparison_images = []

    comparison_images.extend(
        keep_key_images_with_prefix(
            images_low,
            "threshold_1",
        )
    )

    comparison_images.extend(
        keep_key_images_with_prefix(
            images_high,
            "threshold_10",
        )
    )

    skyline(images=comparison_images)

    return images_low, images_high


def run_all_visual_tests(individual_labels=True):
    results = {}

    cases = [
        make_case_segmented_helix(),
        make_case_double_helix_with_bridges(),
        make_case_nested_shells_with_spiral(),
        make_case_branching_crossing_vessels(),
    ]

    for case in cases:
        results[case["name"]] = run_case(
            case,
            individual_labels=individual_labels,
            show=True,
        )

    results["threshold_sweep_segmented_helix"] = (
        run_threshold_sweep_on_segmented_helix()
    )

    return results


def run_one_visual_test(case_name, individual_labels=True):
    """
    Run one case by name.

    Valid names:
        segmented_helix
        double_helix_with_bridges
        nested_shells_with_spiral
        branching_crossing_vessels
        threshold_sweep
    """
    if case_name == "segmented_helix":
        return run_case(
            make_case_segmented_helix(),
            individual_labels=individual_labels,
            show=True,
        )

    if case_name == "double_helix_with_bridges":
        return run_case(
            make_case_double_helix_with_bridges(),
            individual_labels=individual_labels,
            show=True,
        )

    if case_name == "nested_shells_with_spiral":
        return run_case(
            make_case_nested_shells_with_spiral(),
            individual_labels=individual_labels,
            show=True,
        )

    if case_name == "branching_crossing_vessels":
        return run_case(
            make_case_branching_crossing_vessels(),
            individual_labels=individual_labels,
            show=True,
        )

    if case_name == "threshold_sweep":
        return run_threshold_sweep_on_segmented_helix()

    raise ValueError("Unknown case_name: %s" % case_name)


if __name__ == "__main__":
    run_all_visual_tests(individual_labels=True)