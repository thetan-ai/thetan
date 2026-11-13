import numpy as np
import numpy.testing as npt

from thetan.metrics import distance_matrix


def euclidean_ground(x,y):
    return np.linalg.norm(x-y)

def ssd_ground(x,y):
    return np.sum((x-y)**2)

def norm_ssd_ground(x,y):
    return np.mean((x-y)**2)

def neg_dot_ground(x,y):
    return -np.dot(x,y)

def l1_ground(x,y):
    return np.sum(np.abs(x-y))

def inf_norm_ground(x,y):
    return np.max(np.abs(x-y))

def cosine_ground(x,y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 and norm_y == 0:
        return 0.0
    if norm_x == 0 or norm_y == 0:
        return 1.0
    sim = np.dot(x,y) / (norm_x * norm_y)
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim

def arccos_ground(x,y):
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    if norm_x == 0 and norm_y == 0:
        return 0.0
    if norm_x == 0 or norm_y == 0:
        return np.pi / 2
    sim = np.dot(x,y) / (norm_x * norm_y)
    sim = np.clip(sim, -1.0, 1.0)
    return np.arccos(sim)

ground_metrics = {
    'euclidean': euclidean_ground,
    'ssd':ssd_ground,
    'norm_ssd': norm_ssd_ground,
    'norm_ssd_un': norm_ssd_ground,
    'dot_prod' : neg_dot_ground,
    'l1_norm' : l1_ground,
    'inf_norm' : inf_norm_ground,
    'cosine' : cosine_ground,
    'arccos': arccos_ground
}

def ground_distance_matrix(X1, X2, metric_name):
    x,y = X1.shape[0], X2.shape[0]
    mat = np.zeros((x,y))
    func = ground_metrics[metric_name]
    for i in range(x):
        for j in range(y):
            mat[i,j] = func(X1[i], X2[j])
    return mat

def test_distance_matrix():
    rng = np.random.default_rng(42)
    # several column counts (norm_ssd_un reads them four at a time), values
    # of both signs, and different numbers of rows in X1 and X2
    for n_cols in [1, 2, 3, 4, 5, 8, 10, 33]:
        X1 = rng.normal(size=(5, n_cols))
        X2 = rng.normal(size=(7, n_cols))
        for metric in ground_metrics:
            expected = ground_distance_matrix(X1, X2, metric)
            actual = distance_matrix(X1, X2, metric)
            npt.assert_equal(actual.shape, (5, 7))
            npt.assert_allclose(actual, expected, err_msg=f'{metric}, {n_cols} columns')

def test_known_values():
    # worked out by hand: x - y = [-3, -4, 0], x.y = 25, |x|^2 = 14, |y|^2 = 61
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[4.0, 6.0, 3.0]])
    expected = {'euclidean': 5.0, 'ssd': 25.0, 'norm_ssd': 25.0 / 3, 'norm_ssd_un': 25.0 / 3,
                'dot_prod': -25.0, 'l1_norm': 7.0, 'inf_norm': 4.0,
                'cosine': 1 - 25 / np.sqrt(14 * 61), 'arccos': np.arccos(25 / np.sqrt(14 * 61))}
    assert set(expected) == set(ground_metrics)
    for metric, value in expected.items():
        npt.assert_allclose(distance_matrix(x, y, metric)[0, 0], value, rtol=1e-14, err_msg=metric)
    # vectors more than a right angle apart: x - y = [2, -1], x.y = -1
    x = np.array([[1.0, 0.0]])
    y = np.array([[-1.0, 1.0]])
    expected = {'euclidean': np.sqrt(5), 'ssd': 5.0, 'norm_ssd': 2.5, 'norm_ssd_un': 2.5,
                'dot_prod': 1.0, 'l1_norm': 3.0, 'inf_norm': 2.0,
                'cosine': 1 + 1 / np.sqrt(2), 'arccos': 3 * np.pi / 4}
    for metric, value in expected.items():
        npt.assert_allclose(distance_matrix(x, y, metric)[0, 0], value, rtol=1e-14, err_msg=metric)

def test_default_metric_is_norm_ssd():
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(4,6))
    X2 = rng.normal(size=(3,6))
    npt.assert_array_equal(distance_matrix(X1, X2), distance_matrix(X1, X2, 'norm_ssd'))

def test_inf_norm_identical_vectors_is_zero():
    rng = np.random.default_rng(0)
    X = rng.random((5,10)).astype(np.float64)
    npt.assert_equal(np.diag(distance_matrix(X, X, 'inf_norm')), 0.0)

def test_nan_gives_nan_for_every_metric():
    x = np.array([[np.nan, 1.0, 5.0]])
    y = np.array([[1.0, 2.0, 3.0]])
    for metric in ground_metrics:
        assert np.isnan(distance_matrix(x, y, metric)[0, 0]), metric
        assert np.isnan(distance_matrix(y, x, metric)[0, 0]), metric
    # NaN that comes after a larger difference was already seen
    x = np.array([[9.0, np.nan, 1.0]])
    assert np.isnan(distance_matrix(x, y, 'inf_norm')[0, 0])
    # a row that is all NaN is not 0 away from everything
    x = np.full((1, 3), np.nan)
    assert np.isnan(distance_matrix(x, y, 'inf_norm')[0, 0])

def test_cosine_and_arccos_with_zero_vectors():
    rng = np.random.default_rng(0)
    X = np.vstack([np.zeros((2,10)), rng.random((3,10))])
    for metric in ['cosine', 'arccos']:
        expected = ground_distance_matrix(X, X, metric)
        actual = distance_matrix(X, X, metric)
        npt.assert_allclose(actual[:2], expected[:2])
        npt.assert_allclose(actual[:, :2], expected[:, :2])
    npt.assert_equal(distance_matrix(X[:1], X[2:], 'cosine'), 1.0)
    npt.assert_equal(distance_matrix(X[:1], X[2:], 'arccos'), np.pi / 2)
    # two zero vectors are exactly 0 apart
    for metric in ['cosine', 'arccos']:
        npt.assert_equal(distance_matrix(X[:2], X[:2], metric), 0.0)

def test_arccos_identical_opposite_and_small_angles():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200,13)) * rng.uniform(0.1, 100, (200,1))
    # identical vectors are exactly 0 apart, opposite vectors exactly pi
    npt.assert_equal(np.diag(distance_matrix(X, X, 'arccos')), 0.0)
    npt.assert_equal(np.diag(distance_matrix(X, -X, 'arccos')), np.pi)
    # small angles keep their accuracy
    for angle in [1e-3, 1e-6, 1e-9]:
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[np.cos(angle), np.sin(angle), 0.0]])
        npt.assert_allclose(distance_matrix(a, b, 'arccos')[0, 0], angle, rtol=1e-6)

def test_cosine_identical_opposite_and_small_angles():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200,13)) * rng.uniform(0.1, 100, (200,1))
    # identical vectors are exactly 0 apart, opposite vectors 2 apart
    npt.assert_equal(np.diag(distance_matrix(X, X, 'cosine')), 0.0)
    npt.assert_allclose(np.diag(distance_matrix(X, -X, 'cosine')), 2.0, rtol=1e-15)
    # small angles keep their accuracy: 1 - cos(angle) = 2 * sin(angle / 2)**2
    for angle in [1e-3, 1e-6, 1e-9]:
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[np.cos(angle), np.sin(angle), 0.0]])
        npt.assert_allclose(distance_matrix(a, b, 'cosine')[0, 0], 2 * np.sin(angle / 2)**2, rtol=1e-6)

def test_cosine_and_arccos_do_not_depend_on_vector_length():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[3.0, -1.0, 0.5]])
    for metric in ['cosine', 'arccos']:
        expected = distance_matrix(a, b, metric)[0, 0]
        opposite = distance_matrix(a, -a, metric)[0, 0]
        # squaring these values overflows or underflows
        for scale in [1e-300, 1e-170, 1e-160, 1e150, 1e200, 1e300]:
            A, B = a * scale, b * scale
            npt.assert_allclose(distance_matrix(A, B, metric)[0, 0], expected, rtol=1e-12)
            npt.assert_allclose(distance_matrix(A, b, metric)[0, 0], expected, rtol=1e-12)
            npt.assert_allclose(distance_matrix(a, B, metric)[0, 0], expected, rtol=1e-12)
            npt.assert_allclose(distance_matrix(A, -A, metric)[0, 0], opposite, rtol=1e-12)
            npt.assert_equal(distance_matrix(A, A, metric)[0, 0], 0.0)
        # NaN and infinite values give NaN, also against a zero vector
        for bad in [np.nan, np.inf]:
            x = np.array([[bad, 1.0, 5.0]])
            assert np.isnan(distance_matrix(x, b, metric)[0, 0])
            assert np.isnan(distance_matrix(b, x, metric)[0, 0])
            assert np.isnan(distance_matrix(x, np.zeros((1, 3)), metric)[0, 0])

def test_unknown_metric_message_names_it():
    X = np.ones((2,3))
    try:
        distance_matrix(X, X, 'manhattan')
    except ValueError as e:
        message = str(e)
    else:
        raise AssertionError('expected a ValueError for an unknown metric')
    assert "'manhattan'" in message
    for name in ground_metrics:
        assert f"'{name}'" in message

def test_metric_name_can_be_any_kind_of_str():
    import enum

    class Name(str):
        pass

    class Metric(str, enum.Enum):
        EUCLIDEAN = 'euclidean'

    rng = np.random.default_rng(0)
    X = rng.random((4,3))
    expected = distance_matrix(X, X, 'euclidean')
    names = np.array(['euclidean', 'cosine'])
    for metric in [names[0], Name('euclidean'), Metric.EUCLIDEAN]:
        npt.assert_array_equal(distance_matrix(X, X, metric), expected)
    # anything that is not a str is a TypeError that names the argument
    for metric in [None, b'euclidean', 3]:
        try:
            distance_matrix(X, X, metric)
        except TypeError as e:
            assert 'metric' in str(e)
        else:
            raise AssertionError('expected a TypeError for a metric that is not a str')

def test_distance_matrix_rejects_different_column_counts():
    X1 = np.ones((2,4))
    X2 = np.ones((2,2))
    npt.assert_raises(ValueError, distance_matrix, X1, X2, 'euclidean')
    npt.assert_raises(ValueError, distance_matrix, X2, X1, 'euclidean')

if __name__ == '__main__':
    test_distance_matrix()
    test_known_values()
    test_default_metric_is_norm_ssd()
    test_inf_norm_identical_vectors_is_zero()
    test_nan_gives_nan_for_every_metric()
    test_cosine_and_arccos_with_zero_vectors()
    test_arccos_identical_opposite_and_small_angles()
    test_cosine_identical_opposite_and_small_angles()
    test_cosine_and_arccos_do_not_depend_on_vector_length()
    test_unknown_metric_message_names_it()
    test_metric_name_can_be_any_kind_of_str()
    test_distance_matrix_rejects_different_column_counts()
    print('Tests passed!')