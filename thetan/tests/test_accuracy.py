import itertools

import numpy as np
import pytest
from thetan.accuracy import accuracy_munkres, f_matrix, f1_hungarian


def best_score_by_trying_every_matching(f1):
    n_rows, n_cols = f1.shape
    best = 0.0
    if n_rows <= n_cols:
        for cols in itertools.permutations(range(n_cols), n_rows):
            best = max(best, sum(f1[r, c] for r, c in zip(range(n_rows), cols)))
    else:
        for rows in itertools.permutations(range(n_rows), n_cols):
            best = max(best, sum(f1[r, c] for r, c in zip(rows, range(n_cols))))
    return best / min(n_rows, n_cols)

def test_same_grouping_scores_one_whatever_the_label_numbers():
    true = np.array([0, 0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 0, 2, 2, 1, 1])
    assert accuracy_munkres(true, pred) == 1.0
    assert accuracy_munkres(true, true) == 1.0

def test_known_values():
    # one of six points in the wrong cluster. Class 0 has precision 1 and
    # recall 3/4, an F1 of 0.857 after rounding. Class 1 has precision 2/3
    # and recall 1, an F1 of 0.8. The mean is 0.8285.
    true = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([0, 0, 0, 1, 1, 1])
    assert accuracy_munkres(true, pred) == pytest.approx(0.8285, rel=1e-12, abs=0)
    # scores this function gave when it used the munkres package (1.1.4),
    # before the matching was moved to scipy
    assert accuracy_munkres([0, 0, 0, 1, 1, 2, 3], [0, 0, 0, 2, 2, 1, 1]) == pytest.approx(0.889, rel=1e-12, abs=0)
    assert accuracy_munkres([0, 0, 0, 0], [0, 1, 2, 3]) == pytest.approx(0.4, rel=1e-12, abs=0)
    assert accuracy_munkres(['a', 'a', 'b', 'b'], ['x', 'y', 'y', 'y']) == pytest.approx(0.7335, rel=1e-12, abs=0)

def test_more_clusters_than_classes_and_fewer():
    true = np.repeat([0, 1], 50)
    # each class split into 5 clusters: a matched pair has precision 1 and
    # recall 0.2, an F1 of 0.333
    assert accuracy_munkres(true, np.arange(100) // 10) == pytest.approx(0.333, rel=1e-12, abs=0)
    # everything in one cluster: the only matched pair has precision 0.5
    # and recall 1, an F1 of 0.667. The class left without a partner is
    # not counted.
    assert accuracy_munkres(true, np.zeros(100, dtype=int)) == pytest.approx(0.667, rel=1e-12, abs=0)

def test_order_of_arguments_does_not_matter():
    rng = np.random.default_rng(0)
    for _ in range(20):
        true = rng.integers(0, 4, 200)
        pred = rng.integers(0, 6, 200)
        assert accuracy_munkres(true, pred) == pytest.approx(accuracy_munkres(pred, true), rel=1e-12, abs=0)

def test_matching_is_the_best_one():
    rng = np.random.default_rng(0)
    for trial in range(60):
        true = rng.integers(0, int(rng.integers(1, 6)), 80)
        pred = rng.integers(0, int(rng.integers(1, 7)), 80)
        if trial % 2:
            # mostly agreeing with the true classes
            pred = np.where(rng.random(80) < 0.8, true, pred)
        expected = best_score_by_trying_every_matching(f_matrix(pred, true))
        assert accuracy_munkres(true, pred) == pytest.approx(expected, rel=1e-12, abs=0)

def test_lists_and_string_labels():
    assert accuracy_munkres([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert accuracy_munkres(['a', 'a', 'b', 'b'], ['x', 'x', 'y', 'y']) == 1.0

def test_f_matrix_known_values():
    # rows are the true classes, columns the predicted clusters
    f1 = f_matrix([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1])
    assert np.array_equal(f1, [[0.857, 0.286], [0.0, 0.8]])
    assert f_matrix([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1]).shape == (3, 2)

def test_f1_hungarian_known_values():
    # the best matching takes the diagonal here, not the largest entry first
    f1 = np.array([[0.9, 0.8], [0.8, 0.1]])
    assert f1_hungarian(f1) == pytest.approx(0.8, rel=1e-12, abs=0)
    assert f1_hungarian(np.array([[0.2, 0.7, 0.4]])) == pytest.approx(0.7, rel=1e-12, abs=0)
