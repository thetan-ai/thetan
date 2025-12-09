import numpy as np
import pytest

from thetan import distance_matrix


class TestGetMetricFunc:
    """Test suite for metric function selection in distance_matrix."""

    @pytest.fixture
    def sample_vectors(self):
        """Create sample input vectors for testing."""
        X1 = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float64)
        X2 = np.array([[1.0, 2.0, 3.0],
                       [7.0, 8.0, 9.0],
                       [0.0, 0.0, 0.0]], dtype=np.float64)
        return X1, X2

    @pytest.fixture
    def identical_vectors(self):
        """Create identical vectors for testing zero-distance cases."""
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float64)
        return X, X.copy()

    @pytest.fixture
    def unit_vectors(self):
        """Create unit vectors for testing normalized metrics."""
        X1 = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=np.float64)
        X2 = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        return X1, X2

    # -------------------- Basic Functionality Tests --------------------

    @pytest.mark.parametrize("metric", [
        'euclidean', 'ssd', 'norm_ssd', 'norm_ssd_un',
        'dot_prod', 'l1_norm', 'cosine', 'arccos'
    ])
    def test_metric_returns_valid_array(self, sample_vectors, metric):
        """Test that each metric returns a valid distance matrix."""
        X1, X2 = sample_vectors
        result = distance_matrix(X1, X2, metric=metric)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (X1.shape[0], X2.shape[0])
        assert result.dtype == np.float64
        assert not np.any(np.isnan(result))

    def test_invalid_metric_raises_error(self, sample_vectors):
        """Test that an invalid metric name raises ValueError."""
        X1, X2 = sample_vectors
        
        with pytest.raises(ValueError, match="Unknown metric"):
            distance_matrix(X1, X2, metric='invalid_metric')

    def test_default_metric_is_norm_ssd(self, sample_vectors):
        """Test that default metric is 'norm_ssd'."""
        X1, X2 = sample_vectors
        
        result_default = distance_matrix(X1, X2)
        result_explicit = distance_matrix(X1, X2, metric='norm_ssd')
        
        np.testing.assert_array_equal(result_default, result_explicit)

    # -------------------- Metric-Specific Behavior Tests --------------------

    def test_euclidean_identical_vectors_zero(self, identical_vectors):
        """Test that euclidean distance of identical vectors is zero."""
        X1, X2 = identical_vectors
        result = distance_matrix(X1, X2, metric='euclidean')
        
        # Diagonal should be zero (comparing each vector with itself)
        np.testing.assert_array_almost_equal(np.diag(result), 0.0)

    def test_euclidean_known_values(self):
        """Test euclidean distance with known values."""
        X1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        X2 = np.array([[3.0, 4.0, 0.0]], dtype=np.float64)
        
        result = distance_matrix(X1, X2, metric='euclidean')
        
        # Distance should be 5 (3-4-5 triangle)
        np.testing.assert_almost_equal(result[0, 0], 5.0)

    def test_ssd_known_values(self):
        """Test SSD (Sum of Squared Differences) with known values."""
        X1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        X2 = np.array([[3.0, 4.0, 0.0]], dtype=np.float64)
        
        result = distance_matrix(X1, X2, metric='ssd')
        
        # SSD should be 9 + 16 + 0 = 25
        np.testing.assert_almost_equal(result[0, 0], 25.0)

    def test_l1_norm_known_values(self):
        """Test L1 norm (Manhattan distance) with known values."""
        X1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        X2 = np.array([[3.0, 4.0, 5.0]], dtype=np.float64)
        
        result = distance_matrix(X1, X2, metric='l1_norm')
        
        # L1 should be |3| + |4| + |5| = 12
        np.testing.assert_almost_equal(result[0, 0], 12.0)

    def test_dot_prod_known_values(self):
        """Test dot product with known values."""
        X1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        X2 = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
        
        result = distance_matrix(X1, X2, metric='dot_prod')
        
        # Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        np.testing.assert_almost_equal(result[0, 0], 32.0)

    def test_cosine_orthogonal_vectors(self, unit_vectors):
        """Test cosine distance for orthogonal unit vectors."""
        X1, X2 = unit_vectors
        result = distance_matrix(X1, X2, metric='cosine')
        
        # Cosine distance between orthogonal vectors should be 1
        # (cosine similarity = 0, so distance = 1 - 0 = 1)
        np.testing.assert_almost_equal(result[0, 1], 1.0)  # [1,0,0] vs [0,0,1]
        np.testing.assert_almost_equal(result[1, 0], 1.0)  # [0,1,0] vs [1,0,0]

    def test_cosine_identical_vectors(self, unit_vectors):
        """Test cosine distance for identical vectors is zero."""
        X1, X2 = unit_vectors
        result = distance_matrix(X1, X2, metric='cosine')
        
        # Cosine distance between identical vectors should be 0
        np.testing.assert_almost_equal(result[0, 0], 0.0)  # [1,0,0] vs [1,0,0]

    def test_arccos_orthogonal_vectors(self, unit_vectors):
        """Test arccosine distance for orthogonal unit vectors."""
        X1, X2 = unit_vectors
        result = distance_matrix(X1, X2, metric='arccos')
        
        # Arccos distance between orthogonal vectors should be pi/2
        np.testing.assert_almost_equal(result[0, 1], np.pi / 2, decimal=5)

    # -------------------- Metric Differentiation Tests --------------------

    def test_different_metrics_produce_different_results(self, sample_vectors):
        """Test that different metrics produce different distance matrices."""
        X1, X2 = sample_vectors
        
        results = {}
        for metric in ['euclidean', 'ssd', 'l1_norm', 'cosine']:
            results[metric] = distance_matrix(X1, X2, metric=metric)
        
        # Verify that at least some metrics produce different results
        assert not np.allclose(results['euclidean'], results['ssd'])
        assert not np.allclose(results['euclidean'], results['l1_norm'])
        assert not np.allclose(results['ssd'], results['cosine'])

    def test_ssd_is_euclidean_squared(self, sample_vectors):
        """Test that SSD equals euclidean distance squared."""
        X1, X2 = sample_vectors
        
        euclidean_result = distance_matrix(X1, X2, metric='euclidean')
        ssd_result = distance_matrix(X1, X2, metric='ssd')
        
        np.testing.assert_array_almost_equal(
            euclidean_result ** 2, 
            ssd_result
        )

    # -------------------- Edge Case Tests --------------------

    def test_single_element_vectors(self):
        """Test with single-element vectors."""
        X1 = np.array([[5.0]], dtype=np.float64)
        X2 = np.array([[3.0]], dtype=np.float64)
        
        result = distance_matrix(X1, X2, metric='euclidean')
        
        np.testing.assert_almost_equal(result[0, 0], 2.0)

    def test_high_dimensional_vectors(self):
        """Test with high-dimensional vectors."""
        np.random.seed(42)
        X1 = np.random.rand(10, 100).astype(np.float64)
        X2 = np.random.rand(15, 100).astype(np.float64)
        
        result = distance_matrix(X1, X2, metric='euclidean')
        
        assert result.shape == (10, 15)
        assert np.all(result >= 0)

    def test_single_row_matrices(self):
        """Test with single-row input matrices."""
        X1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        X2 = np.array([[4.0, 5.0, 6.0]], dtype=np.float64)
        
        result = distance_matrix(X1, X2, metric='euclidean')
        
        assert result.shape == (1, 1)

    # -------------------- Symmetry Tests --------------------

    def test_euclidean_symmetry(self):
        """Test that euclidean distance is symmetric."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        
        result = distance_matrix(X, X, metric='euclidean')
        
        np.testing.assert_array_almost_equal(result, result.T)

    def test_cosine_symmetry(self):
        """Test that cosine distance is symmetric."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        
        result = distance_matrix(X, X, metric='cosine')
        
        np.testing.assert_array_almost_equal(result, result.T)


# -------------------- Run with pytest --------------------
if __name__ == '__main__':
    pytest.main([__file__, '-v'])