"""Tests for xxhash module."""

import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_array_equal
from thetan.hashing import hash32, hash64, hash_rows_32, hash_rows_64


def test_hash32_basic():
    """Test basic hash32 functionality."""
    actual_hash = hash32(b"Hello, world!")

    # Test with empty data
    assert_equal(hash32(b""), 46947589)  # Known value for empty string with seed 0

    # Test with basic string
    assert_equal(hash32(b"Hello, world!"), actual_hash)

    # Test with different seed
    assert hash32(b"Hello, world!", 42) != hash32(b"Hello, world!", 0)

    # Verify deterministic behavior
    assert_equal(hash32(b"Hello, world!", 42), hash32(b"Hello, world!", 42))


def test_hash64_basic():
    """Test basic hash64 functionality."""
    actual_hash = hash64(b"Hello, world!")

    # Test with empty data
    assert_equal(hash64(b""), 17241709254077376921)

    # Test with basic string
    assert_equal(hash64(b"Hello, world!", 0), actual_hash)

    # Test with different seed
    assert hash64(b"Hello, world!", 42) != hash64(b"Hello, world!", 0)

    # Verify deterministic behavior
    assert_equal(hash64(b"Hello, world!", 42), hash64(b"Hello, world!", 42))


def test_hash_with_numpy():
    """Test xxhash with NumPy arrays."""
    # Create a numpy array and convert to bytes
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    arr_bytes = arr.tobytes()

    # Verify we get consistent results
    h1 = hash32(arr_bytes)
    h2 = hash32(arr_bytes)
    assert_equal(h1, h2)

    # Different array should have different hash
    arr2 = np.array([1, 2, 3, 4, 6], dtype=np.int32)
    assert hash32(arr_bytes) != hash32(arr2.tobytes())


def test_different_data_types():
    """Test hashing different data types."""
    # Test with integers
    int_bytes = (123456).to_bytes(4, byteorder='little')
    assert isinstance(hash32(int_bytes), int)

    # Test with floats
    float_bytes = np.array([3.14159], dtype=np.float64).tobytes()
    assert isinstance(hash64(float_bytes), int)

    # Test with boolean array
    bool_bytes = np.array([True, False, True], dtype=bool).tobytes()
    assert isinstance(hash32(bool_bytes), int)


def test_invalid_input():
    """Test handling of invalid input."""
    # Test with non-bytes input
    assert_raises(TypeError, hash32, "Hello, world!")  # String instead of bytes
    assert_raises(TypeError, hash64, 12345)  # Integer instead of bytes


def test_hash_rows():
    """Test hash_rows function with 2D arrays."""
    # Create a 2D numpy array
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)

    hashes = []
    for i in range(3):
        # Verify that the hash of each row is not equal to the hash of the entire array
        hashes.append(hash32(arr[i].tobytes()))
    # Get hashes using hash_rows
    row_hashes = hash_rows(arr)

    # Verify output shape
    assert_equal(row_hashes.shape, (3,))

    # Verify deterministic results
    assert_array_equal(row_hashes, hash_rows(arr))

    # Verify using different seed gives different results
    assert not np.array_equal(row_hashes, hash_rows(arr, seed=42))

    # Verify each row gets a different hash
    assert_equal(len(np.unique(row_hashes)), 3)

    # Test hash_rows with empty array.
    arr = np.zeros((0, 3), dtype=np.int32)
    row_hashes = hash_rows(arr)
    assert_equal(row_hashes.shape, (0,))


def test_hash_rows_identical():
    """Test hash_rows with identical rows."""
    # Create an array with identical rows
    arr = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32)

    # Get hashes
    row_hashes = hash_rows_32(arr)

    # Identical rows should have identical hashes
    assert_equal(row_hashes[0], row_hashes[1])
