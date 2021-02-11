import numpy as np
from scipy.linalg import qr, svd


def bksvd(data, k=6, block_size=None, num_iter=3):

    if block_size is None:
        block_size = k

    k = min(k, min(data.shape))
    u = np.zeros((1, data.shape[1]))

    l = np.ones((data.shape[0], 1))

    K = np.zeros((data.shape[1], block_size * num_iter))
    block = np.random.randn(data.shape[1], block_size)

    block, _ = qr(block, mode='economic')

    T = np.zeros((data.shape[1], block_size))

    for i in range(num_iter):
        T = data @ block - l * (u @ block)
        block = data.T @ T - (u.T * (l.T @ T))
        block, _ = qr(block, mode='economic')
        K[:, i * block_size: (i + 1) * block_size] = block
    Q, _ = qr(K, mode='economic')

    # Rayleigh-Ritz iteration
    T = data @ Q - l @ (u @ Q)

    Ut, St, Vt = svd(T, full_matrices=False)
    U = Ut[:, :k]
    S = St[:k]
    V = Q @ Vt.T
    return U, S, V[:, :k].T


def compute_rqb(A, rank):
    """Randomized QB Decomposition.
    Randomized algorithm for computing the approximate low-rank QB
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank
    `rank << min{m, n}`. The input matrix is factored as `A = Q * B`.
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.
    rank : integer
        Target rank. Best if `rank << min{m,n}`
    Returns
    -------
    Q:  array_like, shape `(m, rank)`.
        Orthonormal basis matrix.
    B : array_like, shape `(rank, n)`.
        Smaller matrix.
    """
    def johnson_lindenstrauss(A, l):
        """
        Given an m x n matrix A, and an integer l, this scheme computes an
        m x l orthonormal matrix Q whose range approximates the range of A
        """
        # construct gaussian random matrix
        Omega = np.random.standard_normal(size=(A.shape[1], l)).astype(A.dtype)

        # project A onto Omega
        return A.dot(Omega)

    Q = johnson_lindenstrauss(A, rank)
    Qt = Q.T

    B = np.dot(Qt, A)
    return Q, B


def rsvd(A, rank):
    """Randomized Singular Value Decomposition.
    Randomized algorithm for computing the approximate low-rank singular value
    decomposition of a rectangular `(m, n)` matrix `A`, with target rank
    `k << min{m, n}`. The input matrix is factored as `A = U * diag(s) * Vt`.
    The right singular vectors are the columns of the real or complex unitary
    matrix `U`. The left singular vectors are the columns of the real or
    complex unitary matrix `V`. The singular values `s` are non-negative and
    real numbers.
    Parameters
    ----------
    A : array_like, shape `(m, n)`.
        Input array.
    rank : integer
        Target rank. Best if `rank << min{m,n}`
    Returns
    -------
    U:  array_like
        Right singular values, array of shape `(m, rank)`.
    s : array_like
        Singular values, 1-d array of length `rank`.
    Vh : array_like
        Left singular values, array of shape `(rank, n)`.
    """
    m, n = A.shape

    # Compute QB decomposition
    Q, B = compute_rqb(A, rank)

    # Compute SVD
    U, s, Vt = svd(B)

    # Recover right singular vectors
    U = Q.dot(U)

    # Return Trunc
    return U[:, :rank], s[:rank], Vt[:rank, :]


if __name__ == '__main__':

    A = np.random.rand(50, 50)

    U, s, V = bksvd(A, k=40, block_size=None, num_iter=3)

    # U, s, V = rsvd(A, rank=10)

    from pylab import imshow, show, figure

    figure()
    imshow(A)

    figure()
    imshow(np.dot(U, np.dot(np.diag(s), V)))

    show()



    pass