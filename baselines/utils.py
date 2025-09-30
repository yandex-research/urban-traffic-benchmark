import numpy as np
import scipy.sparse as sp

from scipy.sparse import linalg


def calculate_normalized_laplacian_matrix(adjacency_matrix):
    adjacency_matrix = sp.coo_matrix(adjacency_matrix)
    d = np.array(adjacency_matrix.sum(1))
    d[d == 0] = 1
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    res = sp.eye(adjacency_matrix.shape[0]) - d_mat_inv_sqrt.dot(adjacency_matrix).dot(d_mat_inv_sqrt).tocoo()
    return res


def calculate_scaled_laplacian_matrix(adjacency_matrix):
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    L = calculate_normalized_laplacian_matrix(adjacency_matrix)
    lambda_max, _ = linalg.eigsh(L, 1, which='LM')
    lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    res = (2 / lambda_max * L) - I
    return res.todense()


def calculate_transition_matrix(adjacency_matrix):
    adjacency_matrix = sp.coo_matrix(adjacency_matrix)
    d = np.array(adjacency_matrix.sum(1)).flatten()
    d[d == 0] = 1
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    res = d_mat_inv.dot(adjacency_matrix)
    return res.todense()


def calculate_cheb_polynomials(matrix, order=2):
    n = matrix.shape[0]
    matrices = [np.eye(n), matrix.copy()]
    for i in range(2, order):
        matrices.append(np.matmul(2 * matrix, matrices[i - 1]) - matrices[i - 2])
    return np.asarray(matrices)
