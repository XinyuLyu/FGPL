import numpy as np
import scipy.sparse as sp
import torch

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx_output = r_mat_inv.dot(mx)
    return mx_output

def adj_laplacian(adj):
    adj = normalize(adj)
    return adj