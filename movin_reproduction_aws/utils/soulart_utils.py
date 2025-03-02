import torch
import numpy as np

def get_ours_skeleton():
    return np.array(
        [
            [ 0, 1],
            [ 1, 2],
            [ 2, 3],
            [ 3, 4],
            [ 2, 5],
            [ 5, 6],
            [ 6, 7],
            [ 7, 8],
            [ 2, 9],
            [ 9, 10],
            [10, 11],
            [11, 12],
            [ 0, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [ 0, 17],
            [17, 18],
            [18, 19],
            [19, 20]
        ]
    )

def get_ours_joint_num():
    return 21

def get_ours_adj_matrix(normalize=True):
    connected = get_ours_skeleton()
    adj = np.zeros(shape=(get_ours_joint_num(), get_ours_joint_num()))

    for item in connected:
        s, f = int(item[0]), int(item[1])
        adj[s][f] = 1. ; adj[f][s] = 1.
    
    if normalize: 
        return symmetric_normalize_adj(adj)
    return torch.from_numpy(adj + np.eye(adj.shape[0])).unsqueeze(0)
    

def symmetric_normalize_adj(A):
    """
    Perform symmetric normalization of an adjacency matrix, 
    which is adopted in ST-GCN (https://arxiv.org/abs/1801.07455).
    Formula: D^(-1/2) * (A + I) * D^(-1/2), where D_ii = sum_j(A_ij + I_ij)

    Args:
        A (np.ndarray): Adjacency matrix of shape (N, N)
        
    Returns:
        np.ndarray: Symmetrically normalized adjacency matrix
    """
    
    A_hat = A + np.eye(A.shape[0])
    D = np.sum(A_hat, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))  # Adding epsilon to avoid division by zero
    A_normalized = D_inv_sqrt @ A_hat @ D_inv_sqrt
    
    return torch.from_numpy(A_normalized).to(torch.float32).unsqueeze(0)
