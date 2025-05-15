import numpy as np

def validate_generator(G: np.ndarray) -> bool:
    if len(G.shape) != 2 or G.shape[0] != G.shape[1]:
        return False
        
    if not np.allclose(np.sum(G, axis=1), 0):
        return False
        
    diag_mask = np.eye(G.shape[0], dtype=bool)
    if np.any(G[~diag_mask] < 0):
        return False
        
    return True