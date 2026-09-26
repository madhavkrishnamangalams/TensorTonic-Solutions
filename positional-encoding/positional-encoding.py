import numpy as np

def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """
    out = np.zeros((seq_len,d_model))
    pos = np.arange(seq_len)
    pos = pos.reshape(-1, 1)
    two_i = np.arange(0, d_model, 2)
    denom = np.power(base, (two_i / d_model))
    angle = pos / denom
    out[:,0::2] = np.sin(angle)
    out[:,1::2] = np.cos(angle[:,:d_model//2])
    return out