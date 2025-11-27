"""
similarity.py
-------------
Implements similarity metrics between vectors.
"""

import numpy as np

def cosine(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vector u and each row of V.

    Parameters
    ----------
    u : np.ndarray, shape (d,)
        Single vector.
    V : np.ndarray, shape (n, d)
        Matrix of candidate vectors.

    Returns
    -------
    np.ndarray, shape (n,)
        Cosine similarity scores in [-1, 1].
    """
    if u.ndim != 1:
        raise ValueError("Input u must be a 1D vector.")
    if V.ndim != 2:
        raise ValueError("Input V must be a 2D array.")

    # normalize u
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(V.shape[0], dtype=np.float32)

    # normalize rows of V
    V_norms = np.linalg.norm(V, axis=1)
    safe_V = np.where(V_norms[:, None] != 0, V / V_norms[:, None], 0.0)

    # normalize u once
    u_unit = u / u_norm

    # dot product = cosine similarity
    sims = safe_V @ u_unit
    return sims.astype(np.float32)


def weighted_cosine(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    *Placeholder function*

    Parameters
    ----------
    u : np.ndarray, shape (d,)
    V : np.ndarray, shape (n, d)

    Returns
    -------
    np.ndarray, shape (n,)
    """
    return cosine(u, V)
