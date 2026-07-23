"""Vector similarity metrics used by recommendation scoring."""

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
    if V.shape[1] != u.shape[0]:
        raise ValueError("Input u and rows of V must have the same number of features.")

    # normalize u
    u_norm = np.linalg.norm(u)
    if u_norm == 0:
        return np.zeros(V.shape[0], dtype=np.float32)

    # normalize rows of V
    V_norms = np.linalg.norm(V, axis=1)
    safe_V = np.zeros_like(V, dtype=np.result_type(V.dtype, np.float32))
    np.divide(
        V,
        V_norms[:, None],
        out=safe_V,
        where=V_norms[:, None] != 0,
    )

    # normalize u once
    u_unit = u / u_norm

    # dot product = cosine similarity
    sims = safe_V @ u_unit
    return sims.astype(np.float32)
