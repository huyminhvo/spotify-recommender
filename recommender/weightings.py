# weightings.py
"""
weightings.py
-------------
Handles application of feature weights (from sliders/UI).
"""

import numpy as np
from typing import Dict, Sequence

def apply_weights(
    X: np.ndarray, 
    weights: Dict[str, float], 
    feature_order: Sequence[str]
) -> np.ndarray:
    """
    Apply feature weights to a vector or matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix or vector. Shape (n, d) or (d,).
    weights : dict[str, float]
        Mapping from feature name -> weight multiplier.
        Any features not in this dict get weight 1.0.
    feature_order : sequence of str
        Ordered list of features corresponding to columns in X.

    Returns
    -------
    np.ndarray
        Weighted features, same shape as input.
    """
    # Build weight vector aligned with feature_order
    w = np.array([weights.get(f, 1.0) for f in feature_order], dtype=np.float32)

    if X.ndim == 1:  # single vector
        return X * w
    elif X.ndim == 2:  # matrix
        return X * w[None, :]
    else:
        raise ValueError("X must be 1D or 2D numpy array.")
