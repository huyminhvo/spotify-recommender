"""
profile.py
----------
Builds user profile vector(s) from their seed tracks.
"""

import numpy as np
from typing import Literal, List, Union

def build_user_profile(
    X_user: np.ndarray,
    method: Literal["mean", "median", "clustered"] = "median",
    n_clusters: int = 2
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Aggregate user track features into one or more profile vectors.
    
    Parameters
    ----------
    X_user : np.ndarray
        Scaled feature matrix of user's tracks. Shape (U, d).
    method : {"mean", "median", "clustered"}, default="median"
        Aggregation method:
        - "mean": one profile, mean across tracks.
        - "median": one profile, median across tracks.
        - "clustered": (future) multiple profiles via clustering.
    n_clusters : int
        Number of clusters to use if method="clustered".
    
    Returns
    -------
    np.ndarray or list[np.ndarray]
        - For "mean"/"median": a single vector of shape (d,).
        - For "clustered": list of vectors, each shape (d,).
    """
    if X_user.size == 0:
        raise ValueError("User feature matrix is empty — cannot build profile.")

    if method == "mean":
        return np.nanmean(X_user, axis=0)

    elif method == "median":
        return np.nanmedian(X_user, axis=0)

    elif method == "clustered":
        # Placeholder — implement later with k-means on X_user
        raise NotImplementedError("Clustered user profiling not yet implemented.")

    else:
        raise ValueError(f"Unknown method: {method}")
