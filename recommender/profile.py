from typing import Literal

import numpy as np


def build_user_profile(
    X_user: np.ndarray,
    method: Literal["mean", "median"] = "median",
) -> np.ndarray:
    """
    Aggregate user track features into one or more profile vectors.

    Parameters
    ----------
    X_user : np.ndarray
        Scaled feature matrix of user's tracks. Shape (U, d).
    method : {"mean", "median"}, default="median"
        Aggregation method:
        - "mean": one profile, mean across tracks.
        - "median": one profile, median across tracks.

    Returns
    -------
    np.ndarray
        A single profile vector of shape (d,).
    """
    if X_user.size == 0:
        raise ValueError("User feature matrix is empty, cannot build profile.")

    if method == "mean":
        return np.nanmean(X_user, axis=0)

    elif method == "median":
        return np.nanmedian(X_user, axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")
