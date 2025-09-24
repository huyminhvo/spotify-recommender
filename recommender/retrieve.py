# retrieve.py
"""
retrieve.py
-----------
Filters and retrieves candidate tracks from the catalog.
"""

import pandas as pd
from typing import Optional, Tuple, Iterable

def filter_candidates(
    catalog: pd.DataFrame,
    exclude_ids: Optional[Iterable[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Apply optional filters to reduce catalog before similarity scoring.

    Parameters
    ----------
    catalog : pd.DataFrame
        The full catalog DataFrame with metadata + features.
    exclude_ids : iterable of str, optional
        Track IDs to exclude (e.g., user's seed tracks).
    min_popularity : int, optional
        Minimum popularity cutoff.
    max_popularity : int, optional
        Maximum popularity cutoff.
    year_range : (int, int), optional
        Inclusive (min_year, max_year) range.

    Returns
    -------
    pd.DataFrame
        Filtered catalog DataFrame.
    """
    df = catalog.copy()

    # Exclude seeds
    if exclude_ids is not None:
        df = df[~df["spotify_id"].isin(set(exclude_ids))]

    # Popularity cut
    if min_popularity is not None:
        df = df[df["popularity"].fillna(0) >= min_popularity]
    if max_popularity is not None:
        df = df[df["popularity"].fillna(100) <= max_popularity]

    # Release year range
    if year_range is not None:
        lo, hi = year_range
        df = df[df["release_year"].between(lo, hi, inclusive="both")]

    return df.reset_index(drop=True)
