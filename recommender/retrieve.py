# retrieve.py
import pandas as pd
from typing import Optional, Tuple, Iterable

def filter_candidates(
    catalog: pd.DataFrame,
    exclude_ids: Optional[Iterable[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    df = catalog.copy()

    if exclude_ids is not None:
        df = df[~df["spotify_id"].isin(set(exclude_ids))]

    if min_popularity is not None:
        df = df[df["popularity"].fillna(0) >= min_popularity]
    if max_popularity is not None:
        df = df[df["popularity"].fillna(100) <= max_popularity]

    if year_range is not None:
        lo, hi = year_range
        df = df[df["release_year"].between(lo, hi, inclusive="both")]

    return df.reset_index(drop=True)
