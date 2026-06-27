from typing import Iterable, Optional, Tuple

import pandas as pd

from utils.matcher import canon_artist_primary


def filter_candidates(
    catalog: pd.DataFrame,
    exclude_ids: Optional[Iterable[str]] = None,
    exclude_artists: Optional[Iterable[str]] = None,
    min_popularity: Optional[int] = None,
    max_popularity: Optional[int] = None,
    year_range: Optional[Tuple[int, int]] = None,
) -> pd.DataFrame:
    df = catalog.copy()

    if exclude_ids is not None:
        df = df[~df["spotify_id"].isin(set(exclude_ids))]

    if exclude_artists is not None:
        artist_set = {artist for artist in exclude_artists if artist}
        if artist_set:
            if "artist_primary_canon" in df.columns:
                candidate_artists = df["artist_primary_canon"]
            else:
                candidate_artists = df["artists_raw"].apply(canon_artist_primary)
            df = df[~candidate_artists.isin(artist_set)]

    if min_popularity is not None:
        df = df[df["popularity"].fillna(0) >= min_popularity]
    if max_popularity is not None:
        df = df[df["popularity"].fillna(100) <= max_popularity]

    if year_range is not None:
        lo, hi = year_range
        df = df[df["release_year"].between(lo, hi, inclusive="both")]

    return df.reset_index(drop=True)
