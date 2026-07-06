from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from utils.matcher import canon_artist_primary, canon_title

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_LIMIT = 100_000
FLOAT32_COLUMNS = [
    "popularity",
    "release_year",
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "loudness",
    "tempo",
    "key",
    "mode",
]


class CatalogStore:
    """Query a Parquet catalog without materializing the whole file in memory."""

    def __init__(self, parquet_path: str | Path, candidate_limit: int | None = None):
        self.path = Path(parquet_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        configured_limit = os.getenv("CATALOG_CANDIDATE_LIMIT")
        self.candidate_limit = candidate_limit or int(configured_limit or DEFAULT_CANDIDATE_LIMIT)

    @staticmethod
    def _connect():
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "DuckDB is required for memory-efficient catalog access. "
                "Install the packages in requirements.txt."
            ) from exc
        return duckdb.connect(":memory:")

    def _query(self, sql: str, parameters: list | None = None) -> pd.DataFrame:
        for attempt in range(2):
            try:
                with self._connect() as connection:
                    return connection.execute(sql, parameters or []).fetch_df()
            except Exception as exc:
                if attempt == 0 and "ZSTD Decompression failure" in str(exc):
                    logger.warning(
                        "Retrying catalog query after a transient ZSTD decompression failure"
                    )
                    continue
                raise
        raise AssertionError("catalog query retry loop exited unexpectedly")

    def match_track(self, track: dict, duration_tol: int = 2000) -> dict | None:
        track_id = track.get("id")
        if track_id:
            matched = self._query(
                "SELECT * FROM read_parquet(?) WHERE spotify_id = ? LIMIT 1",
                [str(self.path), track_id],
            )
            if not matched.empty:
                return matched.iloc[0].to_dict()

        title = canon_title(track.get("name", ""))
        artists = track.get("artists", [])
        artist = canon_artist_primary([item["name"] for item in artists] if artists else [])
        if not title or not artist:
            return None

        duration = track.get("duration_ms")
        duration_clause = ""
        parameters = [str(self.path), title, artist]
        if duration:
            duration_clause = "AND abs(duration_ms - ?) <= ?"
            parameters.extend([duration, duration_tol])

        matched = self._query(
            f"""
            SELECT * FROM read_parquet(?)
            WHERE title_canon = ? AND artist_primary_canon = ?
              {duration_clause}
            ORDER BY
              CASE WHEN regexp_matches(lower(title_raw),
                '(live|remix|instrumental|clean|explicit|karaoke|cover|demo|edit)')
                THEN 1 ELSE 0 END,
              popularity DESC NULLS LAST, release_year DESC NULLS LAST
            LIMIT 1
            """,
            parameters,
        )
        return None if matched.empty else matched.iloc[0].to_dict()

    def load_candidates(
        self,
        exclude_ids: Optional[Iterable[str]] = None,
        exclude_artists: Optional[Iterable[str]] = None,
        min_popularity: Optional[int] = None,
        max_popularity: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        parameters: list = [str(self.path)]

        ids = [value for value in (exclude_ids or []) if value]
        if ids:
            clauses.append(f"spotify_id NOT IN ({', '.join('?' for _ in ids)})")
            parameters.extend(ids)

        artists = [value for value in (exclude_artists or []) if value]
        if artists:
            clauses.append(f"artist_primary_canon NOT IN ({', '.join('?' for _ in artists)})")
            parameters.extend(artists)

        if min_popularity is not None:
            clauses.append("coalesce(popularity, 0) >= ?")
            parameters.append(min_popularity)
        if max_popularity is not None:
            clauses.append("coalesce(popularity, 100) <= ?")
            parameters.append(max_popularity)
        if year_range is not None:
            clauses.append("release_year BETWEEN ? AND ?")
            parameters.extend(year_range)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters.append(self.candidate_limit)
        candidates = self._query(
            f"""
            SELECT * FROM read_parquet(?)
            {where}
            ORDER BY hash(coalesce(spotify_id, title_canon, ''))
            LIMIT ?
            """,
            parameters,
        )
        for column in FLOAT32_COLUMNS:
            if column in candidates:
                candidates[column] = pd.to_numeric(
                    candidates[column], errors="coerce", downcast="float"
                )
        if "duration_ms" in candidates:
            candidates["duration_ms"] = pd.to_numeric(
                candidates["duration_ms"], errors="coerce", downcast="integer"
            )
        return candidates
