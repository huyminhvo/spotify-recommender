from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

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


class CatalogQueryError(RuntimeError):
    """Raised when the local deployment catalog cannot be queried."""


class CatalogStore:
    """Query a Parquet catalog without materializing the whole file in memory."""

    def __init__(self, parquet_path: str | Path, candidate_limit: int | None = None):
        self.path = Path(parquet_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        configured_limit = os.getenv("CATALOG_CANDIDATE_LIMIT")
        self.candidate_limit = (
            candidate_limit
            if candidate_limit is not None
            else int(configured_limit or DEFAULT_CANDIDATE_LIMIT)
        )
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1.")

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
                raise CatalogQueryError(f"Could not query catalog {self.path}: {exc}") from exc
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

    @staticmethod
    def _optimize_dtypes(tracks: pd.DataFrame) -> pd.DataFrame:
        tracks = tracks.copy()
        for column in FLOAT32_COLUMNS:
            if column in tracks:
                tracks[column] = pd.to_numeric(tracks[column], errors="coerce", downcast="float")
        if "duration_ms" in tracks:
            tracks["duration_ms"] = pd.to_numeric(
                tracks["duration_ms"], errors="coerce", downcast="integer"
            )
        return tracks

    @staticmethod
    def _candidate_filters(
        exclude_ids: Iterable[str] | None = None,
        exclude_artists: Iterable[str] | None = None,
        min_popularity: int | None = None,
        max_popularity: int | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> tuple[list[str], list]:
        clauses: list[str] = []
        parameters: list = []

        id_values = exclude_ids if exclude_ids is not None else ()
        ids = list(dict.fromkeys(value for value in id_values if value))
        if ids:
            clauses.append(f"spotify_id NOT IN ({', '.join('?' for _ in ids)})")
            parameters.extend(ids)

        artist_values = exclude_artists if exclude_artists is not None else ()
        artists = list(dict.fromkeys(value for value in artist_values if value))
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

        return clauses, parameters

    def load_tracks(self, spotify_ids: Iterable[str]) -> pd.DataFrame:
        """Load unique catalog rows by Spotify ID, preserving requested ID order."""
        ids = list(dict.fromkeys(value for value in spotify_ids if value))
        if not ids:
            empty = self._query("SELECT * FROM read_parquet(?) LIMIT 0", [str(self.path)])
            return self._optimize_dtypes(empty)

        tracks = self._query(
            f"""
            SELECT * FROM read_parquet(?)
            WHERE spotify_id IN ({", ".join("?" for _ in ids)})
            """,
            [str(self.path), *ids],
        )
        if tracks.empty:
            return tracks

        requested_order = {spotify_id: index for index, spotify_id in enumerate(ids)}
        tracks["_requested_order"] = tracks["spotify_id"].map(requested_order)
        tracks = tracks.sort_values("_requested_order").drop(columns="_requested_order")
        return self._optimize_dtypes(tracks).reset_index(drop=True)

    def count_candidates(
        self,
        exclude_ids: Iterable[str] | None = None,
        exclude_artists: Iterable[str] | None = None,
        min_popularity: int | None = None,
        max_popularity: int | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> int:
        """Count all eligible candidates before applying the configured sample limit."""
        clauses, filter_parameters = self._candidate_filters(
            exclude_ids=exclude_ids,
            exclude_artists=exclude_artists,
            min_popularity=min_popularity,
            max_popularity=max_popularity,
            year_range=year_range,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        result = self._query(
            f"""
            SELECT count(*) AS candidate_count
            FROM read_parquet(?)
            {where}
            """,
            [str(self.path), *filter_parameters],
        )
        return int(result.iloc[0]["candidate_count"])

    def load_candidates(
        self,
        exclude_ids: Iterable[str] | None = None,
        exclude_artists: Iterable[str] | None = None,
        min_popularity: int | None = None,
        max_popularity: int | None = None,
        year_range: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        clauses, filter_parameters = self._candidate_filters(
            exclude_ids=exclude_ids,
            exclude_artists=exclude_artists,
            min_popularity=min_popularity,
            max_popularity=max_popularity,
            year_range=year_range,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters = [str(self.path), *filter_parameters]
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
        return self._optimize_dtypes(candidates)
