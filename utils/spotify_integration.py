"""Spotify playlist retrieval and catalog-matching helpers."""

import logging
import re
from collections.abc import Iterator

import pandas as pd
from spotipy import Spotify

from utils.matcher import canon_artist_primary, canon_title, match_track

logger = logging.getLogger(__name__)

MEMBERSHIP_COLUMNS = [
    "playlist_id",
    "position",
    "source_spotify_id",
    "catalog_spotify_id",
    "matched",
    "source_title",
    "source_artist",
]


def extract_playlist_id(url_or_uri: str) -> str:
    """
    Extract a Spotify playlist ID from a full URL, URI, or raw ID string.
    """
    m = re.search(r"playlist/([a-zA-Z0-9]+)", url_or_uri)
    if m:
        return m.group(1)
    m = re.search(r"spotify:playlist:([a-zA-Z0-9]+)", url_or_uri)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9]+", url_or_uri):
        return url_or_uri
    raise ValueError(f"Invalid Spotify playlist link/URI: {url_or_uri}")


def _iter_playlist_tracks(sp: Spotify, playlist_id: str) -> Iterator[tuple[int, dict]]:
    """Yield Spotify track objects with their zero-based playlist positions."""
    position = 0
    # Spotify renamed the Development Mode endpoint from /tracks to /items in
    # 2026. Spotipy versions that still target /tracks cannot use the new API.
    results = sp._get(
        f"playlists/{playlist_id}/items",
        limit=50,
        additional_types="track",
    )
    while results:
        for item in results["items"]:
            # New responses use `item`; tolerate the former shape for Extended
            # Quota apps and older test fixtures.
            track = item.get("item") or item.get("track")
            current_position = position
            position += 1
            if track:
                yield current_position, track

        results = sp.next(results) if results.get("next") else None


def _match_catalog_track(track, indexes=None, catalog_df=None, catalog_store=None):
    """Match a Spotify track using the deployment store or legacy DataFrame path."""
    if catalog_store is not None:
        return catalog_store.match_track(track)
    if indexes is None or catalog_df is None:
        raise ValueError("Provide catalog_store or both indexes and catalog_df.")
    return match_track(track, indexes, catalog_df)


def _source_track_identity(track: dict, position: int) -> tuple:
    """Build a stable deduplication key, retaining metadata-only local tracks."""
    spotify_id = track.get("id")
    if spotify_id:
        return ("spotify_id", str(spotify_id))

    artist_names = [
        artist.get("name", "")
        for artist in (track.get("artists") or [])
        if isinstance(artist, dict)
    ]
    title = canon_title(track.get("name", ""))
    artist = canon_artist_primary(artist_names)
    duration_ms = track.get("duration_ms")
    if title or artist or duration_ms:
        return ("metadata", title, artist, duration_ms)
    return ("position", position)


def fetch_playlist_membership(
    sp: Spotify,
    playlist_id: str,
    indexes=None,
    catalog_df: pd.DataFrame | None = None,
    catalog_store=None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int | float]]:
    """
    Build a label-only playlist membership table and preserve catalog misses.

    Duplicate occurrences of the same source track are collapsed to their first
    playlist position. Catalog fallback matches retain both the source Spotify ID
    and the matched catalog Spotify ID so the match ceiling remains auditable.
    """
    rows = []
    seen_tracks = set()
    total_source_tracks = 0

    for position, track in _iter_playlist_tracks(sp, playlist_id):
        total_source_tracks += 1
        identity = _source_track_identity(track, position)
        if identity in seen_tracks:
            continue
        seen_tracks.add(identity)

        source_spotify_id = track.get("id")
        source_spotify_id = str(source_spotify_id) if source_spotify_id else None
        artist_names = [
            artist.get("name", "")
            for artist in (track.get("artists") or [])
            if isinstance(artist, dict) and artist.get("name")
        ]
        match = _match_catalog_track(
            track,
            indexes=indexes,
            catalog_df=catalog_df,
            catalog_store=catalog_store,
        )
        catalog_spotify_id = match.get("spotify_id") if match is not None else None
        if pd.isna(catalog_spotify_id):
            catalog_spotify_id = None
        elif catalog_spotify_id is not None:
            catalog_spotify_id = str(catalog_spotify_id)

        rows.append(
            {
                "playlist_id": playlist_id,
                "position": position,
                "source_spotify_id": source_spotify_id,
                "catalog_spotify_id": catalog_spotify_id,
                "matched": catalog_spotify_id is not None,
                "source_title": track.get("name"),
                "source_artist": ", ".join(artist_names) or None,
            }
        )

    membership = pd.DataFrame(rows, columns=MEMBERSHIP_COLUMNS)
    matched_unique_tracks = int(membership["matched"].sum()) if not membership.empty else 0
    total_unique_source_tracks = len(membership)
    stats: dict[str, int | float] = {
        "total_source_tracks": total_source_tracks,
        "total_unique_source_tracks": total_unique_source_tracks,
        "duplicate_tracks_removed": total_source_tracks - total_unique_source_tracks,
        "matched_unique_tracks": matched_unique_tracks,
        "match_rate": (
            matched_unique_tracks / total_unique_source_tracks
            if total_unique_source_tracks
            else 0.0
        ),
        # Backward-compatible aliases now intentionally use unique-track counts.
        "total_tracks": total_unique_source_tracks,
        "matched_tracks": matched_unique_tracks,
    }
    return (membership, stats) if return_stats else membership


def fetch_playlist_profile(
    sp: Spotify,
    playlist_id: str,
    indexes=None,
    catalog_df: pd.DataFrame | None = None,
    catalog_store=None,
    return_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, int]]:
    """
    Fetch a playlist from Spotify, normalize its tracks, and match them against
    the catalog using prebuilt indexes + DataFrame.
    Returns a DataFrame of matched rows (with features).
    """
    matched_rows = []
    seen_tracks = set()
    total_tracks = 0
    for position, track in _iter_playlist_tracks(sp, playlist_id):
        total_tracks += 1
        identity = _source_track_identity(track, position)
        if identity in seen_tracks:
            continue
        seen_tracks.add(identity)
        match = _match_catalog_track(
            track,
            indexes=indexes,
            catalog_df=catalog_df,
            catalog_store=catalog_store,
        )
        if match is not None:
            matched_rows.append(match)

    stats = {"total_tracks": total_tracks, "matched_tracks": len(matched_rows)}
    if not matched_rows:
        logger.info("No tracks from playlist %s matched the catalog", playlist_id)
        empty = pd.DataFrame()
        return (empty, stats) if return_stats else empty

    logger.info(
        "Matched %d tracks from playlist %s against the catalog",
        len(matched_rows),
        playlist_id,
    )
    matched_df = pd.DataFrame(matched_rows)
    return (matched_df, stats) if return_stats else matched_df
