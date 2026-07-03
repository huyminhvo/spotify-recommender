import re

import pandas as pd
from spotipy import Spotify

from utils.matcher import match_track


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
    total_tracks = 0
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
            if not track:
                continue
            total_tracks += 1

            if catalog_store is not None:
                match = catalog_store.match_track(track)
            else:
                match = match_track(track, indexes, catalog_df)
            if match is not None:
                matched_rows.append(match)

        results = sp.next(results) if results.get("next") else None

    stats = {"total_tracks": total_tracks, "matched_tracks": len(matched_rows)}
    if not matched_rows:
        print("No tracks from this playlist matched the catalog.")
        empty = pd.DataFrame()
        return (empty, stats) if return_stats else empty

    print(f"Matched {len(matched_rows)} tracks from playlist against catalog")
    matched_df = pd.DataFrame(matched_rows)
    return (matched_df, stats) if return_stats else matched_df
