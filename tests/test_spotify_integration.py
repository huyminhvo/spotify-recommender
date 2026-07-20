import pandas as pd
import pytest

from utils.spotify_integration import (
    MEMBERSHIP_COLUMNS,
    extract_playlist_id,
    fetch_playlist_membership,
    fetch_playlist_profile,
)


def test_extract_playlist_id_from_url_uri_and_raw_id():
    assert (
        extract_playlist_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")
        == "37i9dQZF1DXcBWIGoYBM5M"
    )
    assert (
        extract_playlist_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M?si=abc123")
        == "37i9dQZF1DXcBWIGoYBM5M"
    )
    assert (
        extract_playlist_id("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M") == "37i9dQZF1DXcBWIGoYBM5M"
    )
    assert extract_playlist_id("37i9dQZF1DXcBWIGoYBM5M") == "37i9dQZF1DXcBWIGoYBM5M"


def test_extract_playlist_id_rejects_invalid_input():
    with pytest.raises(ValueError, match="Invalid Spotify playlist"):
        extract_playlist_id("https://open.spotify.com/track/not-a-playlist")


def test_fetch_playlist_profile_uses_2026_items_endpoint_and_shape():
    class FakeSpotify:
        def __init__(self):
            self.path = None

        def _get(self, path, **kwargs):
            self.path = path
            return {"items": [{"item": {"id": "track-id"}}], "next": None}

    sp = FakeSpotify()
    catalog = pd.DataFrame([{"spotify_id": "track-id"}])
    indexes = {"by_id": {"track-id": 0}, "by_key": {}, "by_artist": {}}

    result = fetch_playlist_profile(sp, "playlist-id", indexes, catalog)

    assert sp.path == "playlists/playlist-id/items"
    assert result["spotify_id"].tolist() == ["track-id"]


def test_fetch_playlist_membership_preserves_misses_and_first_duplicate_position():
    class FakeSpotify:
        def __init__(self):
            self.path = None
            self.second_page = {
                "items": [
                    {
                        "item": {
                            "id": None,
                            "name": "Local Song",
                            "artists": [{"name": "Local Artist"}],
                            "duration_ms": 180_000,
                        }
                    }
                ],
                "next": None,
            }

        def _get(self, path, **kwargs):
            self.path = path
            return {
                "items": [
                    {
                        "item": {
                            "id": "source-match",
                            "name": "Matched Song",
                            "artists": [{"name": "Matched Artist"}],
                        }
                    },
                    {
                        "item": {
                            "id": "source-match",
                            "name": "Matched Song",
                            "artists": [{"name": "Matched Artist"}],
                        }
                    },
                    {
                        "track": {
                            "id": "source-miss",
                            "name": "Missing Song",
                            "artists": [{"name": "Missing Artist"}],
                        }
                    },
                ],
                "next": "page-2",
            }

        def next(self, results):
            return self.second_page

    class FakeCatalogStore:
        def __init__(self):
            self.matched_ids = []

        def match_track(self, track):
            self.matched_ids.append(track.get("id"))
            if track.get("id") == "source-match":
                return {"spotify_id": "catalog-match"}
            return None

    sp = FakeSpotify()
    store = FakeCatalogStore()

    membership, stats = fetch_playlist_membership(
        sp,
        "playlist-id",
        catalog_store=store,
        return_stats=True,
    )

    assert sp.path == "playlists/playlist-id/items"
    assert membership.columns.tolist() == MEMBERSHIP_COLUMNS
    assert membership["position"].tolist() == [0, 2, 3]
    assert membership["source_spotify_id"].tolist()[:2] == ["source-match", "source-miss"]
    assert pd.isna(membership.loc[2, "source_spotify_id"])
    assert membership["catalog_spotify_id"].tolist()[:2] == ["catalog-match", None]
    assert membership["matched"].tolist() == [True, False, False]
    assert membership["source_title"].tolist() == [
        "Matched Song",
        "Missing Song",
        "Local Song",
    ]
    assert store.matched_ids == ["source-match", "source-miss", None]
    assert stats == {
        "total_source_tracks": 4,
        "total_unique_source_tracks": 3,
        "duplicate_tracks_removed": 1,
        "matched_unique_tracks": 1,
        "match_rate": 1 / 3,
        "total_tracks": 3,
        "matched_tracks": 1,
    }


def test_fetch_playlist_membership_supports_dataframe_indexes():
    class FakeSpotify:
        def _get(self, path, **kwargs):
            return {
                "items": [
                    {
                        "item": {
                            "id": "track-id",
                            "name": "Track",
                            "artists": [{"name": "Artist"}],
                        }
                    }
                ],
                "next": None,
            }

    catalog = pd.DataFrame([{"spotify_id": "track-id"}])
    indexes = {"by_id": {"track-id": 0}, "by_key": {}, "by_artist": {}}

    membership = fetch_playlist_membership(
        FakeSpotify(),
        "playlist-id",
        indexes=indexes,
        catalog_df=catalog,
    )

    assert membership[["source_spotify_id", "catalog_spotify_id", "matched"]].to_dict(
        "records"
    ) == [
        {
            "source_spotify_id": "track-id",
            "catalog_spotify_id": "track-id",
            "matched": True,
        }
    ]
