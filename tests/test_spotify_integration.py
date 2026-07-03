import pandas as pd
import pytest

from utils.spotify_integration import extract_playlist_id, fetch_playlist_profile


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
