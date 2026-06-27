import pytest

from utils.spotify_integration import extract_playlist_id


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
