from utils.spotify_playlist import create_recommendation_playlist


def test_create_playlist_uses_current_me_and_items_endpoints():
    class FakeSpotify:
        def __init__(self):
            self.calls = []

        def _post(self, path, payload):
            self.calls.append((path, payload))
            if path == "me/playlists":
                return {
                    "id": "playlist-id",
                    "external_urls": {"spotify": "https://spotify.test/playlist"},
                }
            return {}

    sp = FakeSpotify()
    result = create_recommendation_playlist(
        sp,
        ["spotify:track:a"],
        name="Recommendations",
    )

    assert result == "https://spotify.test/playlist"
    assert sp.calls[0][0] == "me/playlists"
    assert sp.calls[1] == (
        "playlists/playlist-id/items",
        {"uris": ["spotify:track:a"]},
    )
