import pytest
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from utils.spotify_auth import (
    SpotifyConfig,
    create_oauth_state,
    create_user_oauth,
    decode_oauth_state,
    get_public_spotify_client,
)

CONFIG = SpotifyConfig("client-id", "client-secret", "https://example.test/")


def test_public_client_uses_client_credentials_without_filesystem_cache():
    client = get_public_spotify_client(CONFIG)

    assert isinstance(client.auth_manager, SpotifyClientCredentials)
    assert isinstance(client.auth_manager.cache_handler, MemoryCacheHandler)


def test_user_oauth_uses_only_memory_cache_and_expected_scopes():
    token = {"access_token": "user-token"}
    oauth, cache = create_user_oauth(CONFIG, token)

    assert isinstance(oauth, SpotifyOAuth)
    assert isinstance(cache, MemoryCacheHandler)
    assert cache.get_cached_token() == token
    assert "playlist-read-private" in oauth.scope
    assert "playlist-modify-private" in oauth.scope
    assert oauth.redirect_uri == "https://example.test/"


def test_signed_oauth_state_survives_session_reconnect():
    state = create_oauth_state(
        CONFIG,
        "browser",
        {"action": "recommend", "playlist_url": "spotify:playlist:test"},
    )

    request = decode_oauth_state(CONFIG, state, "browser")

    assert request == {
        "action": "recommend",
        "playlist_url": "spotify:playlist:test",
    }


def test_oauth_state_rejects_tampering_and_wrong_browser():
    state = create_oauth_state(CONFIG, "browser", {"action": "recommend"})

    with pytest.raises(ValueError, match="invalid"):
        decode_oauth_state(CONFIG, state[:-1] + "x", "browser")
    with pytest.raises(ValueError, match="browser binding"):
        decode_oauth_state(CONFIG, state, "other-browser")
