import pytest
from spotipy.cache_handler import CacheFileHandler, MemoryCacheHandler
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from utils.spotify_auth import (
    SpotifyConfig,
    create_cached_user_oauth,
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


def test_cached_user_oauth_uses_explicit_file_and_read_only_scope(tmp_path):
    cache_path = tmp_path / "private" / "spotify-token.json"

    oauth = create_cached_user_oauth(CONFIG, cache_path)

    assert isinstance(oauth, SpotifyOAuth)
    assert isinstance(oauth.cache_handler, CacheFileHandler)
    assert oauth.cache_handler.cache_path == str(cache_path.resolve())
    assert oauth.scope == "playlist-read-private"
    assert cache_path.parent.is_dir()


def test_cached_user_oauth_requires_redirect_uri(tmp_path):
    with pytest.raises(ValueError, match="SPOTIPY_REDIRECT_URI"):
        create_cached_user_oauth(
            SpotifyConfig("client-id", "client-secret"),
            tmp_path / "token.json",
        )


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
