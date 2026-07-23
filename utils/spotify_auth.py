from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.cache_handler import CacheFileHandler, MemoryCacheHandler
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

USER_PLAYLIST_SCOPE = "playlist-read-private playlist-modify-private"
EVALUATION_PLAYLIST_SCOPE = "playlist-read-private"
OAUTH_STATE_MAX_AGE_SECONDS = 10 * 60
DEFAULT_EVALUATION_TOKEN_CACHE = Path(".spotify_cache") / "evaluation-token.json"


@dataclass(frozen=True)
class SpotifyConfig:
    client_id: str
    client_secret: str
    redirect_uri: str | None = None


def create_oauth_state(
    config: SpotifyConfig,
    browser_binding: str,
    request: Mapping[str, object],
) -> str:
    """Create a signed, short-lived state that survives a Streamlit reconnect."""
    payload = {
        "issued_at": int(time.time()),
        "nonce": secrets.token_urlsafe(24),
        "binding": browser_binding,
        "request": dict(request),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).rstrip(b"=")
    signature = hmac.new(config.client_secret.encode(), encoded, hashlib.sha256).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return f"{encoded.decode()}.{encoded_signature.decode()}"


def decode_oauth_state(
    config: SpotifyConfig,
    state: str,
    browser_binding: str,
    now: int | None = None,
) -> dict[str, object]:
    """Validate and decode state without relying on Streamlit session memory."""
    try:
        encoded, encoded_signature = state.split(".", 1)
        expected_signature = hmac.new(
            config.client_secret.encode(), encoded.encode(), hashlib.sha256
        ).digest()
        signature = base64.urlsafe_b64decode(
            encoded_signature + "=" * (-len(encoded_signature) % 4)
        )
        expected_encoded_signature = (
            base64.urlsafe_b64encode(expected_signature).rstrip(b"=").decode()
        )
        if not hmac.compare_digest(encoded_signature, expected_encoded_signature):
            raise ValueError("OAuth state signature encoding did not match.")
        if not hmac.compare_digest(signature, expected_signature):
            raise ValueError("OAuth state signature did not match.")
        payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("OAuth state was invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("OAuth state payload was invalid.")

    current_time = int(time.time()) if now is None else now
    issued_at = payload.get("issued_at")
    if (
        not isinstance(issued_at, int)
        or not 0 <= current_time - issued_at <= OAUTH_STATE_MAX_AGE_SECONDS
    ):
        raise ValueError("OAuth state expired.")
    if not hmac.compare_digest(str(payload.get("binding", "")), browser_binding):
        raise ValueError("OAuth state browser binding did not match.")
    request = payload.get("request")
    if not isinstance(request, dict):
        raise ValueError("OAuth state request was invalid.")
    return request


def get_spotify_config(secrets: Mapping[str, object] | None = None) -> SpotifyConfig:
    """Read Spotify settings from Streamlit secrets or local environment."""
    load_dotenv()
    secrets = secrets or {}
    client_id = str(secrets.get("SPOTIPY_CLIENT_ID") or os.getenv("SPOTIPY_CLIENT_ID") or "")
    client_secret = str(
        secrets.get("SPOTIPY_CLIENT_SECRET") or os.getenv("SPOTIPY_CLIENT_SECRET") or ""
    )
    redirect_uri = str(
        secrets.get("SPOTIPY_REDIRECT_URI") or os.getenv("SPOTIPY_REDIRECT_URI") or ""
    )
    if not client_id or not client_secret:
        raise ValueError("SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET must be configured.")
    return SpotifyConfig(client_id, client_secret, redirect_uri or None)


def get_public_spotify_client(config: SpotifyConfig | None = None) -> spotipy.Spotify:
    """Create an app-only client for public Spotify data."""
    config = config or get_spotify_config()
    return spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=config.client_id,
            client_secret=config.client_secret,
            cache_handler=MemoryCacheHandler(),
        )
    )


def create_user_oauth(
    config: SpotifyConfig,
    token_info: dict | None = None,
) -> tuple[SpotifyOAuth, MemoryCacheHandler]:
    """Create a user OAuth manager whose token exists only in this session."""
    if not config.redirect_uri:
        raise ValueError("SPOTIPY_REDIRECT_URI must be configured for Spotify authorization.")
    cache_handler = MemoryCacheHandler(token_info=token_info)
    oauth = SpotifyOAuth(
        scope=USER_PLAYLIST_SCOPE,
        client_id=config.client_id,
        client_secret=config.client_secret,
        redirect_uri=config.redirect_uri,
        cache_handler=cache_handler,
        open_browser=False,
    )
    return oauth, cache_handler


def get_user_spotify_client(
    config: SpotifyConfig,
    token_info: dict,
) -> tuple[spotipy.Spotify, MemoryCacheHandler]:
    oauth, cache_handler = create_user_oauth(config, token_info)
    return spotipy.Spotify(auth_manager=oauth), cache_handler


def create_cached_user_oauth(
    config: SpotifyConfig,
    cache_path: str | Path = DEFAULT_EVALUATION_TOKEN_CACHE,
    scope: str = EVALUATION_PLAYLIST_SCOPE,
) -> SpotifyOAuth:
    """Create OAuth backed by a local, gitignored token cache.

    This is intended for command-line development tools, not the multi-user web app.
    SpotifyOAuth refreshes expired access tokens when the cached token contains a
    refresh token.
    """
    if not config.redirect_uri:
        raise ValueError("SPOTIPY_REDIRECT_URI must be configured for Spotify authorization.")

    path = Path(cache_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_handler = CacheFileHandler(cache_path=str(path))
    return SpotifyOAuth(
        scope=scope,
        client_id=config.client_id,
        client_secret=config.client_secret,
        redirect_uri=config.redirect_uri,
        cache_handler=cache_handler,
        open_browser=False,
    )


def get_cached_user_spotify_client(
    config: SpotifyConfig | None = None,
    cache_path: str | Path = DEFAULT_EVALUATION_TOKEN_CACHE,
) -> spotipy.Spotify:
    """Return a user client, refreshing a valid cached token when necessary."""
    oauth = create_cached_user_oauth(config or get_spotify_config(), cache_path)
    token_info = oauth.validate_token(oauth.cache_handler.get_cached_token())
    if not token_info:
        raise ValueError(
            "No valid cached Spotify user authorization. Run "
            "`python scripts/authorize_spotify.py` first."
        )
    return spotipy.Spotify(auth_manager=oauth)


# Backwards-compatible name for callers that only need public catalog access.
get_spotify_client = get_public_spotify_client
