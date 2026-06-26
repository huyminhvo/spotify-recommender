import pandas as pd
import pytest

from webapp import errors
from webapp import services


class FakeSpotify:
    def __init__(self):
        self.created = None

    def track(self, spotify_id):
        if spotify_id == "bad":
            raise RuntimeError("spotify failure")
        return {"album": {"images": [{"url": f"{spotify_id}-small"}, {"url": f"{spotify_id}-medium"}]}}

    def me(self):
        return {"id": "user-id"}


class FakeSpotifyError(Exception):
    def __init__(self, status):
        self.http_status = status
        super().__init__(f"spotify status {status}")


def test_fetch_album_art_urls_handles_success_missing_and_errors():
    sp = FakeSpotify()

    urls = services.fetch_album_art_urls(sp, ["a", "bad"])

    assert urls == ["a-medium", None]


def test_recommendation_track_uris_drop_missing_ids():
    recs = pd.DataFrame({"spotify_id": ["a", None, "b"]})

    assert services.recommendation_track_uris(recs) == ["spotify:track:a", "spotify:track:b"]


def test_add_recommendations_to_spotify_rejects_empty_recommendations():
    with pytest.raises(errors.NoRecommendationTracksError):
        services.add_recommendations_to_spotify(pd.DataFrame({"spotify_id": [None]}), sp=FakeSpotify())


def test_add_recommendations_to_spotify_creates_playlist(monkeypatch):
    calls = {}

    def fake_create_playlist(sp, user_id, track_uris, name):
        calls["user_id"] = user_id
        calls["track_uris"] = track_uris
        calls["name"] = name
        return "https://spotify.test/playlist"

    monkeypatch.setattr(services, "create_recommendation_playlist", fake_create_playlist)

    url = services.add_recommendations_to_spotify(
        pd.DataFrame({"spotify_id": ["a", "b"]}),
        playlist_name="Test Playlist",
        sp=FakeSpotify(),
    )

    assert url == "https://spotify.test/playlist"
    assert calls == {
        "user_id": "user-id",
        "track_uris": ["spotify:track:a", "spotify:track:b"],
        "name": "Test Playlist",
    }


def test_get_recommendations_orchestrates_services(monkeypatch):
    sp = FakeSpotify()
    bundle = services.CatalogBundle(paths=["catalog.csv"], catalog=pd.DataFrame(), indexes={})
    user_tracks = pd.DataFrame({"spotify_id": ["seed"]})
    recs = pd.DataFrame({"spotify_id": ["rec"]})

    monkeypatch.setattr(services, "get_spotify_client", lambda: sp)
    monkeypatch.setattr(services, "load_catalog_bundle", lambda: bundle)
    monkeypatch.setattr(services, "match_playlist_tracks", lambda sp_arg, playlist_url, bundle_arg: user_tracks)
    monkeypatch.setattr(
        services,
        "generate_recommendations",
        lambda bundle_arg, user_tracks_arg, top_n: recs,
    )

    result = services.get_recommendations("spotify:playlist:test", top_n=1)

    assert result["spotify_id"].tolist() == ["rec"]
    assert result["album_art_url"].tolist() == ["rec-medium"]


def test_match_playlist_tracks_rejects_invalid_playlist_url():
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    with pytest.raises(errors.InvalidPlaylistURLError):
        services.match_playlist_tracks(FakeSpotify(), "not a playlist url!", bundle)


def test_match_playlist_tracks_raises_no_catalog_matches(monkeypatch):
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    monkeypatch.setattr(services, "fetch_playlist_profile", lambda sp, playlist_id, indexes, catalog: pd.DataFrame())

    with pytest.raises(errors.NoCatalogMatchesError):
        services.match_playlist_tracks(FakeSpotify(), "spotify:playlist:abc123", bundle)


def test_match_playlist_tracks_classifies_spotify_rate_limit(monkeypatch):
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    def raise_rate_limit(sp, playlist_id, indexes, catalog):
        raise FakeSpotifyError(429)

    monkeypatch.setattr(services, "fetch_playlist_profile", raise_rate_limit)

    with pytest.raises(errors.SpotifyRateLimitError):
        services.match_playlist_tracks(FakeSpotify(), "spotify:playlist:abc123", bundle)


def test_add_recommendations_to_spotify_classifies_auth_and_access_errors(monkeypatch):
    class AuthFailSpotify(FakeSpotify):
        def me(self):
            raise FakeSpotifyError(401)

    with pytest.raises(errors.SpotifyAuthenticationError):
        services.add_recommendations_to_spotify(pd.DataFrame({"spotify_id": ["a"]}), sp=AuthFailSpotify())

    def raise_access_error(sp, user_id, track_uris, name):
        raise FakeSpotifyError(403)

    monkeypatch.setattr(services, "create_recommendation_playlist", raise_access_error)

    with pytest.raises(errors.SpotifyPlaylistAccessError):
        services.add_recommendations_to_spotify(pd.DataFrame({"spotify_id": ["a"]}), sp=FakeSpotify())


def test_load_catalog_bundle_raises_missing_dataset(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(errors.MissingDatasetError):
        services.load_catalog_bundle(catalog_paths=[str(missing_path)])
