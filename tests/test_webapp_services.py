import pandas as pd
import pytest

from webapp import errors, services


class FakeSpotify:
    def __init__(self):
        self.created = None
        self.track_ids = []

    def track(self, spotify_id):
        self.track_ids.append(spotify_id)
        if spotify_id == "bad":
            raise RuntimeError("spotify failure")
        return {
            "album": {
                "images": [
                    {"url": f"{spotify_id}-small"},
                    {"url": f"{spotify_id}-medium"},
                ]
            }
        }

    def me(self):
        return {"id": "user-id"}


class FakeSpotifyError(Exception):
    def __init__(self, status):
        self.http_status = status
        super().__init__(f"spotify status {status}")


def test_fetch_album_art_urls_handles_success_and_errors():
    sp = FakeSpotify()

    assert services.fetch_album_art_urls(sp, ["a"]) == ["a-medium"]
    assert services.fetch_album_art_urls(sp, ["bad"]) == [None]


def test_fetch_album_art_urls_uses_current_individual_endpoint_and_preserves_order():
    sp = FakeSpotify()
    spotify_ids = [f"track-{index}" for index in range(51)]

    urls = services.fetch_album_art_urls(sp, spotify_ids)

    assert sp.track_ids == spotify_ids
    assert urls == [f"{spotify_id}-medium" for spotify_id in spotify_ids]


def test_fetch_album_art_urls_handles_missing_tracks_and_images():
    class IncompleteSpotify:
        def track(self, spotify_id):
            return None if spotify_id == "a" else {"album": {"images": []}}

    assert services.fetch_album_art_urls(IncompleteSpotify(), ["a", "b", "c"]) == [
        None,
        None,
        None,
    ]


def test_recommendation_track_uris_drop_missing_ids():
    recs = pd.DataFrame({"spotify_id": ["a", None, "b"]})

    assert services.recommendation_track_uris(recs) == ["spotify:track:a", "spotify:track:b"]


def test_add_recommendations_to_spotify_rejects_empty_recommendations():
    with pytest.raises(errors.NoRecommendationTracksError):
        services.add_recommendations_to_spotify(
            pd.DataFrame({"spotify_id": [None]}), sp=FakeSpotify()
        )


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
    public_sp = FakeSpotify()
    bundle = services.CatalogBundle(paths=["catalog.csv"], catalog=pd.DataFrame(), indexes={})
    user_tracks = pd.DataFrame({"spotify_id": ["seed"]})
    recs = pd.DataFrame({"spotify_id": ["rec"]})

    monkeypatch.setattr(services, "load_catalog_bundle", lambda: bundle)
    monkeypatch.setattr(
        services, "match_playlist_tracks", lambda sp_arg, playlist_url, bundle_arg: user_tracks
    )

    def fake_generate_recommendations(
        bundle_arg, user_tracks_arg, top_n, adjustments, exclude_spotify_ids
    ):
        assert list(exclude_spotify_ids) == ["seen"]
        return recs

    monkeypatch.setattr(services, "generate_recommendations", fake_generate_recommendations)

    result = services.get_recommendations(
        "spotify:playlist:test",
        top_n=1,
        sp=sp,
        public_sp=public_sp,
        exclude_spotify_ids=["seen"],
    )

    assert result["spotify_id"].tolist() == ["rec"]
    assert result["album_art_url"].tolist() == ["rec-medium"]
    assert public_sp.track_ids == ["rec"]


def test_get_recommendations_uses_injected_catalog_bundle(monkeypatch):
    sp = FakeSpotify()
    bundle = services.CatalogBundle(paths=["catalog.parquet"], catalog=pd.DataFrame())

    def fail_if_loaded():
        raise AssertionError("catalog should come from the Streamlit resource cache")

    monkeypatch.setattr(services, "load_catalog_bundle", fail_if_loaded)
    monkeypatch.setattr(
        services, "match_playlist_tracks", lambda sp_arg, playlist_url, bundle_arg: pd.DataFrame()
    )
    monkeypatch.setattr(
        services,
        "generate_recommendations",
        lambda bundle_arg, user_tracks_arg, top_n, adjustments, exclude_spotify_ids: pd.DataFrame(
            {"spotify_id": []}
        ),
    )

    services.get_recommendations(
        "spotify:playlist:test",
        sp=sp,
        public_sp=FakeSpotify(),
        catalog_bundle=bundle,
    )


def test_get_recommendations_requires_user_authorization():
    with pytest.raises(errors.SpotifyAuthenticationError, match="Connect Spotify"):
        services.get_recommendations("spotify:playlist:test")


@pytest.mark.parametrize(
    ("artists_raw", "expected"),
    [
        (["AJR"], "AJR"),
        (["Artist One", "Artist Two"], "Artist One, Artist Two"),
        ("['From Indian Lakes']", "From Indian Lakes"),
        ('["GAMMAL", "Guest"]', "GAMMAL, Guest"),
        (("Pablo",), "Pablo"),
        (None, "Unknown artist"),
        ("", "Unknown artist"),
    ],
)
def test_format_artist_names(artists_raw, expected):
    assert services.format_artist_names(artists_raw) == expected


def test_match_playlist_tracks_rejects_invalid_playlist_url():
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    with pytest.raises(errors.InvalidPlaylistURLError):
        services.match_playlist_tracks(FakeSpotify(), "not a playlist url!", bundle)


def test_match_playlist_tracks_raises_no_catalog_matches(monkeypatch):
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    monkeypatch.setattr(
        services, "fetch_playlist_profile", lambda sp, playlist_id, indexes, catalog: pd.DataFrame()
    )

    with pytest.raises(errors.NoCatalogMatchesError):
        services.match_playlist_tracks(FakeSpotify(), "spotify:playlist:abc123", bundle)


def test_match_playlist_tracks_classifies_spotify_rate_limit(monkeypatch):
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    def raise_rate_limit(sp, playlist_id, indexes, catalog):
        raise FakeSpotifyError(429)

    monkeypatch.setattr(services, "fetch_playlist_profile", raise_rate_limit)

    with pytest.raises(errors.SpotifyRateLimitError):
        services.match_playlist_tracks(FakeSpotify(), "spotify:playlist:abc123", bundle)


def test_match_playlist_tracks_classifies_catalog_decompression_failure(monkeypatch):
    bundle = services.CatalogBundle(paths=[], catalog=pd.DataFrame(), indexes={})

    def raise_catalog_failure(sp, playlist_id, indexes, catalog):
        raise RuntimeError("ZSTD Decompression failure")

    monkeypatch.setattr(services, "fetch_playlist_profile", raise_catalog_failure)

    with pytest.raises(errors.CatalogReadError):
        services.match_playlist_tracks(FakeSpotify(), "spotify:playlist:abc123", bundle)


def test_add_recommendations_to_spotify_classifies_auth_and_access_errors(monkeypatch):
    class AuthFailSpotify(FakeSpotify):
        def me(self):
            raise FakeSpotifyError(401)

    with pytest.raises(errors.SpotifyAuthenticationError):
        services.add_recommendations_to_spotify(
            pd.DataFrame({"spotify_id": ["a"]}), sp=AuthFailSpotify()
        )

    def raise_access_error(sp, user_id, track_uris, name):
        raise FakeSpotifyError(403)

    monkeypatch.setattr(services, "create_recommendation_playlist", raise_access_error)

    with pytest.raises(errors.SpotifyPlaylistAccessError):
        services.add_recommendations_to_spotify(
            pd.DataFrame({"spotify_id": ["a"]}), sp=FakeSpotify()
        )


def test_load_catalog_bundle_raises_missing_dataset(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(errors.MissingDatasetError):
        services.load_catalog_bundle(catalog_paths=[str(missing_path)])


def test_load_catalog_bundle_uses_deployment_manifest(monkeypatch, tmp_path):
    catalog_dir = tmp_path / "catalog"
    catalog_dir.mkdir()
    parquet_path = catalog_dir / "catalog-v1-contenthash.parquet"
    parquet_path.write_bytes(b"parquet-placeholder")
    manifest = catalog_dir / "CURRENT"
    manifest.write_text(f"{parquet_path.name}\n", encoding="utf-8")
    monkeypatch.setattr(services, "CATALOG_MANIFEST_PATH", manifest)
    monkeypatch.delenv("CATALOG_PARQUET_PATH", raising=False)

    bundle = services.load_catalog_bundle()

    assert bundle.paths == [str(parquet_path)]
    assert bundle.catalog.path == parquet_path.resolve()


def test_load_catalog_bundle_does_not_fall_back_to_raw_data(monkeypatch, tmp_path):
    manifest = tmp_path / "missing" / "CURRENT"
    monkeypatch.setattr(services, "CATALOG_MANIFEST_PATH", manifest)
    monkeypatch.delenv("CATALOG_PARQUET_PATH", raising=False)
    monkeypatch.setattr(
        services,
        "get_merged_dataset",
        lambda *args, **kwargs: pytest.fail("raw catalog rebuild was attempted"),
    )

    with pytest.raises(services.MissingDatasetError, match="manifest is missing"):
        services.load_catalog_bundle()
