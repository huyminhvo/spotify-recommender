import pandas as pd
import pytest

from scripts import build_evaluation_dataset as builder


def _catalog_row(spotify_id):
    return {
        "spotify_id": spotify_id,
        "title_raw": f"Track {spotify_id}",
        "artists_raw": ["Artist"],
        "artist_primary_canon": "artist",
        "duration_ms": 210_000,
        "popularity": 50,
        "release_year": 2020,
        "danceability": 0.5,
        "energy": 0.5,
        "valence": 0.5,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "speechiness": 0.05,
        "tempo": 120.0,
        "loudness": -8.0,
    }


def test_load_playlist_inputs_dedupes_urls_and_reads_file(tmp_path):
    playlist_file = tmp_path / "playlists.txt"
    playlist_file.write_text(
        "\n".join(
            [
                "# comment",
                "spotify:playlist:file-id",
                "",
                "spotify:playlist:duplicate",
            ]
        ),
        encoding="utf-8",
    )

    inputs = builder.load_playlist_inputs(
        ["spotify:playlist:duplicate", "spotify:playlist:inline-id"],
        str(playlist_file),
    )

    assert inputs == [
        "spotify:playlist:duplicate",
        "spotify:playlist:inline-id",
        "spotify:playlist:file-id",
    ]


def _membership_row(
    playlist_id,
    position,
    source_spotify_id,
    catalog_spotify_id=None,
):
    return {
        "playlist_id": playlist_id,
        "position": position,
        "source_spotify_id": source_spotify_id,
        "catalog_spotify_id": catalog_spotify_id,
        "matched": catalog_spotify_id is not None,
        "source_title": f"Source {source_spotify_id}",
        "source_artist": "Artist",
    }


def test_build_evaluation_dataset_preserves_unmatched_labels_and_skips_small_playlists(
    monkeypatch,
):
    catalog = pd.DataFrame([_catalog_row("a"), _catalog_row("b"), _catalog_row("c")])

    def fake_fetch_playlist_membership(
        sp,
        playlist_id,
        indexes=None,
        catalog_df=None,
        catalog_store=None,
        return_stats=False,
    ):
        if playlist_id == "small":
            df = pd.DataFrame(
                [
                    _membership_row("small", 0, "a", "a"),
                    _membership_row("small", 1, "missing"),
                ]
            )
            stats = {
                "total_source_tracks": 2,
                "total_unique_source_tracks": 2,
                "duplicate_tracks_removed": 0,
                "matched_unique_tracks": 1,
            }
        else:
            df = pd.DataFrame(
                [
                    _membership_row("keep", 0, "a", "a"),
                    _membership_row("keep", 2, "missing"),
                    _membership_row("keep", 3, "b", "b"),
                ]
            )
            stats = {
                "total_source_tracks": 4,
                "total_unique_source_tracks": 3,
                "duplicate_tracks_removed": 1,
                "matched_unique_tracks": 2,
            }
        return (df, stats) if return_stats else df

    monkeypatch.setattr(builder, "fetch_playlist_membership", fake_fetch_playlist_membership)

    dataset, summary = builder.build_evaluation_dataset(
        sp=object(),
        catalog_df=catalog,
        playlist_inputs=["spotify:playlist:keep", "spotify:playlist:small"],
        min_matched_tracks=2,
        raw_catalog_rows=5,
    )

    assert dataset.columns.tolist() == [
        "playlist_id",
        "position",
        "source_spotify_id",
        "catalog_spotify_id",
        "matched",
        "source_title",
        "source_artist",
    ]
    assert dataset["playlist_id"].tolist() == ["keep", "keep", "keep"]
    assert dataset["position"].tolist() == [0, 2, 3]
    assert dataset["source_spotify_id"].tolist() == ["a", "missing", "b"]
    assert dataset["matched"].tolist() == [True, False, True]
    assert summary[
        [
            "playlist_id",
            "total_source_tracks",
            "total_unique_source_tracks",
            "duplicate_tracks_removed",
            "matched_unique_tracks",
            "total_tracks",
            "matched_tracks",
            "match_rate",
            "included",
            "raw_catalog_rows",
            "merged_catalog_rows",
            "duplicate_reduction_rate",
        ]
    ].to_dict("records") == [
        {
            "playlist_id": "keep",
            "total_source_tracks": 4,
            "total_unique_source_tracks": 3,
            "duplicate_tracks_removed": 1,
            "matched_unique_tracks": 2,
            "total_tracks": 3,
            "matched_tracks": 2,
            "match_rate": 2 / 3,
            "included": True,
            "raw_catalog_rows": 5,
            "merged_catalog_rows": 3,
            "duplicate_reduction_rate": 0.4,
        },
        {
            "playlist_id": "small",
            "total_source_tracks": 2,
            "total_unique_source_tracks": 2,
            "duplicate_tracks_removed": 0,
            "matched_unique_tracks": 1,
            "total_tracks": 2,
            "matched_tracks": 1,
            "match_rate": 0.5,
            "included": False,
            "raw_catalog_rows": 5,
            "merged_catalog_rows": 3,
            "duplicate_reduction_rate": 0.4,
        },
    ]


def test_build_evaluation_dataset_prefers_catalog_store_path(monkeypatch):
    store = object()
    seen = {}

    def fake_fetch_playlist_membership(
        sp,
        playlist_id,
        indexes=None,
        catalog_df=None,
        catalog_store=None,
        return_stats=False,
    ):
        seen.update(
            {
                "playlist_id": playlist_id,
                "indexes": indexes,
                "catalog_df": catalog_df,
                "catalog_store": catalog_store,
            }
        )
        membership = pd.DataFrame([_membership_row(playlist_id, 0, "a", "a")])
        stats = {
            "total_source_tracks": 1,
            "total_unique_source_tracks": 1,
            "duplicate_tracks_removed": 0,
            "matched_unique_tracks": 1,
        }
        return (membership, stats) if return_stats else membership

    monkeypatch.setattr(builder, "fetch_playlist_membership", fake_fetch_playlist_membership)

    dataset, summary = builder.build_evaluation_dataset(
        sp=object(),
        catalog_store=store,
        catalog_rows=2_206_451,
        playlist_inputs=["spotify:playlist:keep"],
        min_matched_tracks=1,
    )

    assert len(dataset) == 1
    assert seen == {
        "playlist_id": "keep",
        "indexes": None,
        "catalog_df": None,
        "catalog_store": store,
    }
    assert summary.loc[0, "merged_catalog_rows"] == 2_206_451


def test_deployment_catalog_path_resolves_manifest_artifact(tmp_path):
    catalog_dir = tmp_path / "data" / "catalog"
    catalog_dir.mkdir(parents=True)
    artifact = catalog_dir / "catalog-v1-test.parquet"
    artifact.touch()
    (catalog_dir / "CURRENT").write_text(f"{artifact.name}\n", encoding="utf-8")

    assert builder.deployment_catalog_path(tmp_path) == artifact


def test_deployment_catalog_path_rejects_manifest_traversal(tmp_path):
    catalog_dir = tmp_path / "data" / "catalog"
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "CURRENT").write_text("../outside.parquet\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid deployment catalog manifest"):
        builder.deployment_catalog_path(tmp_path)


def test_playlist_spotify_client_prefers_user_access_token(monkeypatch):
    seen = {}
    client = object()
    monkeypatch.setenv("TEST_SPOTIFY_TOKEN", "user-token")
    monkeypatch.setattr(
        builder.spotipy,
        "Spotify",
        lambda auth: seen.update(auth=auth) or client,
    )
    monkeypatch.setattr(
        builder,
        "get_spotify_client",
        lambda: pytest.fail("app-only client should not be used"),
    )

    result = builder.get_playlist_spotify_client("TEST_SPOTIFY_TOKEN")

    assert result is client
    assert seen == {"auth": "user-token"}


def test_playlist_spotify_client_uses_refreshable_cache(monkeypatch, tmp_path):
    client = object()
    cache_path = tmp_path / "token.json"
    monkeypatch.delenv("TEST_SPOTIFY_TOKEN", raising=False)
    monkeypatch.setattr(
        builder,
        "get_cached_user_spotify_client",
        lambda cache_path: client,
    )
    monkeypatch.setattr(
        builder,
        "get_spotify_client",
        lambda: pytest.fail("app-only client should not be used"),
    )

    result = builder.get_playlist_spotify_client("TEST_SPOTIFY_TOKEN", cache_path)

    assert result is client


def test_playlist_spotify_client_falls_back_to_app_credentials(monkeypatch):
    client = object()
    monkeypatch.delenv("TEST_SPOTIFY_TOKEN", raising=False)
    monkeypatch.setattr(
        builder,
        "get_cached_user_spotify_client",
        lambda cache_path: (_ for _ in ()).throw(ValueError("no cached token")),
    )
    monkeypatch.setattr(builder, "get_spotify_client", lambda: client)

    assert builder.get_playlist_spotify_client("TEST_SPOTIFY_TOKEN") is client
