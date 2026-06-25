import pandas as pd

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


def test_build_evaluation_dataset_adds_playlist_id_and_skips_small_playlists(monkeypatch):
    catalog = pd.DataFrame([_catalog_row("a"), _catalog_row("b"), _catalog_row("c")])

    def fake_fetch_playlist_profile(sp, playlist_id, indexes, catalog_df, return_stats=False):
        if playlist_id == "small":
            df = pd.DataFrame([_catalog_row("a")])
        else:
            df = pd.DataFrame([_catalog_row("a"), _catalog_row("b"), _catalog_row("b")])
        stats = {"total_tracks": len(df), "matched_tracks": len(df)}
        return (df, stats) if return_stats else df

    monkeypatch.setattr(builder, "fetch_playlist_profile", fake_fetch_playlist_profile)

    dataset, summary = builder.build_evaluation_dataset(
        sp=object(),
        catalog_df=catalog,
        playlist_inputs=["spotify:playlist:keep", "spotify:playlist:small"],
        min_matched_tracks=2,
        raw_catalog_rows=5,
    )

    assert dataset["playlist_id"].tolist() == ["keep", "keep"]
    assert dataset["spotify_id"].tolist() == ["a", "b"]
    assert summary[
        [
            "playlist_id",
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
            "total_tracks": 3,
            "matched_tracks": 3,
            "match_rate": 1.0,
            "included": True,
            "raw_catalog_rows": 5,
            "merged_catalog_rows": 3,
            "duplicate_reduction_rate": 0.4,
        },
        {
            "playlist_id": "small",
            "total_tracks": 1,
            "matched_tracks": 1,
            "match_rate": 1.0,
            "included": False,
            "raw_catalog_rows": 5,
            "merged_catalog_rows": 3,
            "duplicate_reduction_rate": 0.4,
        },
    ]
