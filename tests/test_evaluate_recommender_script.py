from argparse import Namespace

import pandas as pd

from scripts import evaluate_recommender


def _args(**overrides):
    defaults = {
        "catalog_csv": None,
        "playlist_url": [],
        "playlist_file": None,
        "raw_catalog_path": None,
        "output_csv": "unused.csv",
        "summary_csv": None,
        "min_matched_tracks": 10,
        "force_rebuild_catalog": False,
        "playlist_col": "playlist_id",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_or_build_catalog_reads_prebuilt_csv(tmp_path):
    csv_path = tmp_path / "eval.csv"
    pd.DataFrame(
        [
            {"playlist_id": "p1", "spotify_id": "a"},
            {"playlist_id": "p1", "spotify_id": "b"},
        ]
    ).to_csv(csv_path, index=False)

    catalog = evaluate_recommender.load_or_build_catalog(_args(catalog_csv=str(csv_path)))

    assert catalog["spotify_id"].tolist() == ["a", "b"]


def test_load_or_build_catalog_uses_default_output_csv_when_no_args(monkeypatch, tmp_path):
    output_csv = tmp_path / "real_playlist_eval.csv"
    pd.DataFrame([{"playlist_id": "p1", "spotify_id": "a"}]).to_csv(output_csv, index=False)
    monkeypatch.setattr(evaluate_recommender, "DEFAULT_PLAYLIST_FILE", tmp_path / "missing.txt")

    catalog = evaluate_recommender.load_or_build_catalog(_args(output_csv=str(output_csv)))

    assert catalog["spotify_id"].tolist() == ["a"]


def test_load_or_build_catalog_uses_default_playlist_file_when_no_cache(monkeypatch, tmp_path):
    default_playlist_file = tmp_path / "playlists.txt"
    default_playlist_file.write_text("spotify:playlist:p1\n", encoding="utf-8")
    output_csv = tmp_path / "built.csv"
    built_dataset = pd.DataFrame([{"playlist_id": "p1", "spotify_id": "a"}])

    monkeypatch.setattr(evaluate_recommender, "DEFAULT_PLAYLIST_FILE", default_playlist_file)
    monkeypatch.setattr(
        evaluate_recommender, "get_merged_dataset", lambda paths, force_rebuild: pd.DataFrame()
    )
    monkeypatch.setattr(evaluate_recommender, "count_raw_catalog_rows", lambda paths: 0)
    monkeypatch.setattr(evaluate_recommender, "get_spotify_client", lambda: object())
    monkeypatch.setattr(
        evaluate_recommender,
        "build_evaluation_dataset",
        lambda sp, catalog_df, playlist_inputs, min_matched_tracks, raw_catalog_rows: (
            built_dataset,
            pd.DataFrame(),
        ),
    )

    catalog = evaluate_recommender.load_or_build_catalog(_args(output_csv=str(output_csv)))

    assert catalog["spotify_id"].tolist() == ["a"]
    assert pd.read_csv(output_csv)["spotify_id"].tolist() == ["a"]


def test_load_or_build_catalog_builds_from_playlist_inputs(monkeypatch, tmp_path):
    output_csv = tmp_path / "built.csv"
    summary_csv = tmp_path / "summary.csv"
    raw_catalog = pd.DataFrame([{"spotify_id": "raw"}])
    built_dataset = pd.DataFrame(
        [
            {"playlist_id": "p1", "spotify_id": "a"},
            {"playlist_id": "p1", "spotify_id": "b"},
        ]
    )
    summary = pd.DataFrame([{"playlist_id": "p1", "matched_tracks": 2, "included": True}])

    monkeypatch.setattr(
        evaluate_recommender, "get_merged_dataset", lambda paths, force_rebuild: raw_catalog
    )
    monkeypatch.setattr(evaluate_recommender, "count_raw_catalog_rows", lambda paths: 1)
    monkeypatch.setattr(evaluate_recommender, "get_spotify_client", lambda: object())
    monkeypatch.setattr(
        evaluate_recommender,
        "build_evaluation_dataset",
        lambda sp, catalog_df, playlist_inputs, min_matched_tracks, raw_catalog_rows: (
            built_dataset,
            summary,
        ),
    )

    catalog = evaluate_recommender.load_or_build_catalog(
        _args(
            playlist_url=["spotify:playlist:p1"],
            output_csv=str(output_csv),
            summary_csv=str(summary_csv),
            min_matched_tracks=2,
        )
    )

    assert catalog["spotify_id"].tolist() == ["a", "b"]
    assert pd.read_csv(output_csv)["spotify_id"].tolist() == ["a", "b"]
    assert pd.read_csv(summary_csv)["matched_tracks"].tolist() == [2]


def test_load_or_build_catalog_renames_built_playlist_column(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate_recommender, "get_merged_dataset", lambda paths, force_rebuild: pd.DataFrame()
    )
    monkeypatch.setattr(evaluate_recommender, "count_raw_catalog_rows", lambda paths: 0)
    monkeypatch.setattr(evaluate_recommender, "get_spotify_client", lambda: object())
    monkeypatch.setattr(
        evaluate_recommender,
        "build_evaluation_dataset",
        lambda sp, catalog_df, playlist_inputs, min_matched_tracks, raw_catalog_rows: (
            pd.DataFrame([{"playlist_id": "p1", "spotify_id": "a"}]),
            pd.DataFrame(),
        ),
    )

    catalog = evaluate_recommender.load_or_build_catalog(
        _args(
            playlist_url=["spotify:playlist:p1"],
            output_csv=str(tmp_path / "built.csv"),
            playlist_col="source_playlist",
        )
    )

    assert "source_playlist" in catalog.columns
    assert "playlist_id" not in catalog.columns
