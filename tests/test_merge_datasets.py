from pathlib import Path

import pandas as pd
import pytest

from utils import merge_datasets
from utils.merge_datasets import (
    _auto_columns,
    _fingerprint_inputs,
    _merge_two_rows,
    _normalize_row,
)


def _normalize(**values):
    frame = pd.DataFrame([values])
    return _normalize_row(frame.iloc[0], _auto_columns(frame))


def test_auto_columns_recognizes_artist_names_and_preserves_original_case():
    frame = pd.DataFrame(columns=["Artist_Names", "Danceability", "ISRC"])

    columns = _auto_columns(frame)

    assert columns["artists"] == "Artist_Names"
    assert columns["danceability"] == "Danceability"
    assert columns["isrc"] == "ISRC"


def test_singular_artist_name_preserves_internal_commas():
    row = _normalize(artist_name="Tyler, The Creator", track_name="EARFQUAKE")

    assert row["artists_raw"] == ["Tyler, The Creator"]
    assert row["artist_primary_canon"] == "tyler, the creator"


def test_artists_literal_is_parsed_as_an_exact_list():
    row = _normalize(artists="['Tyler, The Creator', 'Kali Uchis']", name="See You Again")

    assert row["artists_raw"] == ["Tyler, The Creator", "Kali Uchis"]
    assert row["artist_primary_canon"] == "tyler, the creator"


def test_delimited_artist_names_preserves_collaborators():
    row = _normalize(artist_names="ZAYN, PARTYNEXTDOOR", track_name="Song")

    assert row["artists_raw"] == ["ZAYN", "PARTYNEXTDOOR"]
    assert row["artist_primary_canon"] == "zayn"


def test_merge_uses_authoritative_sources_to_preserve_comma_bearing_artist_names(tmp_path):
    chart_source = tmp_path / "chart.csv"
    pd.DataFrame(
        [
            {
                "id": "chart-track",
                "track_name": "Chart Song",
                "artist_names": "Tyler, The Creator, Kali Uchis",
                "duration_ms": 200_000,
            }
        ]
    ).to_csv(chart_source, index=False)
    authoritative_source = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "track_id": "catalog-track",
                "track_name": "Catalog Song",
                "artist_name": "Tyler, The Creator",
                "duration_ms": 210_000,
            }
        ]
    ).to_csv(authoritative_source, index=False)

    merged = merge_datasets.merge_datasets([str(chart_source), str(authoritative_source)])
    chart_row = merged.loc[merged["spotify_id"] == "chart-track"].iloc[0]

    assert chart_row["artists_raw"] == ["Tyler, The Creator", "Kali Uchis"]
    assert chart_row["artist_primary_canon"] == "tyler, the creator"


def test_merge_recomputes_title_and_artist_canonical_fields():
    sparse = {
        "spotify_id": "track-id",
        "title_raw": "Song",
        "title_canon": "stale title",
        "artists_raw": [],
        "artist_primary_canon": "stale artist",
    }
    authoritative = {
        "spotify_id": "track-id",
        "title_raw": "Longer Song Title",
        "title_canon": "another stale title",
        "artists_raw": ["Tyler, The Creator"],
        "artist_primary_canon": "another stale artist",
    }

    merged = _merge_two_rows(sparse, authoritative)

    assert merged["title_canon"] == "longer song title"
    assert merged["artist_primary_canon"] == "tyler, the creator"


def test_missing_ids_and_titles_do_not_become_nan_strings():
    row = _normalize(id=float("nan"), name=float("nan"), artist_name="Artist")

    assert row["spotify_id"] is None
    assert row["title_raw"] == ""
    assert row["title_canon"] == ""


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("False", False), ("true", True), (0, False), (1, True), ("unknown", None)],
)
def test_explicit_values_are_parsed_without_string_truthiness(raw_value, expected):
    row = _normalize(id="track-id", name="Song", artist_name="Artist", explicit=raw_value)

    assert row["explicit"] is expected


def test_nonpositive_duration_bucket_is_rejected():
    with pytest.raises(ValueError, match="greater than zero"):
        merge_datasets.merge_datasets([], conservative_duration_ms=0)


def test_cache_fingerprint_includes_merge_schema_version(monkeypatch, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id\ntrack-id\n", encoding="utf-8")
    initial = _fingerprint_inputs([str(source)])

    monkeypatch.setattr(
        merge_datasets,
        "MERGE_SCHEMA_VERSION",
        merge_datasets.MERGE_SCHEMA_VERSION + 1,
    )

    assert _fingerprint_inputs([str(source)]) != initial


def test_cache_write_removes_partial_temporary_file(monkeypatch, tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id\ntrack-id\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    frame = pd.DataFrame({"spotify_id": ["track-id"]})
    monkeypatch.setattr(merge_datasets, "merge_datasets", lambda paths: frame)

    def fail_write(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(RuntimeError, match="simulated"):
        merge_datasets.get_merged_dataset([str(source)], cache_dir=str(cache_dir))

    assert list(cache_dir.glob("*.tmp.parquet")) == []
    assert list(cache_dir.glob("merged_*.parquet")) == []
