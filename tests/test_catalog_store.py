import pandas as pd
import pytest

from utils.catalog_store import CatalogQueryError, CatalogStore


def _catalog():
    return pd.DataFrame(
        [
            {
                "spotify_id": "seed",
                "title_raw": "Seed Song",
                "title_canon": "seed song",
                "artists_raw": ["Seed Artist"],
                "artist_primary_canon": "seed artist",
                "duration_ms": 200_000,
                "popularity": 80,
                "release_year": 2024,
            },
            {
                "spotify_id": "candidate",
                "title_raw": "Candidate",
                "title_canon": "candidate",
                "artists_raw": ["Other Artist"],
                "artist_primary_canon": "other artist",
                "duration_ms": 180_000,
                "popularity": 50,
                "release_year": 2020,
            },
        ]
    )


def test_catalog_store_matches_without_an_in_memory_index(tmp_path):
    path = tmp_path / "catalog.parquet"
    _catalog().to_parquet(path, index=False)
    store = CatalogStore(path)

    by_id = store.match_track({"id": "seed"})
    by_metadata = store.match_track(
        {
            "name": "Seed Song",
            "artists": [{"name": "Seed Artist"}],
            "duration_ms": 200_000,
        }
    )

    assert by_id["spotify_id"] == "seed"
    assert by_metadata["spotify_id"] == "seed"


def test_catalog_store_filters_and_bounds_candidates(tmp_path):
    path = tmp_path / "catalog.parquet"
    _catalog().to_parquet(path, index=False)
    store = CatalogStore(path, candidate_limit=1)

    candidates = store.load_candidates(exclude_ids=["seed"], min_popularity=20)

    assert candidates["spotify_id"].tolist() == ["candidate"]


def test_catalog_store_loads_tracks_in_requested_order(tmp_path):
    path = tmp_path / "catalog.parquet"
    _catalog().to_parquet(path, index=False)
    store = CatalogStore(path)

    tracks = store.load_tracks(["candidate", "missing", "seed", "candidate"])

    assert tracks["spotify_id"].tolist() == ["candidate", "seed"]


def test_catalog_store_counts_all_eligible_candidates_before_limit(tmp_path):
    path = tmp_path / "catalog.parquet"
    _catalog().to_parquet(path, index=False)
    store = CatalogStore(path, candidate_limit=1)

    assert store.count_candidates(min_popularity=20) == 2
    assert store.count_candidates(exclude_ids=["seed"], min_popularity=20) == 1
    assert store.count_candidates(exclude_artists=["other artist"]) == 1
    assert store.count_candidates(year_range=(2021, 2025)) == 1


def test_catalog_store_retries_transient_zstd_failure(monkeypatch, tmp_path):
    path = tmp_path / "catalog.parquet"
    path.touch()
    store = CatalogStore(path)
    attempts = 0

    class FakeResult:
        def fetch_df(self):
            return pd.DataFrame({"spotify_id": ["recovered"]})

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, sql, parameters):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("ZSTD Decompression failure")
            return FakeResult()

    monkeypatch.setattr(store, "_connect", lambda: FakeConnection())

    result = store._query("SELECT 1")

    assert attempts == 2
    assert result["spotify_id"].tolist() == ["recovered"]


def test_catalog_store_wraps_nontransient_local_failures(monkeypatch, tmp_path):
    path = tmp_path / "catalog.parquet"
    path.touch()
    store = CatalogStore(path)

    class FailingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def execute(self, sql, parameters):
            raise RuntimeError("invalid parquet footer")

    monkeypatch.setattr(store, "_connect", lambda: FailingConnection())

    with pytest.raises(CatalogQueryError, match="invalid parquet footer"):
        store._query("SELECT 1")
