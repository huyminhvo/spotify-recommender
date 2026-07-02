import pandas as pd

from utils.catalog_store import CatalogStore


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
