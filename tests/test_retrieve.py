import pandas as pd

from recommender.retrieve import filter_candidates


def test_filter_candidates_excludes_seed_tracks_and_applies_filters():
    catalog = pd.DataFrame(
        [
            {"spotify_id": "seed", "popularity": 99, "release_year": 2020},
            {"spotify_id": "too_obscure", "popularity": 10, "release_year": 2020},
            {"spotify_id": "too_old", "popularity": 80, "release_year": 1975},
            {"spotify_id": "keep", "popularity": 75, "release_year": 2018},
        ]
    )

    filtered = filter_candidates(
        catalog,
        exclude_ids=["seed"],
        min_popularity=20,
        max_popularity=90,
        year_range=(2000, 2022),
    )

    assert filtered["spotify_id"].tolist() == ["keep"]
    assert filtered.index.tolist() == [0]


def test_filter_candidates_treats_missing_popularity_conservatively():
    catalog = pd.DataFrame(
        [
            {"spotify_id": "missing", "popularity": None, "release_year": 2020},
            {"spotify_id": "known", "popularity": 50, "release_year": 2020},
        ]
    )

    filtered = filter_candidates(catalog, min_popularity=20)

    assert filtered["spotify_id"].tolist() == ["known"]
