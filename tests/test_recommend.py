import pandas as pd

from recommender.recommend import recommend, recommend_from_catalog


def _track(spotify_id, danceability, energy, popularity=80, release_year=2020):
    return {
        "spotify_id": spotify_id,
        "title_raw": f"Track {spotify_id}",
        "artists_raw": ["Artist"],
        "artist_primary_canon": "artist",
        "danceability": danceability,
        "energy": energy,
        "valence": 0.5,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "speechiness": 0.05,
        "tempo": 120.0,
        "loudness": -8.0,
        "duration_ms": 210_000,
        "popularity": popularity,
        "release_year": release_year,
    }


def test_recommend_returns_ranked_dataframe_with_expected_shape(monkeypatch):
    catalog = pd.DataFrame(
        [
            _track("seed", 0.9, 0.8),
            _track("close", 0.88, 0.78),
            _track("far", 0.1, 0.2),
            _track("filtered", 0.9, 0.8, popularity=5),
        ]
    )
    user_tracks = catalog[catalog["spotify_id"] == "seed"].copy()

    monkeypatch.setattr(
        "recommender.recommend.get_merged_dataset",
        lambda catalog_paths: catalog,
    )

    recs = recommend(
        catalog_paths=["unused.csv"],
        user_tracks_df=user_tracks,
        top_n=2,
        min_popularity=20,
        use_pca=False,
    )

    assert recs.shape[0] == 2
    assert "similarity" in recs.columns
    assert recs["spotify_id"].tolist() == ["close", "far"]
    assert "seed" not in recs["spotify_id"].tolist()
    assert recs["similarity"].is_monotonic_decreasing
    assert recs["recommendation_reason"].str.startswith("Recommended because").all()


def test_recommend_default_pca_handles_small_catalog(monkeypatch):
    catalog = pd.DataFrame(
        [
            _track("seed", 0.9, 0.8),
            _track("close", 0.88, 0.78),
            _track("far", 0.1, 0.2),
        ]
    )
    user_tracks = catalog[catalog["spotify_id"] == "seed"].copy()

    monkeypatch.setattr(
        "recommender.recommend.get_merged_dataset",
        lambda catalog_paths: catalog,
    )

    recs = recommend(
        catalog_paths=["unused.csv"],
        user_tracks_df=user_tracks,
        top_n=2,
        min_popularity=20,
    )

    assert recs.shape[0] == 2
    assert recs["spotify_id"].tolist() == ["close", "far"]


def test_recommend_supports_popularity_and_random_baselines(monkeypatch):
    catalog = pd.DataFrame(
        [
            _track("seed", 0.9, 0.8, popularity=10),
            _track("popular", 0.1, 0.2, popularity=99),
            _track("middle", 0.2, 0.3, popularity=50),
            _track("low", 0.9, 0.8, popularity=20),
        ]
    )
    user_tracks = catalog[catalog["spotify_id"] == "seed"].copy()

    monkeypatch.setattr(
        "recommender.recommend.get_merged_dataset",
        lambda catalog_paths: catalog,
    )

    popular_recs = recommend(
        catalog_paths=["unused.csv"],
        user_tracks_df=user_tracks,
        top_n=2,
        min_popularity=None,
        strategy="popularity",
    )
    random_recs = recommend(
        catalog_paths=["unused.csv"],
        user_tracks_df=user_tracks,
        top_n=2,
        min_popularity=None,
        strategy="random",
        random_state=7,
    )

    assert popular_recs["spotify_id"].tolist() == ["popular", "middle"]
    assert len(random_recs) == 2
    assert "seed" not in random_recs["spotify_id"].tolist()


def test_recommend_supports_unweighted_cosine_and_same_artist_exclusion(monkeypatch):
    catalog = pd.DataFrame(
        [
            _track("seed", 0.9, 0.8),
            _track("same_artist_close", 0.89, 0.79),
            _track("other_artist", 0.6, 0.5),
        ]
    )
    catalog.loc[catalog["spotify_id"] == "same_artist_close", "artist_primary_canon"] = "artist"
    catalog.loc[catalog["spotify_id"] == "other_artist", "artist_primary_canon"] = "other artist"
    user_tracks = catalog[catalog["spotify_id"] == "seed"].copy()

    monkeypatch.setattr(
        "recommender.recommend.get_merged_dataset",
        lambda catalog_paths: catalog,
    )

    recs = recommend(
        catalog_paths=["unused.csv"],
        user_tracks_df=user_tracks,
        top_n=2,
        min_popularity=None,
        strategy="unweighted_cosine",
        use_pca=False,
        same_artist_exclusion=True,
    )

    assert recs["spotify_id"].tolist() == ["other_artist"]


def test_neutral_adjustments_preserve_the_base_ranking():
    catalog = pd.DataFrame(
        [
            _track("seed", 0.5, 0.5),
            _track("a", 0.52, 0.52),
            _track("b", 0.7, 0.7),
        ]
    )
    user_tracks = catalog[catalog["spotify_id"] == "seed"].copy()

    base = recommend_from_catalog(catalog, user_tracks, min_popularity=None, use_pca=False)
    neutral = recommend_from_catalog(
        catalog,
        user_tracks,
        min_popularity=None,
        use_pca=False,
        adjustments={"energy": 0.0, "valence": 0.0},
    )

    assert neutral["spotify_id"].tolist() == base["spotify_id"].tolist()
    assert neutral["score"].tolist() == base["score"].tolist()


def test_energy_adjustment_penalizes_candidates_by_distance_to_target():
    catalog = pd.DataFrame(
        [
            _track("seed", 0.5, 0.5),
            _track("lower", 0.5, 0.4),
            _track("higher", 0.5, 0.7),
        ]
    )
    user_tracks = catalog[catalog["spotify_id"] == "seed"].copy()

    recs = recommend_from_catalog(
        catalog,
        user_tracks,
        min_popularity=None,
        use_pca=False,
        adjustments={"energy": 0.3},
    )

    penalties = (recs["similarity"] - recs["score"]).set_axis(recs["spotify_id"])
    assert penalties["higher"] < penalties["lower"]
    assert (recs["score"] <= recs["similarity"]).all()
