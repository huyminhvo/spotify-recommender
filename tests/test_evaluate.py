import pandas as pd

from recommender.evaluate import (
    EvaluationConfig,
    evaluate_catalog_playlists,
    ranking_metrics,
)


def _track(playlist_id, spotify_id, danceability, energy, popularity=80):
    return {
        "playlist_id": playlist_id,
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
        "release_year": 2020,
    }


def test_ranking_metrics_compute_precision_recall_hit_rate_and_ndcg():
    metrics = ranking_metrics(["a", "b", "c"], {"b", "d"}, k=3)

    assert metrics["precision_at_k"] == 1 / 3
    assert metrics["recall_at_k"] == 1 / 2
    assert metrics["hit_rate_at_k"] == 1.0
    assert 0.0 < metrics["ndcg_at_k"] < 1.0


def test_evaluate_catalog_playlists_compares_baselines():
    catalog = pd.DataFrame(
        [
            _track("p1", "p1_seed", 0.9, 0.8, popularity=20),
            _track("p1", "p1_holdout", 0.88, 0.78, popularity=30),
            _track("p1", "p1_other", 0.1, 0.2, popularity=99),
            _track("p2", "p2_seed", 0.2, 0.3, popularity=20),
            _track("p2", "p2_holdout", 0.21, 0.31, popularity=30),
            _track("p2", "p2_other", 0.95, 0.95, popularity=99),
        ]
    )

    results = evaluate_catalog_playlists(
        catalog,
        playlist_col="playlist_id",
        config=EvaluationConfig(
            top_k=2,
            seed_size=1,
            holdout_size=1,
            random_state=0,
            min_popularity=None,
            use_pca=False,
        ),
        strategies=("popularity", "unweighted_cosine"),
    )

    assert set(results["strategy"]) == {"popularity", "unweighted_cosine"}
    assert set(results["num_playlists"]) == {2}
    assert {
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "ndcg_at_k",
        "coverage_rate",
        "avg_similarity",
        "artist_diversity",
        "artist_duplication_rate",
    }.issubset(results.columns)
