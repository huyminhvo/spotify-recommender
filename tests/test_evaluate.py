import pandas as pd

from recommender.evaluate import (
    DEFAULT_STRATEGIES,
    EvaluationConfig,
    audit_memberships,
    bootstrap_confidence_intervals,
    evaluate_benchmark,
    ranking_metrics,
    summarize_evaluations,
)


def _track(spotify_id, danceability, energy, popularity=80):
    return {
        "spotify_id": spotify_id,
        "title_raw": f"Track {spotify_id}",
        "artists_raw": [f"Artist {spotify_id}"],
        "artist_primary_canon": f"artist {spotify_id}",
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


def _strategy_subset(*names):
    return tuple(strategy for strategy in DEFAULT_STRATEGIES if strategy.name in names)


def _labels(playlist_id, *spotify_ids):
    return pd.DataFrame(
        [
            {
                "playlist_id": playlist_id,
                "position": position,
                "source_spotify_id": spotify_id,
                "catalog_spotify_id": spotify_id,
                "matched": True,
            }
            for position, spotify_id in enumerate(spotify_ids)
        ]
    )


def test_ranking_metrics_compute_bounded_metrics_without_duplicate_hits():
    metrics = ranking_metrics(["a", "a", "b"], {"a", "b"}, k=3)

    assert metrics["precision_at_k"] == 2 / 3
    assert metrics["recall_at_k"] == 1.0
    assert metrics["hit_rate_at_k"] == 1.0
    assert 0.0 < metrics["ndcg_at_k"] <= 1.0


def test_evaluator_keeps_catalog_and_memberships_separate_and_uses_all_positives():
    catalog = pd.DataFrame(
        [
            _track("p1-a", 0.9, 0.8, popularity=20),
            _track("p1-b", 0.88, 0.78, popularity=30),
            _track("p1-c", 0.86, 0.76, popularity=40),
            _track("distractor", 0.1, 0.2, popularity=99),
        ]
    )
    memberships = _labels("p1", "p1-a", "p1-b", "p1-c")

    result = evaluate_benchmark(
        catalog,
        memberships,
        config=EvaluationConfig(
            top_k=2,
            seed_size=1,
            num_splits=1,
            holdout_size=1,
            random_state=0,
            min_popularity=None,
            bootstrap_samples=50,
        ),
        strategies=_strategy_subset("popularity", "unweighted_cosine"),
    )

    assert set(result.per_split["strategy"]) == {
        "popularity",
        "unweighted_cosine",
    }
    assert set(result.per_split["num_positive_tracks"]) == {2}
    assert set(result.per_split["num_matched_positive_tracks"]) == {2}
    assert set(result.summary["num_playlists"]) == {1}
    assert "playlist_id" not in catalog.columns


def test_unmatched_membership_reduces_end_to_end_recall_ceiling():
    catalog = pd.DataFrame(
        [
            _track("matched-a", 0.9, 0.8),
            _track("matched-b", 0.88, 0.78),
            _track("distractor", 0.1, 0.2),
        ]
    )
    memberships = pd.DataFrame(
        [
            {
                "playlist_id": "p1",
                "position": 0,
                "source_spotify_id": "source-a",
                "catalog_spotify_id": "matched-a",
            },
            {
                "playlist_id": "p1",
                "position": 1,
                "source_spotify_id": "source-b",
                "catalog_spotify_id": "matched-b",
            },
            {
                "playlist_id": "p1",
                "position": 2,
                "source_spotify_id": "missing",
                "catalog_spotify_id": None,
            },
        ]
    )

    result = evaluate_benchmark(
        catalog,
        memberships,
        config=EvaluationConfig(
            top_k=2,
            seed_size=1,
            num_splits=1,
            min_popularity=None,
            bootstrap_samples=50,
        ),
        strategies=_strategy_subset("popularity"),
    )

    row = result.per_split.iloc[0]
    assert row["num_positive_tracks"] == 2
    assert row["num_matched_positive_tracks"] == 1
    assert row["matched_recall_ceiling"] == 0.5
    assert row["recall_at_k"] <= row["matched_recall_ceiling"]


def test_repeated_splits_and_randomized_policy_are_reproducible():
    catalog = pd.DataFrame(
        [_track(f"track-{index}", 0.1 + index / 20, 0.2 + index / 20) for index in range(8)]
    )
    memberships = _labels("p1", *(f"track-{index}" for index in range(6)))
    config = EvaluationConfig(
        top_k=2,
        seed_size=2,
        num_splits=3,
        min_popularity=None,
        random_state=19,
        bootstrap_samples=50,
    )

    first = evaluate_benchmark(
        catalog,
        memberships,
        config=config,
        strategies=_strategy_subset("deployed"),
    )
    second = evaluate_benchmark(
        catalog,
        memberships,
        config=config,
        strategies=_strategy_subset("deployed"),
    )

    pd.testing.assert_frame_equal(first.per_split, second.per_split)
    pd.testing.assert_frame_equal(first.recommendations, second.recommendations)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    assert first.per_split["split_seed"].nunique() == 3


def test_bootstrap_resamples_playlist_means_deterministically():
    per_playlist = pd.DataFrame(
        {
            "playlist_id": ["p1", "p2", "p3"],
            "strategy": ["s", "s", "s"],
            "recall_at_k": [0.0, 0.5, 1.0],
        }
    )

    first = bootstrap_confidence_intervals(
        per_playlist,
        metric_cols=["recall_at_k"],
        bootstrap_samples=200,
        random_state=7,
    )
    second = bootstrap_confidence_intervals(
        per_playlist,
        metric_cols=["recall_at_k"],
        bootstrap_samples=200,
        random_state=7,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[0, "recall_at_k"] == 0.5
    assert 0.0 <= first.loc[0, "recall_at_k_ci_low"] <= 0.5
    assert 0.5 <= first.loc[0, "recall_at_k_ci_high"] <= 1.0


def test_catalog_coverage_uses_distinct_items_across_all_requests():
    per_split = pd.DataFrame(
        [
            {
                "playlist_id": playlist_id,
                "split_id": split_id,
                "strategy": "s",
                "candidate_pool_size": 10,
                **{
                    metric: 0.0
                    for metric in [
                        "precision_at_k",
                        "recall_at_k",
                        "matched_recall_at_k",
                        "retrievable_recall_at_k",
                        "hit_rate_at_k",
                        "ndcg_at_k",
                        "candidate_recall_ceiling",
                        "matched_recall_ceiling",
                        "fill_rate",
                        "avg_recommendation_popularity",
                        "avg_similarity",
                        "artist_diversity",
                        "artist_duplication_rate",
                        "steering_target_distance",
                    ]
                },
            }
            for playlist_id, split_id in [("p1", 0), ("p2", 0)]
        ]
    )
    recommendations = pd.DataFrame(
        {
            "strategy": ["s", "s", "s", "s"],
            "spotify_id": ["a", "b", "a", "b"],
        }
    )

    summary = summarize_evaluations(
        per_split,
        recommendations,
        catalog_size=10,
        config=EvaluationConfig(bootstrap_samples=50),
    )

    assert summary.loc[0, "unique_recommendations"] == 2
    assert summary.loc[0, "catalog_coverage"] == 0.2
    assert summary.loc[0, "recommendation_repeat_rate"] == 0.5


def test_membership_audit_flags_small_and_near_duplicate_benchmarks():
    memberships = pd.concat(
        [
            _labels("p1", "a", "b", "c"),
            _labels("p2", "a", "b", "c"),
        ],
        ignore_index=True,
    )

    audit = audit_memberships(
        memberships,
        min_playlists=50,
        near_duplicate_jaccard=0.8,
    )

    assert audit["benchmark_ready"] is False
    assert audit["near_duplicate_pairs"] == 1
    assert len(audit["warnings"]) == 2
