import json
from types import SimpleNamespace

import pandas as pd
import pytest

from recommender.schema import FEATURE_COLS
from recommender.tuning import (
    TuningConfig,
    WeightCandidate,
    generate_weight_candidates,
    partition_playlist_ids,
    tune_recommender_weights,
    write_tuning_result,
)
from recommender.weightings import DEFAULT_WEIGHTS
from scripts import tune_recommender_weights as tuning_script


def _memberships(num_playlists=6, tracks_per_playlist=3):
    return pd.DataFrame(
        [
            {
                "playlist_id": f"playlist-{playlist_index}",
                "position": position,
                "source_spotify_id": f"source-{playlist_index}-{position}",
                "catalog_spotify_id": f"catalog-{playlist_index}-{position}",
            }
            for playlist_index in range(num_playlists)
            for position in range(tracks_per_playlist)
        ]
    )


def _config(**overrides):
    values = {
        "num_trials": 3,
        "test_fraction": 0.25,
        "num_splits": 4,
        "top_k": 5,
        "seed_size": 1,
        "random_state": 17,
        "min_playlists": 4,
        "min_tuning_playlists": 2,
        "min_test_playlists": 1,
        "bootstrap_samples": 10,
    }
    values.update(overrides)
    return TuningConfig(**values)


def test_partition_is_deterministic_and_keeps_whole_playlists_together():
    memberships = _memberships()
    config = _config()

    first = partition_playlist_ids(memberships, config)
    second = partition_playlist_ids(memberships, config)

    assert first == second
    assert set(first.tuning_playlist_ids).isdisjoint(first.test_playlist_ids)
    assert set(first.tuning_playlist_ids) | set(first.test_playlist_ids) == {
        f"playlist-{index}" for index in range(6)
    }


def test_partition_refuses_too_few_playlists():
    with pytest.raises(ValueError, match="requires at least 4 playlists"):
        partition_playlist_ids(_memberships(num_playlists=3), _config())


@pytest.mark.parametrize(
    ("weight_min", "weight_max"),
    [
        (float("nan"), 4.0),
        (0.25, float("nan")),
        (0.25, float("inf")),
    ],
)
def test_tuning_config_rejects_nonfinite_weight_bounds(weight_min, weight_max):
    with pytest.raises(ValueError, match="weight_min"):
        _config(weight_min=weight_min, weight_max=weight_max)


def test_weight_candidate_rejects_unknown_features():
    weights = dict.fromkeys(FEATURE_COLS, 1.0)
    weights["energgy"] = 1.0

    with pytest.raises(ValueError, match="unsupported features"):
        WeightCandidate("typo", weights, "test")


def test_candidate_generation_includes_baselines_and_reproducible_log_uniform_trials():
    config = _config(num_trials=5, weight_min=0.5, weight_max=2.0)

    first = generate_weight_candidates(config)
    second = generate_weight_candidates(config)

    assert [candidate.name for candidate in first] == [
        "uniform",
        "hand_set_defaults",
        "random_001",
        "random_002",
        "random_003",
    ]
    assert dict(first[0].weights) == dict.fromkeys(FEATURE_COLS, 1.0)
    assert dict(first[1].weights) == {
        feature: DEFAULT_WEIGHTS.get(feature, 1.0) for feature in FEATURE_COLS
    }
    assert [dict(candidate.weights) for candidate in first] == [
        dict(candidate.weights) for candidate in second
    ]
    for candidate in first[2:]:
        assert all(0.5 <= value <= 2.0 for value in candidate.weights.values())


def test_tuning_uses_only_tuning_playlists_and_selects_ndcg_then_recall(tmp_path):
    memberships = _memberships()
    config = _config()
    observed = {}

    def fake_evaluator(catalog, labels, config, strategies):
        observed["catalog"] = catalog
        observed["playlist_ids"] = set(labels["playlist_id"])
        observed["config"] = config
        observed["strategies"] = strategies
        metrics = {
            "uniform": (0.20, 0.80),
            "hand_set_defaults": (0.55, 0.40),
            "random_001": (0.55, 0.60),
        }
        return SimpleNamespace(
            summary=pd.DataFrame(
                [
                    {
                        "strategy": strategy.name,
                        "ndcg_at_k": metrics[strategy.name][0],
                        "recall_at_k": metrics[strategy.name][1],
                    }
                    for strategy in strategies
                ]
            )
        )

    result = tune_recommender_weights(
        catalog="catalog-store",
        memberships=memberships,
        config=config,
        evaluator=fake_evaluator,
    )

    assert observed["catalog"] == "catalog-store"
    assert observed["playlist_ids"] == set(result.partition.tuning_playlist_ids)
    assert observed["playlist_ids"].isdisjoint(result.partition.test_playlist_ids)
    assert observed["config"].num_splits == 4
    assert all(strategy.policy.strategy == "weighted_cosine" for strategy in observed["strategies"])
    assert all(strategy.policy.use_pca is True for strategy in observed["strategies"])
    assert all(strategy.policy.randomize_results is False for strategy in observed["strategies"])
    assert result.selected_trial_name == "random_001"

    output_path = write_tuning_result(result, tmp_path / "weights.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selected_weights"] == result.selected_weights
    assert payload["partition"]["test_playlist_ids"] == list(result.partition.test_playlist_ids)
    assert payload["test_playlists_evaluated"] is False
    assert payload["final_benchmark_required"] is True
    assert "squared" in payload["weight_semantics"]
    assert len(payload["trials"]) == 3


def test_tuning_json_is_byte_reproducible(tmp_path):
    memberships = _memberships()
    config = _config(num_trials=2)

    def fake_evaluator(catalog, labels, config, strategies):
        return SimpleNamespace(
            summary=pd.DataFrame(
                [
                    {
                        "strategy": strategy.name,
                        "ndcg_at_k": 0.5,
                        "recall_at_k": 0.5,
                    }
                    for strategy in strategies
                ]
            )
        )

    first = tune_recommender_weights(object(), memberships, config, evaluator=fake_evaluator)
    second = tune_recommender_weights(object(), memberships, config, evaluator=fake_evaluator)
    first_path = write_tuning_result(first, tmp_path / "first.json")
    second_path = write_tuning_result(second, tmp_path / "second.json")

    assert first.selected_trial_name == "uniform"
    assert first_path.read_bytes() == second_path.read_bytes()


def test_cli_help_explains_squared_weight_influence_and_held_out_test(capsys):
    with pytest.raises(SystemExit) as exc_info:
        tuning_script.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "squared" in help_text
    assert "final benchmark" in help_text
    assert "never evaluated" in help_text
