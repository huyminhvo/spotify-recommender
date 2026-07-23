import json

import numpy as np
import pandas as pd
import pytest

from recommender.evaluate import EvaluationConfig, EvaluationResult
from scripts import build_evaluation_dataset, evaluate_recommender


def _result(benchmark_ready=False):
    summary = pd.DataFrame(
        [
            {
                "strategy": "deployed",
                "num_playlists": 1,
                "recall_at_k": 0.1,
                "recall_at_k_ci_low": 0.1,
                "recall_at_k_ci_high": 0.1,
                "ndcg_at_k": 0.08,
                "ndcg_at_k_ci_low": 0.08,
                "ndcg_at_k_ci_high": 0.08,
                "hit_rate_at_k": 0.5,
                "candidate_recall_ceiling": 0.2,
                "retrievable_recall_at_k": 0.5,
                "catalog_coverage": 0.001,
            }
        ]
    )
    return EvaluationResult(
        per_split=pd.DataFrame(),
        recommendations=pd.DataFrame(),
        summary=summary,
        skipped=pd.DataFrame(),
        audit={
            "benchmark_ready": benchmark_ready,
            "num_playlists": 1,
            "num_memberships": 42,
            "warnings": ["Only one playlist is labeled."],
        },
    )


def _metadata():
    return {
        "catalog": {
            "artifact": "catalog.parquet",
            "rows": 1000,
            "eligible_rows": 500,
            "candidate_limit": 100,
        },
        "labels": {
            "path": "labels.csv",
            "sha256": "abc",
            "schema": "legacy-matched-only",
            "preserves_unmatched": False,
            "rows": 42,
        },
        "match": {
            "matched_tracks": 42,
            "total_source_tracks": 101,
            "match_rate": 42 / 101,
        },
    }


def test_load_memberships_normalizes_legacy_labels(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"playlist_id": "p1", "spotify_id": "a"},
            {"playlist_id": "p1", "spotify_id": "b"},
        ]
    ).to_csv(path, index=False)

    labels, metadata = evaluate_recommender.load_memberships(path)

    assert labels["catalog_spotify_id"].tolist() == ["a", "b"]
    assert labels["source_spotify_id"].tolist() == ["a", "b"]
    assert metadata["schema"] == "legacy-matched-only"
    assert metadata["preserves_unmatched"] is False


def test_default_membership_path_returns_none_when_no_local_labels(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        evaluate_recommender,
        "DEFAULT_MEMBERSHIP_CSV",
        tmp_path / "missing-modern.csv",
    )
    monkeypatch.setattr(
        evaluate_recommender,
        "EXAMPLE_MEMBERSHIP_CSV",
        tmp_path / "missing-example.csv",
    )
    monkeypatch.setattr(
        evaluate_recommender,
        "LEGACY_EVAL_CSV",
        tmp_path / "missing-legacy.csv",
    )

    assert evaluate_recommender.default_membership_path() is None


def test_builder_and_evaluator_share_default_artifact_paths():
    assert evaluate_recommender.DEFAULT_MEMBERSHIP_CSV == (
        build_evaluation_dataset.DEFAULT_OUTPUT_CSV
    )
    assert evaluate_recommender.DEFAULT_MATCH_SUMMARY_CSV == (
        build_evaluation_dataset.DEFAULT_SUMMARY_CSV
    )


def test_filter_memberships_uses_playlist_level_partition():
    labels = pd.DataFrame(
        {
            "playlist_id": ["p1", "p1", "p2"],
            "catalog_spotify_id": ["a", "b", "c"],
        }
    )

    filtered = evaluate_recommender.filter_memberships(labels, {"p2"})

    assert filtered["playlist_id"].tolist() == ["p2"]
    with pytest.raises(ValueError, match="not found"):
        evaluate_recommender.filter_memberships(labels, {"missing"})


def test_load_match_summary_uses_only_selected_playlists(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "playlist_id": "p1",
                "total_unique_source_tracks": 10,
                "matched_unique_tracks": 8,
            },
            {
                "playlist_id": "p2",
                "total_unique_source_tracks": 20,
                "matched_unique_tracks": 5,
            },
        ]
    ).to_csv(path, index=False)

    summary = evaluate_recommender.load_match_summary(path, {"p2"})

    assert summary["num_playlists"] == 1
    assert summary["total_source_tracks"] == 20
    assert summary["matched_tracks"] == 5
    assert summary["match_rate"] == 0.25
    assert summary["sha256"] == evaluate_recommender.sha256_file(path)


def test_load_playlist_id_filter_reads_tuning_artifact(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text(
        json.dumps(
            {
                "partition": {
                    "tuning_playlist_ids": ["p1"],
                    "test_playlist_ids": ["p2", "p3"],
                }
            }
        ),
        encoding="utf-8",
    )

    assert evaluate_recommender.load_playlist_id_filter(path) == {"p2", "p3"}


def test_tuning_artifact_adds_selected_weight_strategies(tmp_path):
    path = tmp_path / "tuning.json"
    path.write_text(
        json.dumps(
            {
                "selected_weights": {
                    "danceability": 1.1,
                    "energy": 1.2,
                }
            }
        ),
        encoding="utf-8",
    )

    weights = evaluate_recommender.load_selected_weights(path)
    strategies = evaluate_recommender.strategies_with_selected_weights(weights)

    assert strategies[-2].name == "tuned_weighted_cosine_pca"
    assert strategies[-1].name == "tuned_deployed"
    assert strategies[-1].policy.user_weights["energy"] == 1.2


def test_render_report_labels_small_legacy_run_as_smoke_test():
    metadata = _metadata()

    report = evaluate_recommender.render_report(
        _result(),
        config=EvaluationConfig(num_splits=2, bootstrap_samples=50),
        catalog_metadata=metadata["catalog"],
        label_metadata=metadata["labels"],
        match_summary=metadata["match"],
        strategies=evaluate_recommender.DEFAULT_STRATEGIES,
    )

    assert "Engineering smoke test" in report
    assert "legacy matched-only label file" in report
    assert "42/101 tracks" in report
    assert "| deployed | 0.100 |" in report
    assert "playlist-clustered bootstrap" in report


def test_update_readme_replaces_only_generated_section(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Project",
                evaluate_recommender.README_RESULTS_START,
                "old table",
                evaluate_recommender.README_RESULTS_END,
                "Keep me.",
            ]
        ),
        encoding="utf-8",
    )

    evaluate_recommender.update_readme_results(
        readme,
        summary=_result().summary,
        top_k=10,
        report_path=tmp_path / "evaluation.md",
        benchmark_ready=False,
        num_playlists=1,
    )

    text = readme.read_text(encoding="utf-8")
    assert "Smoke test over 1 playlist" in text
    assert "| deployed | 0.100 | 0.080 |" in text
    assert "Keep me." in text
    assert text.count(evaluate_recommender.README_RESULTS_START) == 1


def test_results_json_replaces_non_finite_values(tmp_path):
    result = _result()
    result.summary.loc[0, "catalog_coverage"] = np.nan
    metadata = _metadata()
    payload = evaluate_recommender.build_results_payload(
        result,
        config=EvaluationConfig(bootstrap_samples=50),
        catalog_metadata=metadata["catalog"],
        label_metadata=metadata["labels"],
        match_summary=metadata["match"],
        strategies=evaluate_recommender.DEFAULT_STRATEGIES,
    )
    path = evaluate_recommender.write_results_json(
        payload,
        tmp_path / "results.json",
    )

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"][0]["catalog_coverage"] is None
    assert loaded["status"] == "engineering-smoke-test"


def test_results_payload_records_partition_and_weight_artifacts():
    metadata = _metadata()
    artifacts = {
        "playlist_filter": {
            "path": "partition.json",
            "sha256": "partition-hash",
            "selected_playlist_ids": ["p2"],
        },
        "weights": {"path": "weights.json", "sha256": "weights-hash"},
    }

    payload = evaluate_recommender.build_results_payload(
        _result(),
        config=EvaluationConfig(bootstrap_samples=50),
        catalog_metadata=metadata["catalog"],
        label_metadata=metadata["labels"],
        match_summary=metadata["match"],
        strategies=evaluate_recommender.DEFAULT_STRATEGIES,
        input_artifacts=artifacts,
    )

    assert payload["input_artifacts"] == artifacts
