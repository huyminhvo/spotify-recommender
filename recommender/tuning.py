from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from recommender.evaluate import (
    EvaluationConfig,
    EvaluationStrategy,
    evaluate_benchmark,
    normalize_memberships,
)
from recommender.policy import DEPLOYED_POLICY, RecommendationPolicy
from recommender.schema import FEATURE_COLS
from recommender.weightings import DEFAULT_WEIGHTS

WEIGHT_SEMANTICS = (
    "Each value is a vector multiplier applied to both the query profile and candidate "
    "vectors. Its direct influence inside cosine dot products is therefore squared."
)


@dataclass(frozen=True)
class TuningConfig:
    """Reproducible playlist-level search settings for feature multipliers."""

    num_trials: int = 8
    test_fraction: float = 0.20
    num_splits: int = 3
    top_k: int = 10
    seed_size: int = 5
    random_state: int = 0
    min_playlists: int = 10
    min_tuning_playlists: int = 5
    min_test_playlists: int = 2
    weight_min: float = 0.25
    weight_max: float = 4.0
    bootstrap_samples: int = 200
    min_popularity: int | None = DEPLOYED_POLICY.min_popularity
    pca_components: int = DEPLOYED_POLICY.pca_components

    def __post_init__(self) -> None:
        if self.num_trials < 2:
            raise ValueError(
                "num_trials must be at least 2 to include uniform and hand-set defaults."
            )
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1.")
        if self.num_splits < 1:
            raise ValueError("num_splits must be at least 1.")
        if self.top_k < 1 or self.seed_size < 1:
            raise ValueError("top_k and seed_size must be at least 1.")
        if self.min_tuning_playlists < 1 or self.min_test_playlists < 1:
            raise ValueError("Both partition minimums must be at least 1.")
        required = self.min_tuning_playlists + self.min_test_playlists
        if self.min_playlists < required:
            raise ValueError(
                "min_playlists must be at least min_tuning_playlists + min_test_playlists."
            )
        if (
            not np.isfinite(self.weight_min)
            or not np.isfinite(self.weight_max)
            or self.weight_min <= 0.0
            or self.weight_max <= self.weight_min
        ):
            raise ValueError("Require 0 < weight_min < weight_max.")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be at least 1.")
        if self.pca_components < 1:
            raise ValueError("pca_components must be at least 1.")


@dataclass(frozen=True)
class PlaylistPartition:
    tuning_playlist_ids: tuple[str, ...]
    test_playlist_ids: tuple[str, ...]


@dataclass(frozen=True)
class WeightCandidate:
    name: str
    weights: Mapping[str, float]
    source: str

    def __post_init__(self) -> None:
        missing = [feature for feature in FEATURE_COLS if feature not in self.weights]
        if missing:
            raise ValueError(f"Weight candidate is missing features: {missing}")
        unknown = set(self.weights) - set(FEATURE_COLS)
        if unknown:
            raise ValueError(f"Weight candidate has unsupported features: {sorted(unknown)}")
        values = {feature: float(self.weights[feature]) for feature in FEATURE_COLS}
        if any(not np.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("All feature multipliers must be finite and positive.")
        object.__setattr__(self, "weights", MappingProxyType(values))


@dataclass(frozen=True)
class WeightTrialResult:
    candidate: WeightCandidate
    mean_ndcg_at_k: float
    mean_recall_at_k: float


@dataclass(frozen=True)
class TuningResult:
    config: TuningConfig
    partition: PlaylistPartition
    trials: tuple[WeightTrialResult, ...]
    selected_trial_name: str

    @property
    def selected_trial(self) -> WeightTrialResult:
        return next(
            trial for trial in self.trials if trial.candidate.name == self.selected_trial_name
        )

    @property
    def selected_weights(self) -> dict[str, float]:
        return dict(self.selected_trial.candidate.weights)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "weight_semantics": WEIGHT_SEMANTICS,
            "selection": {
                "primary_metric": "mean_ndcg_at_k",
                "tie_break_metric": "mean_recall_at_k",
                "selected_trial": self.selected_trial_name,
            },
            "selected_weights": self.selected_weights,
            "partition": {
                "tuning_playlist_ids": list(self.partition.tuning_playlist_ids),
                "test_playlist_ids": list(self.partition.test_playlist_ids),
            },
            "config": asdict(self.config),
            "trials": [
                {
                    "name": trial.candidate.name,
                    "source": trial.candidate.source,
                    "weights": dict(trial.candidate.weights),
                    "mean_ndcg_at_k": trial.mean_ndcg_at_k,
                    "mean_recall_at_k": trial.mean_recall_at_k,
                }
                for trial in self.trials
            ],
            "test_playlists_evaluated": False,
            "final_benchmark_required": True,
            "final_benchmark_note": (
                "Run the final benchmark once on test_playlist_ids. Do not use those "
                "results to revise the selected weights."
            ),
        }


def partition_playlist_ids(
    memberships: pd.DataFrame,
    config: TuningConfig,
) -> PlaylistPartition:
    """Create a deterministic playlist-level partition without splitting rows."""
    labels = normalize_memberships(memberships)
    playlist_ids = sorted(labels["playlist_id"].dropna().astype(str).unique())
    if len(playlist_ids) < config.min_playlists:
        raise ValueError(
            f"Weight tuning requires at least {config.min_playlists} playlists; "
            f"received {len(playlist_ids)}."
        )

    rng = np.random.default_rng(config.random_state)
    shuffled = np.asarray(playlist_ids, dtype=object)[rng.permutation(len(playlist_ids))]
    requested_test_count = int(np.ceil(len(playlist_ids) * config.test_fraction))
    test_count = max(requested_test_count, config.min_test_playlists)
    test_count = min(test_count, len(playlist_ids) - config.min_tuning_playlists)
    if test_count < config.min_test_playlists:
        raise ValueError("Not enough playlists to satisfy the requested partition minimums.")

    test_ids = tuple(sorted(str(value) for value in shuffled[:test_count]))
    tuning_ids = tuple(sorted(str(value) for value in shuffled[test_count:]))
    if len(tuning_ids) < config.min_tuning_playlists:
        raise ValueError("Not enough playlists remain in the tuning partition.")
    return PlaylistPartition(
        tuning_playlist_ids=tuning_ids,
        test_playlist_ids=test_ids,
    )


def generate_weight_candidates(config: TuningConfig) -> tuple[WeightCandidate, ...]:
    """Generate uniform, current-default, and reproducible log-uniform candidates."""
    uniform = dict.fromkeys(FEATURE_COLS, 1.0)
    hand_set = {feature: float(DEFAULT_WEIGHTS.get(feature, 1.0)) for feature in FEATURE_COLS}
    candidates = [
        WeightCandidate("uniform", uniform, "baseline"),
        WeightCandidate("hand_set_defaults", hand_set, "baseline"),
    ]

    rng = np.random.default_rng(config.random_state)
    log_low = np.log(config.weight_min)
    log_high = np.log(config.weight_max)
    for index in range(config.num_trials - len(candidates)):
        sampled = np.exp(rng.uniform(log_low, log_high, size=len(FEATURE_COLS)))
        candidates.append(
            WeightCandidate(
                name=f"random_{index + 1:03d}",
                weights=dict(zip(FEATURE_COLS, sampled, strict=True)),
                source="log_uniform_random",
            )
        )
    return tuple(candidates)


def _trial_strategies(
    candidates: tuple[WeightCandidate, ...],
    config: TuningConfig,
) -> tuple[EvaluationStrategy, ...]:
    strategies = []
    for candidate in candidates:
        policy = RecommendationPolicy(
            strategy="weighted_cosine",
            user_weights=candidate.weights,
            min_popularity=config.min_popularity,
            year_range=DEPLOYED_POLICY.year_range,
            use_pca=True,
            pca_components=config.pca_components,
            same_artist_exclusion=DEPLOYED_POLICY.same_artist_exclusion,
            randomize_results=False,
            random_state=config.random_state,
        )
        strategies.append(
            EvaluationStrategy(
                name=candidate.name,
                policy=policy,
                description="Deterministic weighted cosine plus PCA tuning trial.",
            )
        )
    return tuple(strategies)


def _trial_results(
    candidates: tuple[WeightCandidate, ...],
    summary: pd.DataFrame,
) -> tuple[WeightTrialResult, ...]:
    required = {"strategy", "ndcg_at_k", "recall_at_k"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Benchmark summary is missing tuning metrics: {sorted(missing)}")

    rows = summary.set_index("strategy")
    results = []
    for candidate in candidates:
        if candidate.name not in rows.index:
            raise ValueError(f"Benchmark did not return trial {candidate.name!r}.")
        row = rows.loc[candidate.name]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Benchmark returned duplicate rows for trial {candidate.name!r}.")
        ndcg = float(row["ndcg_at_k"])
        recall = float(row["recall_at_k"])
        if not np.isfinite(ndcg) or not np.isfinite(recall):
            raise ValueError(f"Trial {candidate.name!r} returned non-finite metrics.")
        results.append(
            WeightTrialResult(
                candidate=candidate,
                mean_ndcg_at_k=ndcg,
                mean_recall_at_k=recall,
            )
        )
    return tuple(results)


def tune_recommender_weights(
    catalog,
    memberships: pd.DataFrame,
    config: TuningConfig | None = None,
    *,
    evaluator: Callable | None = None,
    progress_callback: Callable[[int, int, str, int, str], None] | None = None,
) -> TuningResult:
    """Tune on one playlist partition while leaving held-out test IDs untouched."""
    cfg = config or TuningConfig()
    labels = normalize_memberships(memberships)
    partition = partition_playlist_ids(labels, cfg)
    tuning_id_set = set(partition.tuning_playlist_ids)
    tuning_labels = labels[labels["playlist_id"].isin(tuning_id_set)].copy()
    if set(tuning_labels["playlist_id"]) & set(partition.test_playlist_ids):
        raise AssertionError("Test playlist rows leaked into the tuning partition.")

    candidates = generate_weight_candidates(cfg)
    strategies = _trial_strategies(candidates, cfg)
    evaluation_config = EvaluationConfig(
        top_k=cfg.top_k,
        seed_size=cfg.seed_size,
        num_splits=cfg.num_splits,
        random_state=cfg.random_state,
        min_popularity=cfg.min_popularity,
        same_artist_exclusion=DEPLOYED_POLICY.same_artist_exclusion,
        pca_components=cfg.pca_components,
        bootstrap_samples=cfg.bootstrap_samples,
        min_playlists_for_claim=cfg.min_tuning_playlists,
    )
    benchmark_evaluator = evaluator or evaluate_benchmark
    evaluator_kwargs = {
        "config": evaluation_config,
        "strategies": strategies,
    }
    if progress_callback is not None:
        evaluator_kwargs["progress_callback"] = progress_callback
    benchmark = benchmark_evaluator(catalog, tuning_labels, **evaluator_kwargs)
    trials = _trial_results(candidates, benchmark.summary)
    selected = max(
        enumerate(trials),
        key=lambda item: (
            item[1].mean_ndcg_at_k,
            item[1].mean_recall_at_k,
            -item[0],
        ),
    )[1]
    return TuningResult(
        config=cfg,
        partition=partition,
        trials=trials,
        selected_trial_name=selected.candidate.name,
    )


def write_tuning_result(result: TuningResult, path: str | Path) -> Path:
    """Write a stable JSON tuning artifact for later held-out evaluation."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False)
    output_path.write_text(f"{payload}\n", encoding="utf-8")
    return output_path
