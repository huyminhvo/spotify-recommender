from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
import pandas as pd

from recommender.policy import DEPLOYED_POLICY, RecommendationPolicy
from recommender.recommend import (
    prepare_recommendation_candidates,
    recommend_from_prepared_candidates,
)
from recommender.retrieve import filter_candidates
from recommender.steering import (
    DEFAULT_STEERING_STRENGTH,
    normalize_adjustments,
)
from recommender.weightings import DEFAULT_WEIGHTS
from utils.matcher import canon_artist_primary

DEFAULT_STEERING_ABLATION: dict[str, float] = {
    "energy": 0.25,
    "valence": 0.25,
}


@dataclass(frozen=True)
class EvaluationStrategy:
    """A named recommendation policy and optional steering intervention."""

    name: str
    policy: RecommendationPolicy
    adjustments: Mapping[str, float] | None = None
    diagnostic_adjustments: Mapping[str, float] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        for field_name in ("adjustments", "diagnostic_adjustments"):
            adjustments = getattr(self, field_name)
            if adjustments is not None:
                object.__setattr__(
                    self,
                    field_name,
                    MappingProxyType(dict(adjustments)),
                )


_DETERMINISTIC_PCA_POLICY = replace(
    DEPLOYED_POLICY,
    randomize_results=False,
    random_state=0,
)

DEFAULT_STRATEGIES: tuple[EvaluationStrategy, ...] = (
    EvaluationStrategy(
        name="random",
        policy=RecommendationPolicy(
            strategy="random",
            min_popularity=DEPLOYED_POLICY.min_popularity,
            use_pca=False,
        ),
        description="Random eligible candidates.",
    ),
    EvaluationStrategy(
        name="popularity",
        policy=RecommendationPolicy(
            strategy="popularity",
            min_popularity=DEPLOYED_POLICY.min_popularity,
            use_pca=False,
        ),
        description="Eligible candidates ordered by Spotify popularity.",
    ),
    EvaluationStrategy(
        name="unweighted_cosine",
        policy=RecommendationPolicy(
            strategy="unweighted_cosine",
            min_popularity=DEPLOYED_POLICY.min_popularity,
            use_pca=False,
        ),
        description="Cosine similarity without feature weights or PCA.",
    ),
    EvaluationStrategy(
        name="weighted_cosine",
        policy=RecommendationPolicy(
            strategy="weighted_cosine",
            user_weights=DEFAULT_WEIGHTS,
            min_popularity=DEPLOYED_POLICY.min_popularity,
            use_pca=False,
        ),
        description="Hand-set feature weights without PCA.",
    ),
    EvaluationStrategy(
        name="weighted_cosine_pca",
        policy=_DETERMINISTIC_PCA_POLICY,
        diagnostic_adjustments=DEFAULT_STEERING_ABLATION,
        description="Weighted cosine after PCA, without result randomization.",
    ),
    EvaluationStrategy(
        name="deployed",
        policy=DEPLOYED_POLICY,
        description="Exact first-request deployed policy with reproducible random seeds.",
    ),
    EvaluationStrategy(
        name="weighted_cosine_pca_steered",
        policy=_DETERMINISTIC_PCA_POLICY,
        adjustments=DEFAULT_STEERING_ABLATION,
        diagnostic_adjustments=DEFAULT_STEERING_ABLATION,
        description="PCA policy reranked toward a fixed higher-energy/valence target.",
    ),
)


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int = 10
    seed_size: int = 5
    num_splits: int = 10
    random_state: int = 0
    min_popularity: int | None = DEPLOYED_POLICY.min_popularity
    same_artist_exclusion: bool = DEPLOYED_POLICY.same_artist_exclusion
    pca_components: int = DEPLOYED_POLICY.pca_components
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    min_playlists_for_claim: int = 50
    near_duplicate_jaccard: float = 0.80
    # Retained for callers of the former API. Positives now always include every
    # non-seed playlist item, so this value is intentionally ignored.
    holdout_size: int | None = None
    # Retained for callers of the former API. PCA is now an explicit ablation.
    use_pca: bool | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if self.seed_size < 1:
            raise ValueError("seed_size must be at least 1.")
        if self.num_splits < 1:
            raise ValueError("num_splits must be at least 1.")
        if self.pca_components < 1:
            raise ValueError("pca_components must be at least 1.")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be at least 1.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")
        if self.min_playlists_for_claim < 1:
            raise ValueError("min_playlists_for_claim must be at least 1.")
        if not 0.0 <= self.near_duplicate_jaccard <= 1.0:
            raise ValueError("near_duplicate_jaccard must be between 0 and 1.")


@dataclass(frozen=True)
class EvaluationResult:
    per_split: pd.DataFrame
    recommendations: pd.DataFrame
    summary: pd.DataFrame
    skipped: pd.DataFrame
    audit: dict[str, object]


def _clean_id(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_memberships(
    memberships: pd.DataFrame,
    playlist_col: str = "playlist_id",
) -> pd.DataFrame:
    """Normalize legacy or label-only playlist membership tables."""
    if playlist_col not in memberships.columns:
        raise KeyError(f"Missing playlist column: {playlist_col}")

    labels = memberships.copy()
    if playlist_col != "playlist_id":
        if "playlist_id" in labels.columns:
            raise ValueError(
                "Cannot rename playlist_col to 'playlist_id' because that column already exists."
            )
        labels = labels.rename(columns={playlist_col: "playlist_id"})

    if "catalog_spotify_id" not in labels.columns:
        if "spotify_id" not in labels.columns:
            raise KeyError("Membership labels need catalog_spotify_id or legacy spotify_id.")
        labels["catalog_spotify_id"] = labels["spotify_id"]

    if "source_spotify_id" not in labels.columns:
        if "spotify_id" in labels.columns:
            labels["source_spotify_id"] = labels["spotify_id"]
        else:
            labels["source_spotify_id"] = labels["catalog_spotify_id"]

    if "position" not in labels.columns:
        labels["position"] = labels.groupby("playlist_id", sort=False).cumcount()

    labels["playlist_id"] = labels["playlist_id"].map(_clean_id)
    labels["source_spotify_id"] = labels["source_spotify_id"].map(_clean_id)
    labels["catalog_spotify_id"] = labels["catalog_spotify_id"].map(_clean_id)
    labels = labels.dropna(subset=["playlist_id"]).copy()
    labels["matched"] = labels["catalog_spotify_id"].notna()

    position_fallback = labels.groupby("playlist_id", sort=False).cumcount()
    labels["position"] = pd.to_numeric(labels["position"], errors="coerce")
    labels["position"] = labels["position"].fillna(position_fallback).astype(int)
    labels["_membership_key"] = labels["source_spotify_id"]
    missing_source = labels["_membership_key"].isna()
    labels.loc[missing_source, "_membership_key"] = labels.loc[missing_source].apply(
        lambda row: f"position:{row['position']}", axis=1
    )
    labels = labels.sort_values(["playlist_id", "position"], kind="stable")
    labels = labels.drop_duplicates(
        subset=["playlist_id", "_membership_key"],
        keep="first",
    )
    return labels.drop(columns="_membership_key").reset_index(drop=True)


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def ranking_metrics(
    recommended_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
    *,
    relevant_count: int | None = None,
) -> dict[str, float]:
    """Compute bounded binary ranking metrics without double-counting duplicates."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = {_clean_id(track_id) for track_id in relevant_ids}
    relevant.discard(None)
    denominator = len(relevant) if relevant_count is None else int(relevant_count)
    if denominator < len(relevant):
        raise ValueError("relevant_count cannot be smaller than the relevant ID set.")

    seen: set[str] = set()
    hits: list[int] = []
    for track_id in recommended_ids[:k]:
        clean_id = _clean_id(track_id)
        if clean_id is None or clean_id in seen:
            hits.append(0)
            continue
        seen.add(clean_id)
        hits.append(1 if clean_id in relevant else 0)
    hits.extend([0] * (k - len(hits)))

    if denominator == 0:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    hit_count = sum(hits)
    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(denominator, k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return {
        "precision_at_k": float(hit_count / k),
        "recall_at_k": float(hit_count / denominator),
        "hit_rate_at_k": float(hit_count > 0),
        "ndcg_at_k": float(dcg / idcg if idcg else 0.0),
    }


def recommendation_diagnostics(
    recs: pd.DataFrame,
    strategy: EvaluationStrategy | str,
    top_k: int,
) -> dict[str, float]:
    """Per-request diagnostics; catalog coverage is aggregated across requests."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    rec_count = len(recs)
    if isinstance(strategy, EvaluationStrategy):
        uses_similarity = strategy.policy.strategy in {"weighted_cosine", "unweighted_cosine"}
    else:
        uses_similarity = "cosine" in strategy or strategy == "deployed"

    if rec_count and "artist_primary_canon" in recs.columns:
        artists = recs["artist_primary_canon"].dropna()
    elif rec_count and "artists_raw" in recs.columns:
        artists = recs["artists_raw"].apply(canon_artist_primary).dropna()
    else:
        artists = pd.Series(dtype=object)
    artists = artists[artists != ""]
    artist_diversity = artists.nunique() / rec_count if rec_count else 0.0

    avg_similarity = np.nan
    if uses_similarity and "similarity" in recs.columns and not recs.empty:
        avg_similarity = float(pd.to_numeric(recs["similarity"], errors="coerce").mean())

    return {
        "fill_rate": float(rec_count / top_k),
        "artist_diversity": float(artist_diversity),
        "artist_duplication_rate": float(1.0 - artist_diversity if rec_count else 0.0),
        "avg_similarity": avg_similarity,
        "avg_recommendation_popularity": (
            float(pd.to_numeric(recs["popularity"], errors="coerce").mean())
            if rec_count and "popularity" in recs.columns
            else np.nan
        ),
    }


def split_playlist_tracks(
    playlist_df: pd.DataFrame,
    seed_size: int,
    holdout_size: int | None = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, set[str]]:
    """Backward-compatible split helper using every non-seed track as relevant."""
    id_col = "catalog_spotify_id" if "catalog_spotify_id" in playlist_df.columns else "spotify_id"
    tracks = playlist_df.dropna(subset=[id_col]).drop_duplicates(id_col)
    if len(tracks) < seed_size + 1:
        raise ValueError("playlist must contain at least seed_size + 1 unique tracks")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(tracks))
    seed_idx = order[:seed_size]
    remaining = order[seed_size:]
    seeds = tracks.iloc[seed_idx].copy()
    return seeds, set(tracks.iloc[remaining][id_col].map(str))


def _split_memberships(
    playlist_df: pd.DataFrame,
    seed_size: int,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    matched_ids = playlist_df["catalog_spotify_id"].dropna().astype(str).drop_duplicates().tolist()
    if len(matched_ids) < seed_size:
        raise ValueError(f"playlist has fewer than {seed_size} matched seed tracks")

    rng = np.random.default_rng(random_state)
    seed_ids = [matched_ids[index] for index in rng.permutation(len(matched_ids))[:seed_size]]
    positives = playlist_df[~playlist_df["catalog_spotify_id"].isin(seed_ids)].copy()
    if positives.empty:
        raise ValueError("playlist has no non-seed positive tracks")
    return seed_ids, positives


def _load_seed_tracks(catalog, spotify_ids: Sequence[str]) -> pd.DataFrame:
    if hasattr(catalog, "load_tracks"):
        seeds = catalog.load_tracks(spotify_ids)
    else:
        seeds = catalog[catalog["spotify_id"].isin(spotify_ids)].copy()
        order = {track_id: index for index, track_id in enumerate(spotify_ids)}
        seeds["_seed_order"] = seeds["spotify_id"].map(order)
        seeds = seeds.sort_values("_seed_order").drop(columns="_seed_order")

    loaded_ids = set(seeds["spotify_id"].dropna().astype(str))
    missing = [track_id for track_id in spotify_ids if track_id not in loaded_ids]
    if missing:
        raise ValueError(f"seed tracks missing from item catalog: {missing[:5]}")
    return seeds.reset_index(drop=True)


def _policy_for_config(
    policy: RecommendationPolicy,
    config: EvaluationConfig,
    random_state: int,
) -> RecommendationPolicy:
    return replace(
        policy,
        min_popularity=config.min_popularity,
        same_artist_exclusion=config.same_artist_exclusion,
        pca_components=config.pca_components,
        random_state=random_state,
    )


def _steering_targets(
    seeds: pd.DataFrame,
    adjustments: Mapping[str, float] | None,
) -> dict[str, float]:
    targets: dict[str, float] = {}
    for feature, delta in normalize_adjustments(adjustments).items():
        if feature not in seeds:
            continue
        baseline = float(pd.to_numeric(seeds[feature], errors="coerce").median())
        if not np.isfinite(baseline):
            continue
        targets[feature] = float(np.clip(baseline + delta * DEFAULT_STEERING_STRENGTH, 0.0, 1.0))
    return targets


def _target_distance(recs: pd.DataFrame, targets: Mapping[str, float]) -> float:
    distances = []
    for feature, target in targets.items():
        if feature not in recs or recs.empty:
            continue
        values = pd.to_numeric(recs[feature], errors="coerce")
        distance = np.abs(values - target).mean()
        if np.isfinite(distance):
            distances.append(float(distance))
    return float(np.mean(distances)) if distances else np.nan


def _eligible_catalog_size(catalog, config: EvaluationConfig) -> int | None:
    if hasattr(catalog, "count_candidates"):
        return catalog.count_candidates(
            min_popularity=config.min_popularity,
        )
    if isinstance(catalog, pd.DataFrame):
        return len(
            filter_candidates(
                catalog,
                min_popularity=config.min_popularity,
            )
        )
    return None


def audit_memberships(
    memberships: pd.DataFrame,
    *,
    min_playlists: int = 50,
    near_duplicate_jaccard: float = 0.80,
) -> dict[str, object]:
    """Summarize label scale, matchability, and track-set overlap."""
    if min_playlists < 1:
        raise ValueError("min_playlists must be at least 1.")
    if not 0.0 <= near_duplicate_jaccard <= 1.0:
        raise ValueError("near_duplicate_jaccard must be between 0 and 1.")

    labels = normalize_memberships(memberships)
    playlist_sizes = labels.groupby("playlist_id").size()
    matched = int(labels["matched"].sum())

    track_sets = {
        playlist_id: set(group["source_spotify_id"].dropna().astype(str))
        for playlist_id, group in labels.groupby("playlist_id", sort=False)
    }
    max_jaccard = 0.0
    near_duplicate_pairs = 0
    playlist_ids = list(track_sets)
    for left_index, left_id in enumerate(playlist_ids):
        for right_id in playlist_ids[left_index + 1 :]:
            left = track_sets[left_id]
            right = track_sets[right_id]
            union = left | right
            score = len(left & right) / len(union) if union else 0.0
            max_jaccard = max(max_jaccard, score)
            if score >= near_duplicate_jaccard:
                near_duplicate_pairs += 1

    num_playlists = int(labels["playlist_id"].nunique())
    warnings: list[str] = []
    if num_playlists < min_playlists:
        playlist_word = "playlist is" if num_playlists == 1 else "playlists are"
        warnings.append(
            f"Only {num_playlists} {playlist_word} labeled; at least {min_playlists} "
            "are required before making recommendation-quality claims."
        )
    if near_duplicate_pairs:
        warnings.append(
            f"{near_duplicate_pairs} playlist pairs have track-set Jaccard overlap "
            f">= {near_duplicate_jaccard:.2f}."
        )

    return {
        "num_playlists": num_playlists,
        "num_memberships": len(labels),
        "matched_memberships": matched,
        "match_rate": float(matched / len(labels)) if len(labels) else 0.0,
        "min_playlist_size": int(playlist_sizes.min()) if len(playlist_sizes) else 0,
        "median_playlist_size": (float(playlist_sizes.median()) if len(playlist_sizes) else 0.0),
        "max_playlist_size": int(playlist_sizes.max()) if len(playlist_sizes) else 0,
        "max_pairwise_jaccard": float(max_jaccard),
        "near_duplicate_pairs": near_duplicate_pairs,
        "benchmark_ready": num_playlists >= min_playlists,
        "warnings": warnings,
    }


SUMMARY_METRICS = [
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


def bootstrap_confidence_intervals(
    per_playlist: pd.DataFrame,
    metric_cols: Sequence[str] = SUMMARY_METRICS,
    *,
    bootstrap_samples: int = 2_000,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    """Cluster-bootstrap playlists after split-level metrics have been averaged."""
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    rows = []
    alpha = (1.0 - confidence_level) / 2.0
    for strategy, strategy_df in per_playlist.groupby("strategy", sort=False):
        strategy_df = strategy_df.reset_index(drop=True)
        rng = np.random.default_rng(_stable_seed(random_state, strategy, "bootstrap"))
        row: dict[str, object] = {
            "strategy": strategy,
            "num_playlists": len(strategy_df),
        }
        for metric in metric_cols:
            values = pd.to_numeric(strategy_df[metric], errors="coerce").to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            row[metric] = float(np.mean(finite_values)) if len(finite_values) else np.nan
            if not len(finite_values):
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
                continue
            samples = rng.choice(
                finite_values,
                size=(bootstrap_samples, len(finite_values)),
                replace=True,
            ).mean(axis=1)
            row[f"{metric}_ci_low"] = float(np.quantile(samples, alpha))
            row[f"{metric}_ci_high"] = float(np.quantile(samples, 1.0 - alpha))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_evaluations(
    per_split: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    catalog_size: int | None,
    config: EvaluationConfig,
) -> pd.DataFrame:
    if per_split.empty:
        return pd.DataFrame()

    per_playlist = (
        per_split.groupby(["playlist_id", "strategy"], as_index=False)[SUMMARY_METRICS]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    summary = bootstrap_confidence_intervals(
        per_playlist,
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        random_state=config.random_state,
    )

    request_counts = per_split.groupby("strategy", as_index=False).agg(
        num_evaluations=("split_id", "size"),
        mean_candidate_pool_size=("candidate_pool_size", "mean"),
    )
    summary = summary.merge(request_counts, on="strategy", how="left")

    exposure = summary[["strategy"]].copy()
    if not recommendations.empty:
        recommendation_counts = recommendations.groupby("strategy", as_index=False).agg(
            unique_recommendations=("spotify_id", "nunique"),
            total_recommendations=("spotify_id", "size"),
        )
        exposure = exposure.merge(recommendation_counts, on="strategy", how="left")
    else:
        exposure["unique_recommendations"] = 0
        exposure["total_recommendations"] = 0
    exposure[["unique_recommendations", "total_recommendations"]] = (
        exposure[["unique_recommendations", "total_recommendations"]].fillna(0).astype(int)
    )
    exposure["recommendation_repeat_rate"] = np.where(
        exposure["total_recommendations"] > 0,
        1.0 - exposure["unique_recommendations"] / exposure["total_recommendations"],
        0.0,
    )
    if catalog_size:
        exposure["catalog_coverage"] = exposure["unique_recommendations"] / catalog_size
    else:
        exposure["catalog_coverage"] = np.nan
    summary = summary.merge(exposure, on="strategy", how="left")
    return summary.sort_values(
        ["ndcg_at_k", "recall_at_k"],
        ascending=False,
    ).reset_index(drop=True)


def evaluate_benchmark(
    catalog,
    memberships: pd.DataFrame,
    config: EvaluationConfig | None = None,
    strategies: Sequence[EvaluationStrategy] = DEFAULT_STRATEGIES,
    *,
    playlist_col: str = "playlist_id",
    catalog_size: int | None = None,
    progress_callback: Callable[[int, int, str, int, str], None] | None = None,
) -> EvaluationResult:
    """Evaluate separate playlist labels against an item catalog."""
    cfg = config or EvaluationConfig()
    labels = normalize_memberships(memberships, playlist_col=playlist_col)
    audit = audit_memberships(
        labels,
        min_playlists=cfg.min_playlists_for_claim,
        near_duplicate_jaccard=cfg.near_duplicate_jaccard,
    )
    if catalog_size is None:
        catalog_size = _eligible_catalog_size(catalog, cfg)
    audit["eligible_catalog_size"] = catalog_size

    split_rows: list[dict[str, object]] = []
    recommendation_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    total_work = labels["playlist_id"].nunique() * cfg.num_splits * len(strategies)
    completed_work = 0

    for playlist_id, playlist_df in labels.groupby("playlist_id", sort=False):
        for split_id in range(cfg.num_splits):
            split_seed = _stable_seed(cfg.random_state, playlist_id, split_id, "split")
            try:
                seed_ids, positives = _split_memberships(
                    playlist_df,
                    seed_size=cfg.seed_size,
                    random_state=split_seed,
                )
                seeds = _load_seed_tracks(catalog, seed_ids)
            except ValueError as exc:
                skipped_rows.append(
                    {
                        "playlist_id": playlist_id,
                        "split_id": split_id,
                        "reason": str(exc),
                    }
                )
                completed_work += len(strategies)
                if progress_callback and strategies:
                    progress_callback(
                        completed_work,
                        total_work,
                        str(playlist_id),
                        split_id,
                        "skipped",
                    )
                continue

            total_positive_count = len(positives)
            matched_positive_ids = set(positives["catalog_spotify_id"].dropna().astype(str))
            matched_positive_count = len(matched_positive_ids)
            prepared_cache: dict[tuple, object] = {}

            for strategy in strategies:
                strategy_seed = _stable_seed(
                    cfg.random_state,
                    playlist_id,
                    split_id,
                    strategy.name,
                )
                policy = _policy_for_config(strategy.policy, cfg, strategy_seed)
                candidate_key = (
                    policy.min_popularity,
                    policy.year_range,
                    policy.same_artist_exclusion,
                )
                if candidate_key not in prepared_cache:
                    prepared_cache[candidate_key] = prepare_recommendation_candidates(
                        catalog=catalog,
                        user_tracks_df=seeds,
                        top_n=cfg.top_k,
                        **policy.candidate_kwargs(),
                    )
                prepared = prepared_cache[candidate_key]
                candidate_ids = set(prepared.candidates["spotify_id"].dropna().astype(str))
                retrievable_ids = matched_positive_ids & candidate_ids

                recs = recommend_from_prepared_candidates(
                    prepared=prepared,
                    user_tracks_df=seeds,
                    top_n=cfg.top_k,
                    adjustments=strategy.adjustments,
                    **policy.scoring_kwargs(),
                )
                recommended_ids = recs["spotify_id"].dropna().astype(str).tolist()
                metrics = ranking_metrics(
                    recommended_ids,
                    matched_positive_ids,
                    cfg.top_k,
                    relevant_count=total_positive_count,
                )
                unique_hits = len(set(recommended_ids[: cfg.top_k]) & matched_positive_ids)
                diagnostics = recommendation_diagnostics(recs, strategy, cfg.top_k)
                diagnostic_targets = _steering_targets(
                    seeds,
                    strategy.diagnostic_adjustments,
                )

                split_rows.append(
                    {
                        "playlist_id": playlist_id,
                        "split_id": split_id,
                        "strategy": strategy.name,
                        "split_seed": split_seed,
                        "recommendation_seed": strategy_seed,
                        "num_seed_tracks": len(seed_ids),
                        "num_positive_tracks": total_positive_count,
                        "num_matched_positive_tracks": matched_positive_count,
                        "num_retrievable_positive_tracks": len(retrievable_ids),
                        "candidate_pool_size": prepared.candidate_pool_size,
                        "matched_recall_ceiling": (
                            matched_positive_count / total_positive_count
                            if total_positive_count
                            else 0.0
                        ),
                        "candidate_recall_ceiling": (
                            len(retrievable_ids) / total_positive_count
                            if total_positive_count
                            else 0.0
                        ),
                        "matched_recall_at_k": (
                            unique_hits / matched_positive_count if matched_positive_count else 0.0
                        ),
                        "retrievable_recall_at_k": (
                            unique_hits / len(retrievable_ids) if retrievable_ids else 0.0
                        ),
                        "steering_target_distance": _target_distance(
                            recs,
                            diagnostic_targets,
                        ),
                        **metrics,
                        **diagnostics,
                    }
                )

                relevant_ids = matched_positive_ids
                for rank, (_, rec) in enumerate(recs.head(cfg.top_k).iterrows(), start=1):
                    spotify_id = _clean_id(rec.get("spotify_id"))
                    if spotify_id is None:
                        continue
                    recommendation_rows.append(
                        {
                            "playlist_id": playlist_id,
                            "split_id": split_id,
                            "strategy": strategy.name,
                            "rank": rank,
                            "spotify_id": spotify_id,
                            "relevant": spotify_id in relevant_ids,
                            "retrievable_positive": spotify_id in retrievable_ids,
                            "score": rec.get("score"),
                            "similarity": rec.get("similarity"),
                        }
                    )
                completed_work += 1
                if progress_callback:
                    progress_callback(
                        completed_work,
                        total_work,
                        str(playlist_id),
                        split_id,
                        strategy.name,
                    )

    per_split = pd.DataFrame(split_rows)
    recommendations = pd.DataFrame(recommendation_rows)
    skipped = pd.DataFrame(skipped_rows)
    summary = summarize_evaluations(
        per_split,
        recommendations,
        catalog_size=catalog_size,
        config=cfg,
    )
    return EvaluationResult(
        per_split=per_split,
        recommendations=recommendations,
        summary=summary,
        skipped=skipped,
        audit=audit,
    )


def evaluate_playlist(
    catalog,
    playlist_df: pd.DataFrame,
    config: EvaluationConfig | None = None,
    strategies: Sequence[EvaluationStrategy] = DEFAULT_STRATEGIES,
) -> pd.DataFrame:
    """Evaluate one playlist and return split-level rows."""
    cfg = config or EvaluationConfig(num_splits=1)
    result = evaluate_benchmark(
        catalog,
        playlist_df,
        config=cfg,
        strategies=strategies,
    )
    return result.per_split


def evaluate_catalog_playlists(
    catalog,
    memberships: pd.DataFrame | None = None,
    playlist_col: str = "playlist_id",
    config: EvaluationConfig | None = None,
    strategies: Sequence[EvaluationStrategy] = DEFAULT_STRATEGIES,
) -> pd.DataFrame:
    """Compatibility wrapper returning the benchmark summary."""
    if memberships is None:
        raise ValueError("Provide playlist membership labels separately from the item catalog.")
    return evaluate_benchmark(
        catalog,
        memberships,
        config=config,
        strategies=strategies,
        playlist_col=playlist_col,
    ).summary
