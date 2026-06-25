from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from recommender.recommend import RecommendationStrategy, recommend_from_catalog
from recommender.weightings import DEFAULT_WEIGHTS
from utils.matcher import canon_artist_primary


DEFAULT_STRATEGIES: tuple[RecommendationStrategy, ...] = (
    "random",
    "popularity",
    "unweighted_cosine",
    "weighted_cosine",
)


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: int = 10
    seed_size: int = 5
    holdout_size: int = 5
    random_state: int = 0
    min_popularity: int | None = None
    same_artist_exclusion: bool = False
    use_pca: bool = False


def ranking_metrics(recommended_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> dict[str, float]:
    relevant = set(relevant_ids)
    recs_at_k = [track_id for track_id in recommended_ids[:k] if pd.notna(track_id)]
    if not relevant:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }

    hits = [1 if track_id in relevant else 0 for track_id in recs_at_k]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant)
    hit_rate = 1.0 if any(hits) else 0.0

    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "hit_rate_at_k": float(hit_rate),
        "ndcg_at_k": float(ndcg),
    }


def recommendation_diagnostics(
    recs: pd.DataFrame,
    catalog: pd.DataFrame,
    strategy: RecommendationStrategy,
) -> dict[str, float]:
    rec_count = len(recs)
    catalog_size = catalog["spotify_id"].dropna().nunique() if "spotify_id" in catalog.columns else len(catalog)
    coverage_rate = (
        recs["spotify_id"].dropna().nunique() / catalog_size
        if rec_count and catalog_size
        else 0.0
    )

    if rec_count and "artist_primary_canon" in recs.columns:
        artists = recs["artist_primary_canon"].dropna()
    elif rec_count and "artists_raw" in recs.columns:
        artists = recs["artists_raw"].apply(canon_artist_primary).dropna()
    else:
        artists = pd.Series(dtype=object)
    artists = artists[artists != ""]
    artist_diversity = artists.nunique() / rec_count if rec_count else 0.0
    artist_duplication_rate = 1.0 - artist_diversity if rec_count else 0.0

    avg_similarity = 0.0
    if strategy in {"weighted_cosine", "unweighted_cosine"} and "similarity" in recs.columns and not recs.empty:
        avg_similarity = float(recs["similarity"].mean())

    return {
        "coverage_rate": float(coverage_rate),
        "artist_diversity": float(artist_diversity),
        "artist_duplication_rate": float(artist_duplication_rate),
        "avg_similarity": avg_similarity,
    }


def split_playlist_tracks(
    playlist_df: pd.DataFrame,
    seed_size: int,
    holdout_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, set[str]]:
    tracks = playlist_df.dropna(subset=["spotify_id"]).drop_duplicates("spotify_id")
    if len(tracks) < seed_size + 1:
        raise ValueError("playlist must contain at least seed_size + 1 unique tracks")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(tracks))
    seed_idx = order[:seed_size]
    remaining = order[seed_size:]
    holdout_idx = remaining[: min(holdout_size, len(remaining))]

    seeds = tracks.iloc[seed_idx].copy()
    holdout_ids = set(tracks.iloc[holdout_idx]["spotify_id"])
    return seeds, holdout_ids


def evaluate_playlist(
    catalog: pd.DataFrame,
    playlist_df: pd.DataFrame,
    config: EvaluationConfig | None = None,
    strategies: Sequence[RecommendationStrategy] = DEFAULT_STRATEGIES,
) -> pd.DataFrame:
    cfg = config or EvaluationConfig()
    seeds, holdout_ids = split_playlist_tracks(
        playlist_df,
        seed_size=cfg.seed_size,
        holdout_size=cfg.holdout_size,
        random_state=cfg.random_state,
    )

    rows = []
    for strategy in strategies:
        recs = recommend_from_catalog(
            catalog=catalog,
            user_tracks_df=seeds,
            user_weights=DEFAULT_WEIGHTS if strategy == "weighted_cosine" else None,
            top_n=cfg.top_k,
            min_popularity=cfg.min_popularity,
            use_pca=cfg.use_pca,
            strategy=strategy,
            same_artist_exclusion=cfg.same_artist_exclusion,
            random_state=cfg.random_state,
        )
        metrics = ranking_metrics(recs["spotify_id"].tolist(), holdout_ids, cfg.top_k)
        diagnostics = recommendation_diagnostics(recs, catalog, strategy)
        rows.append(
            {
                "strategy": strategy,
                "num_seed_tracks": len(seeds),
                "num_holdout_tracks": len(holdout_ids),
                "num_recommendations": len(recs),
                "avg_recommendation_popularity": float(recs["popularity"].fillna(0).mean())
                if not recs.empty and "popularity" in recs.columns
                else 0.0,
                **metrics,
                **diagnostics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_catalog_playlists(
    catalog: pd.DataFrame,
    playlist_col: str,
    config: EvaluationConfig | None = None,
    strategies: Sequence[RecommendationStrategy] = DEFAULT_STRATEGIES,
) -> pd.DataFrame:
    if playlist_col not in catalog.columns:
        raise KeyError(f"Missing playlist column: {playlist_col}")

    cfg = config or EvaluationConfig()
    playlist_rows = []
    for playlist_id, playlist_df in catalog.groupby(playlist_col):
        try:
            result = evaluate_playlist(catalog, playlist_df, config=cfg, strategies=strategies)
        except ValueError:
            continue
        result.insert(0, "playlist_id", playlist_id)
        playlist_rows.append(result)

    if not playlist_rows:
        return pd.DataFrame()

    per_playlist = pd.concat(playlist_rows, ignore_index=True)
    metric_cols = [
        "precision_at_k",
        "recall_at_k",
        "hit_rate_at_k",
        "ndcg_at_k",
        "avg_recommendation_popularity",
        "avg_similarity",
        "coverage_rate",
        "artist_diversity",
        "artist_duplication_rate",
        "num_recommendations",
    ]
    summary = per_playlist.groupby("strategy", as_index=False)[metric_cols].mean()
    summary.insert(1, "num_playlists", per_playlist.groupby("strategy").size().values)
    return summary.sort_values("ndcg_at_k", ascending=False).reset_index(drop=True)
