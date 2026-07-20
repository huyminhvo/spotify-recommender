from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.explain import explain_feature_similarity
from recommender.policy import RecommendationStrategy
from recommender.preprocess import fit_scaler, transform
from recommender.profile import build_user_profile
from recommender.retrieve import filter_candidates
from recommender.schema import FEATURE_COLS
from recommender.similarity import cosine
from recommender.steering import rerank_with_adjustments
from recommender.weightings import apply_weights
from utils.matcher import canon_artist_primary
from utils.merge_datasets import get_merged_dataset


@dataclass(frozen=True)
class PreparedCandidates:
    """A filtered candidate pool plus the catalog distribution used for scaling."""

    candidates: pd.DataFrame
    scaler_source: pd.DataFrame
    min_popularity: int | None

    @property
    def candidate_pool_size(self) -> int:
        return len(self.candidates)


def _sample_from_top_candidates(ranked, top_n, random_state=None):
    """Select a relevance-weighted subset while preserving score order."""
    result_size = min(top_n, len(ranked))
    pool_size = min(len(ranked), max(result_size, result_size * 3))
    pool = ranked.head(pool_size)
    if result_size == pool_size:
        return pool

    # Favor the best-ranked songs while leaving enough probability for nearby
    # candidates to make repeated recommendations feel fresh.
    ranks = np.arange(pool_size)
    weights = np.exp(-ranks / max(result_size, 1))
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(random_state)
    selected_positions = np.sort(
        rng.choice(pool_size, size=result_size, replace=False, p=probabilities)
    )
    return pool.iloc[selected_positions]


def prepare_recommendation_candidates(
    catalog,
    user_tracks_df,
    top_n=10,
    min_popularity=20,
    year_range=None,
    same_artist_exclusion=False,
    exclude_spotify_ids=None,
) -> PreparedCandidates:
    """Build the exact eligible candidate pool used for recommendation scoring."""
    exclude_ids = user_tracks_df["spotify_id"].dropna().tolist()
    if exclude_spotify_ids is not None:
        exclude_ids.extend(track_id for track_id in exclude_spotify_ids if track_id)
    exclude_ids = list(dict.fromkeys(exclude_ids))

    exclude_artists = None
    if same_artist_exclusion:
        if "artist_primary_canon" in user_tracks_df.columns:
            exclude_artists = user_tracks_df["artist_primary_canon"].dropna().tolist()
        else:
            exclude_artists = user_tracks_df["artists_raw"].apply(canon_artist_primary).tolist()

    def load_candidates(candidate_min_popularity):
        if hasattr(catalog, "load_candidates"):
            return catalog.load_candidates(
                exclude_ids=exclude_ids,
                exclude_artists=exclude_artists,
                min_popularity=candidate_min_popularity,
                year_range=year_range,
            )
        return filter_candidates(
            catalog,
            exclude_ids=exclude_ids,
            exclude_artists=exclude_artists,
            min_popularity=candidate_min_popularity,
            year_range=year_range,
        )

    applied_min_popularity = min_popularity
    candidates = load_candidates(min_popularity)
    if min_popularity is not None and len(candidates) < top_n:
        # If session memory or popularity filters leave too few candidates,
        # keep the no-repeat guarantee and widen quality constraints instead.
        candidates = load_candidates(None)
        applied_min_popularity = None

    scaler_source = candidates if hasattr(catalog, "load_candidates") else catalog
    return PreparedCandidates(
        candidates=candidates,
        scaler_source=scaler_source,
        min_popularity=applied_min_popularity,
    )


def _finalize_recommendations(
    recs: pd.DataFrame,
    prepared: PreparedCandidates,
    steering_targets: dict[str, float] | None = None,
) -> pd.DataFrame:
    result = recs.reset_index(drop=True)
    result.attrs["candidate_pool_size"] = prepared.candidate_pool_size
    result.attrs["candidate_min_popularity"] = prepared.min_popularity
    if steering_targets is not None:
        result.attrs["steering_targets"] = steering_targets
    return result


def recommend_from_prepared_candidates(
    prepared: PreparedCandidates,
    user_tracks_df: pd.DataFrame,
    user_weights=None,
    top_n=10,
    use_pca=True,
    pca_components=5,
    strategy: RecommendationStrategy = "weighted_cosine",
    random_state=0,
    randomize_results=False,
    adjustments=None,
) -> pd.DataFrame:
    """Score and select recommendations from an already prepared candidate pool."""
    candidates = prepared.candidates
    if candidates.empty:
        empty = candidates.assign(
            score=pd.Series(dtype=float),
            similarity=pd.Series(dtype=float),
        ).head(0)
        return _finalize_recommendations(empty, prepared)

    if strategy == "popularity":
        recs = candidates.copy()
        recs["score"] = recs["popularity"].fillna(0).astype(float)
        recs["similarity"] = recs["score"]
        recs = recs.sort_values("score", ascending=False).head(top_n)
        return _finalize_recommendations(recs, prepared)

    if strategy == "random":
        recs = candidates.sample(frac=1.0, random_state=random_state).head(top_n).copy()
        recs["score"] = list(range(len(recs), 0, -1))
        recs["similarity"] = recs["score"]
        return _finalize_recommendations(recs, prepared)

    if strategy not in {"weighted_cosine", "unweighted_cosine"}:
        raise ValueError(f"Unknown recommendation strategy: {strategy}")

    scaler_source = prepared.scaler_source
    scaler = fit_scaler(scaler_source, FEATURE_COLS)
    X_user = transform(user_tracks_df, scaler, FEATURE_COLS)
    u_vec = build_user_profile(X_user, method="median")
    X_cands = transform(candidates, scaler, FEATURE_COLS)
    candidates = candidates.copy()
    candidates["recommendation_reason"] = explain_feature_similarity(
        u_vec,
        X_cands,
        FEATURE_COLS,
    )

    weights = user_weights if strategy == "weighted_cosine" else None

    if weights is not None:
        u_vec = apply_weights(u_vec, weights, FEATURE_COLS)

    if weights is not None:
        X_cands = apply_weights(X_cands, weights, FEATURE_COLS)

    # === PCA dimensionality reduction ===
    if use_pca:
        from recommender.cluster import fit_pca, transform_pca

        # Fit PCA on the in-memory catalog or bounded store sample.
        X_catalog = transform(scaler_source, scaler, FEATURE_COLS)
        if weights is not None:
            X_catalog = apply_weights(X_catalog, weights, FEATURE_COLS)

        pca = fit_pca(X_catalog, n_components=pca_components)

        # transform everything into PCA space
        X_cands = transform_pca(X_cands, pca)
        u_vec = transform_pca(u_vec.reshape(1, -1), pca).reshape(-1)

    sims = cosine(u_vec, X_cands)
    candidates["similarity"] = sims
    scores, targets = rerank_with_adjustments(
        candidates,
        user_tracks_df,
        sims,
        adjustments,
    )
    candidates["score"] = scores
    candidates.attrs["steering_targets"] = targets
    ranked = candidates.sort_values("score", ascending=False)
    if randomize_results:
        recs = _sample_from_top_candidates(ranked, top_n, random_state)
    else:
        recs = ranked.head(top_n)
    return _finalize_recommendations(recs, prepared, steering_targets=targets)


def recommend_from_catalog(
    catalog,
    user_tracks_df,
    user_weights=None,
    top_n=10,
    min_popularity=20,
    year_range=None,
    use_pca=True,
    pca_components=5,
    strategy: RecommendationStrategy = "weighted_cosine",
    same_artist_exclusion=False,
    random_state=0,
    randomize_results=False,
    adjustments=None,
    exclude_spotify_ids=None,
):
    prepared = prepare_recommendation_candidates(
        catalog=catalog,
        user_tracks_df=user_tracks_df,
        top_n=top_n,
        min_popularity=min_popularity,
        year_range=year_range,
        same_artist_exclusion=same_artist_exclusion,
        exclude_spotify_ids=exclude_spotify_ids,
    )
    return recommend_from_prepared_candidates(
        prepared=prepared,
        user_tracks_df=user_tracks_df,
        user_weights=user_weights,
        top_n=top_n,
        use_pca=use_pca,
        pca_components=pca_components,
        strategy=strategy,
        random_state=random_state,
        randomize_results=randomize_results,
        adjustments=adjustments,
    )


def recommend(
    catalog_paths,
    user_tracks_df,
    user_weights=None,
    top_n=10,
    min_popularity=20,
    year_range=None,
    use_pca=True,
    pca_components=5,
    strategy: RecommendationStrategy = "weighted_cosine",
    same_artist_exclusion=False,
    random_state=0,
    randomize_results=False,
    adjustments=None,
    exclude_spotify_ids=None,
):
    catalog = get_merged_dataset(catalog_paths)
    return recommend_from_catalog(
        catalog=catalog,
        user_tracks_df=user_tracks_df,
        user_weights=user_weights,
        top_n=top_n,
        min_popularity=min_popularity,
        year_range=year_range,
        use_pca=use_pca,
        pca_components=pca_components,
        strategy=strategy,
        same_artist_exclusion=same_artist_exclusion,
        random_state=random_state,
        randomize_results=randomize_results,
        adjustments=adjustments,
        exclude_spotify_ids=exclude_spotify_ids,
    )
