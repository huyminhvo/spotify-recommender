from typing import Literal

from recommender.explain import explain_feature_similarity
from recommender.preprocess import fit_scaler, transform
from recommender.profile import build_user_profile
from recommender.retrieve import filter_candidates
from recommender.schema import FEATURE_COLS
from recommender.similarity import cosine
from recommender.steering import rerank_with_adjustments
from recommender.weightings import apply_weights
from utils.matcher import canon_artist_primary
from utils.merge_datasets import get_merged_dataset

RecommendationStrategy = Literal[
    "weighted_cosine",
    "unweighted_cosine",
    "popularity",
    "random",
]


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
    adjustments=None,
):
    exclude_ids = user_tracks_df["spotify_id"].dropna().tolist()
    exclude_artists = None
    if same_artist_exclusion:
        if "artist_primary_canon" in user_tracks_df.columns:
            exclude_artists = user_tracks_df["artist_primary_canon"].dropna().tolist()
        else:
            exclude_artists = user_tracks_df["artists_raw"].apply(canon_artist_primary).tolist()

    candidates = filter_candidates(
        catalog,
        exclude_ids=exclude_ids,
        exclude_artists=exclude_artists,
        min_popularity=min_popularity,
        year_range=year_range,
    )
    if candidates.empty:
        return candidates.assign(score=[], similarity=[]).head(0)

    if strategy == "popularity":
        recs = candidates.copy()
        recs["score"] = recs["popularity"].fillna(0).astype(float)
        recs["similarity"] = recs["score"]
        return recs.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    if strategy == "random":
        recs = candidates.sample(frac=1.0, random_state=random_state).head(top_n).copy()
        recs["score"] = list(range(len(recs), 0, -1))
        recs["similarity"] = recs["score"]
        return recs.reset_index(drop=True)

    if strategy not in {"weighted_cosine", "unweighted_cosine"}:
        raise ValueError(f"Unknown recommendation strategy: {strategy}")

    scaler = fit_scaler(catalog, FEATURE_COLS)
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

        # fit PCA on all catalog rows
        X_catalog = transform(catalog, scaler, FEATURE_COLS)
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
    recs = candidates.sort_values("score", ascending=False).head(top_n)
    return recs.reset_index(drop=True)


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
    adjustments=None,
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
        adjustments=adjustments,
    )
