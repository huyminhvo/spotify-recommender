from utils.merge_datasets import get_merged_dataset
from recommender.preprocess import fit_scaler, transform
from recommender.profile import build_user_profile
from recommender.weightings import apply_weights
from recommender.similarity import cosine
from recommender.retrieve import filter_candidates
from recommender.schema import FEATURE_COLS

def recommend(
    catalog_paths,
    user_tracks_df,
    user_weights=None,
    top_n=10,
    min_popularity=20,
    year_range=None,
    use_pca=True,
    pca_components=5
):
    catalog = get_merged_dataset(catalog_paths)
    scaler = fit_scaler(catalog, FEATURE_COLS)
    X_user = transform(user_tracks_df, scaler, FEATURE_COLS)
    u_vec = build_user_profile(X_user, method="median")

    if user_weights is not None:
        u_vec = apply_weights(u_vec, user_weights, FEATURE_COLS)

    exclude_ids = user_tracks_df["spotify_id"].dropna().tolist()
    candidates = filter_candidates(
        catalog,
        exclude_ids=exclude_ids,
        min_popularity=min_popularity,
        year_range=year_range,
    )

    X_cands = transform(candidates, scaler, FEATURE_COLS)
    if user_weights is not None:
        X_cands = apply_weights(X_cands, user_weights, FEATURE_COLS)

    # === PCA dimensionality reduction ===
    if use_pca:
        from recommender.cluster import fit_pca, transform_pca

        # fit PCA on all catalog rows
        X_catalog = transform(catalog, scaler, FEATURE_COLS)
        if user_weights is not None:
            X_catalog = apply_weights(X_catalog, user_weights, FEATURE_COLS)

        pca = fit_pca(X_catalog, n_components=pca_components)

        # transform everything into PCA space
        X_cands = transform_pca(X_cands, pca)
        u_vec = transform_pca(u_vec.reshape(1, -1), pca).reshape(-1)

    sims = cosine(u_vec, X_cands)
    candidates = candidates.copy()
    candidates["similarity"] = sims
    recs = candidates.sort_values("similarity", ascending=False).head(top_n)
    return recs.reset_index(drop=True)
