import pandas as pd
from pathlib import Path

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

    sims = cosine(u_vec, X_cands)
    candidates = candidates.copy()
    candidates["similarity"] = sims
    recs = candidates.sort_values("similarity", ascending=False).head(top_n)
    return recs.reset_index(drop=True)
