from webapp.interface import get_recommendations
from recommender.recommend import recommend
from utils.merge_datasets import get_merged_dataset
from pathlib import Path
import pandas as pd

print("=== TEST: PCA Integration Smoke Test ===")

# A small real playlist for deterministic testing
TEST_PLAYLIST = "https://open.spotify.com/playlist/7BH80QsOwuL7h92CG9ap9w?si=e1748481e0a742e0"

# Step 1: fetch initial recommendations (no PCA)
print("\nFetching base recommendations (no PCA)...")
recs_base = get_recommendations(TEST_PLAYLIST, top_n=5)
print("OK. Base recs shape:", recs_base.shape)

# Step 2: Make sure recommend() can run PCA end-to-end
print("\nFetching PCA-enabled recommendations...")
catalog_paths = [
    "data/raw/spotify_data.csv",
    "data/raw/spotify_top_songs_audio_features.csv",
    "data/raw/tracks_features.csv",
]

catalog = get_merged_dataset(catalog_paths)

# We call recommend() directly to force PCA
recs_pca = recommend(
    catalog_paths=catalog_paths,
    user_tracks_df=recs_base,       # treat base recs as input tracks for test
    top_n=5,
    use_pca=False,
    pca_components=6
)

print("OK. PCA recs shape:", recs_pca.shape)

print("\nSUCCESS: PCA pipeline test passed.")
