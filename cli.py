# cli.py

import argparse
from pathlib import Path
from utils.spotify_auth import get_spotify_client
from utils.spotify_integration import extract_playlist_id, fetch_playlist_profile
from utils.merge_datasets import get_merged_dataset, _fingerprint_inputs
from recommender.recommend import recommend
import pickle

def main():
    parser = argparse.ArgumentParser(description="Spotify Playlist Recommender CLI")
    parser.add_argument("--playlist", required=True, help="Spotify playlist URL, URI, or ID")
    parser.add_argument("--top_n", type=int, default=10, help="Number of recommendations to generate")
    args = parser.parse_args()

    sp = get_spotify_client()
    playlist_id = extract_playlist_id(args.playlist)

    PROJECT_ROOT = Path(__file__).resolve().parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    catalog_paths = [
        str(DATA_RAW / "spotify_data.csv"),
        str(DATA_RAW / "spotify_top_songs_audio_features.csv"),
        str(DATA_RAW / "tracks_features.csv"),
    ]

    print("getting dataset")
    # Ensure parquet + (compact) indexes exist (first run may build them)
    catalog_df = get_merged_dataset(catalog_paths)

    fp = _fingerprint_inputs(catalog_paths)
    index_file = Path(".dataset_cache") / f"indexes_{fp}.pkl"
    with open(index_file, "rb") as f:
        indexes = pickle.load(f)

    print("getting profile")
    # Match playlist tracks using compact indexes + DataFrame
    user_tracks_df = fetch_playlist_profile(sp, playlist_id, indexes, catalog_df)
    if user_tracks_df.empty:
        print("⚠️ No valid tracks from this playlist could be matched. Exiting.")
        return

    print("recommending tracks")
    recs = recommend(
        catalog_paths=catalog_paths,
        user_tracks_df=user_tracks_df,
        top_n=args.top_n,
    )

    print("\n=== Recommendations ===")
    print(recs[["title_raw", "artists_raw", "similarity"]])

if __name__ == "__main__":
    main()
