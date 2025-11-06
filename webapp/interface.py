# webapp/interface.py
import sys
from pathlib import Path

# Ensure project root (spotify-recommender) is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pathlib import Path
import pickle
import pandas as pd

from utils.spotify_auth import get_spotify_client
from utils.spotify_integration import extract_playlist_id, fetch_playlist_profile
from utils.merge_datasets import get_merged_dataset, _fingerprint_inputs
from recommender.recommend import recommend
from recommender.weightings import DEFAULT_WEIGHTS
from utils.spotify_playlist import create_recommendation_playlist


def get_recommendations(playlist_url: str, top_n: int = 10) -> pd.DataFrame:
    """
    End-to-end wrapper that:
      1) Loads Spotify client
      2) Loads cached dataset + indexes
      3) Matches playlist tracks
      4) Computes recommendations
      5) Returns a DataFrame of recommended tracks
    """
    sp = get_spotify_client()
    playlist_id = extract_playlist_id(playlist_url)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    catalog_paths = [
        str(DATA_RAW / "spotify_data.csv"),
        str(DATA_RAW / "spotify_top_songs_audio_features.csv"),
        str(DATA_RAW / "tracks_features.csv"),
    ]

    catalog_df = get_merged_dataset(catalog_paths)

    fp = _fingerprint_inputs(catalog_paths)
    index_file = Path(".dataset_cache") / f"indexes_{fp}.pkl"
    with open(index_file, "rb") as f:
        indexes = pickle.load(f)

    user_tracks_df = fetch_playlist_profile(sp, playlist_id, indexes, catalog_df)
    if user_tracks_df.empty:
        raise ValueError("No valid tracks from this playlist could be matched to the catalog.")

    recs = recommend(
        catalog_paths=catalog_paths,
        user_tracks_df=user_tracks_df,
        top_n=top_n,
        user_weights=DEFAULT_WEIGHTS,
    )

    # --- Fetch album-art URLs for each recommended track ---
    art_urls = []
    for sid in recs["spotify_id"]:
        try:
            # medium-size image (index 1) is usually ~300 px
            track = sp.track(sid)
            art = track["album"]["images"][1]["url"] if track["album"]["images"] else None
        except Exception:
            art = None
        art_urls.append(art)

    recs["album_art_url"] = art_urls
    return recs.reset_index(drop=True)

def add_recommendations_to_spotify(recs_df, playlist_name="Recommended Songs", sp=None):
    """
    Creates a new Spotify playlist in the user's account containing the recommended songs.
    Returns the new playlist URL.
    """
    sp = sp or get_spotify_client()
    user_id = sp.me()["id"]

    track_uris = [f"spotify:track:{sid}" for sid in recs_df["spotify_id"].dropna()]
    if not track_uris:
        raise ValueError("No valid Spotify track IDs found in recommendations.")

    playlist_url = create_recommendation_playlist(sp, user_id, track_uris, name=playlist_name)
    return playlist_url

