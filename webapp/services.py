from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recommender.recommend import recommend_from_catalog
from recommender.weightings import DEFAULT_WEIGHTS
from utils.merge_datasets import _fingerprint_inputs, get_merged_dataset
from utils.spotify_auth import get_spotify_client
from utils.spotify_integration import extract_playlist_id, fetch_playlist_profile
from utils.spotify_playlist import create_recommendation_playlist
from webapp.errors import (
    AppError,
    InvalidPlaylistURLError,
    MissingDatasetError,
    NoCatalogMatchesError,
    NoRecommendationTracksError,
    SpotifyAuthenticationError,
    classify_spotify_error,
)

SPOTIFY_TRACK_BATCH_SIZE = 50


@dataclass(frozen=True)
class CatalogBundle:
    paths: list[str]
    catalog: pd.DataFrame
    indexes: dict


def default_catalog_paths(root_dir: Path = ROOT_DIR) -> list[str]:
    data_raw = root_dir / "data" / "raw"
    return [
        str(data_raw / "spotify_data.csv"),
        str(data_raw / "spotify_top_songs_audio_features.csv"),
        str(data_raw / "tracks_features.csv"),
    ]


def cache_dir(root_dir: Path = ROOT_DIR) -> Path:
    return root_dir / ".dataset_cache"


def load_indexes(catalog_paths: list[str], cache_directory: Path | None = None) -> dict:
    directory = cache_directory or cache_dir()
    try:
        fp = _fingerprint_inputs(catalog_paths)
    except FileNotFoundError as exc:
        raise MissingDatasetError(str(exc)) from exc
    index_file = directory / f"indexes_{fp}.pkl"
    if not index_file.exists():
        raise MissingDatasetError(
            f"Missing catalog index cache: {index_file}. Rebuild the merged dataset first."
        )
    with index_file.open("rb") as f:
        return pickle.load(f)


def load_catalog_bundle(catalog_paths: list[str] | None = None) -> CatalogBundle:
    paths = catalog_paths or default_catalog_paths()
    directory = cache_dir()
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise MissingDatasetError(f"Missing catalog files: {missing}")
    try:
        catalog = get_merged_dataset(paths, cache_dir=str(directory))
    except FileNotFoundError as exc:
        raise MissingDatasetError(str(exc)) from exc
    indexes = load_indexes(paths, cache_directory=directory)
    return CatalogBundle(paths=paths, catalog=catalog, indexes=indexes)


def match_playlist_tracks(sp, playlist_url: str, bundle: CatalogBundle) -> pd.DataFrame:
    try:
        playlist_id = extract_playlist_id(playlist_url)
    except ValueError as exc:
        raise InvalidPlaylistURLError(str(exc)) from exc

    try:
        user_tracks = fetch_playlist_profile(sp, playlist_id, bundle.indexes, bundle.catalog)
    except Exception as exc:
        raise classify_spotify_error(exc) from exc

    if user_tracks.empty:
        raise NoCatalogMatchesError()
    return user_tracks


def generate_recommendations(
    bundle: CatalogBundle,
    user_tracks_df: pd.DataFrame,
    top_n: int = 10,
    adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    return recommend_from_catalog(
        catalog=bundle.catalog,
        user_tracks_df=user_tracks_df,
        top_n=top_n,
        user_weights=DEFAULT_WEIGHTS,
        adjustments=adjustments,
    )


def fetch_album_art_urls(sp, spotify_ids: Iterable[str]) -> list[str | None]:
    spotify_ids = list(spotify_ids)
    art_urls: list[str | None] = []
    for start in range(0, len(spotify_ids), SPOTIFY_TRACK_BATCH_SIZE):
        batch = spotify_ids[start : start + SPOTIFY_TRACK_BATCH_SIZE]
        try:
            tracks = sp.tracks(batch).get("tracks", [])
        except Exception:
            art_urls.extend([None] * len(batch))
            continue

        for track in tracks:
            images = (track or {}).get("album", {}).get("images", [])
            art_urls.append(
                images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
            )
        art_urls.extend([None] * (len(batch) - len(tracks)))
    return art_urls


def attach_album_art(sp, recs: pd.DataFrame) -> pd.DataFrame:
    recs = recs.copy()
    recs["album_art_url"] = fetch_album_art_urls(sp, recs["spotify_id"])
    return recs.reset_index(drop=True)


def get_spotify_client_or_raise():
    try:
        return get_spotify_client()
    except AppError:
        raise
    except Exception as exc:
        raise SpotifyAuthenticationError(str(exc)) from exc


def get_recommendations(
    playlist_url: str,
    top_n: int = 10,
    adjustments: dict[str, float] | None = None,
    sp=None,
) -> pd.DataFrame:
    sp = sp or get_spotify_client_or_raise()
    bundle = load_catalog_bundle()
    user_tracks = match_playlist_tracks(sp, playlist_url, bundle)
    recs = generate_recommendations(
        bundle,
        user_tracks,
        top_n=top_n,
        adjustments=adjustments,
    )
    return attach_album_art(sp, recs)


def recommendation_track_uris(recs_df: pd.DataFrame) -> list[str]:
    return [f"spotify:track:{sid}" for sid in recs_df["spotify_id"].dropna()]


def add_recommendations_to_spotify(recs_df, playlist_name="Recommended Songs", sp=None):
    sp = sp or get_spotify_client_or_raise()
    try:
        user_id = sp.me()["id"]
    except Exception as exc:
        raise classify_spotify_error(exc) from exc

    track_uris = recommendation_track_uris(recs_df)
    if not track_uris:
        raise NoRecommendationTracksError()

    try:
        return create_recommendation_playlist(sp, user_id, track_uris, name=playlist_name)
    except Exception as exc:
        raise classify_spotify_error(exc) from exc
