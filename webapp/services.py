from __future__ import annotations

import ast
import os
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
from utils.catalog_store import CatalogStore
from utils.merge_datasets import _fingerprint_inputs, get_merged_dataset
from utils.spotify_auth import get_public_spotify_client
from utils.spotify_integration import extract_playlist_id, fetch_playlist_profile
from utils.spotify_playlist import create_recommendation_playlist
from webapp.errors import (
    AppError,
    CatalogReadError,
    InvalidPlaylistURLError,
    MissingDatasetError,
    NoCatalogMatchesError,
    NoRecommendationTracksError,
    SpotifyAuthenticationError,
    classify_spotify_error,
)

SPOTIFY_TRACK_BATCH_SIZE = 50
CATALOG_MANIFEST_PATH = ROOT_DIR / "data" / "catalog" / "CURRENT"


@dataclass(frozen=True)
class CatalogBundle:
    paths: list[str]
    catalog: pd.DataFrame | CatalogStore
    indexes: dict | None = None


def format_artist_names(artists_raw) -> str:
    """Format canonical or serialized artist lists for display."""
    if artists_raw is None:
        return "Unknown artist"

    if isinstance(artists_raw, str):
        text = artists_raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                return format_artist_names(ast.literal_eval(text))
            except (SyntaxError, ValueError):
                pass
        return text or "Unknown artist"

    if hasattr(artists_raw, "tolist"):
        artists_raw = artists_raw.tolist()

    if isinstance(artists_raw, (list, tuple, set)):
        names = [str(artist).strip() for artist in artists_raw if str(artist).strip()]
        return ", ".join(names) or "Unknown artist"

    return str(artists_raw).strip() or "Unknown artist"


def default_catalog_paths(root_dir: Path = ROOT_DIR) -> list[str]:
    data_raw = root_dir / "data" / "raw"
    return [
        str(data_raw / "spotify_data.csv"),
        str(data_raw / "spotify_top_songs_audio_features.csv"),
        str(data_raw / "tracks_features.csv"),
    ]


def cache_dir(root_dir: Path = ROOT_DIR) -> Path:
    return root_dir / ".dataset_cache"


def deployment_catalog_path() -> Path:
    configured_path = os.getenv("CATALOG_PARQUET_PATH")
    if configured_path:
        return Path(configured_path).expanduser()

    try:
        artifact_name = CATALOG_MANIFEST_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise MissingDatasetError(
            "The deployment catalog manifest is missing. Run "
            "`python scripts/build_deployment_catalog.py` and commit the generated "
            "data/catalog artifact with Git LFS."
        ) from exc

    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise MissingDatasetError(f"Invalid catalog manifest: {CATALOG_MANIFEST_PATH}")
    return CATALOG_MANIFEST_PATH.parent / artifact_name


def load_catalog_bundle(catalog_paths: list[str] | None = None) -> CatalogBundle:
    if catalog_paths is None:
        parquet_path = deployment_catalog_path()
        try:
            return CatalogBundle(paths=[str(parquet_path)], catalog=CatalogStore(parquet_path))
        except FileNotFoundError as exc:
            raise MissingDatasetError(
                f"Deployment catalog is missing: {parquet_path}. Ensure Git LFS "
                "objects were fetched or set CATALOG_PARQUET_PATH."
            ) from exc

    # Explicit paths are a development/evaluation escape hatch. The deployed app
    # never reaches this raw-data merge path.
    paths = catalog_paths
    directory = cache_dir()
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise MissingDatasetError(f"Missing catalog files: {missing}")
    try:
        fingerprint = _fingerprint_inputs(paths)
        parquet_path = directory / f"merged_{fingerprint}.parquet"
        if not parquet_path.exists():
            # A first-time local build still needs the source DataFrames.
            get_merged_dataset(paths, cache_dir=str(directory))
        catalog = CatalogStore(parquet_path)
    except FileNotFoundError as exc:
        raise MissingDatasetError(str(exc)) from exc
    return CatalogBundle(paths=paths, catalog=catalog)


def match_playlist_tracks(sp, playlist_url: str, bundle: CatalogBundle) -> pd.DataFrame:
    try:
        playlist_id = extract_playlist_id(playlist_url)
    except ValueError as exc:
        raise InvalidPlaylistURLError(str(exc)) from exc

    try:
        if isinstance(bundle.catalog, CatalogStore):
            user_tracks = fetch_playlist_profile(sp, playlist_id, catalog_store=bundle.catalog)
        else:
            user_tracks = fetch_playlist_profile(sp, playlist_id, bundle.indexes, bundle.catalog)
    except Exception as exc:
        if "ZSTD Decompression failure" in str(exc):
            raise CatalogReadError(str(exc)) from exc
        raise classify_spotify_error(exc) from exc

    if user_tracks.empty:
        raise NoCatalogMatchesError()
    return user_tracks


def generate_recommendations(
    bundle: CatalogBundle,
    user_tracks_df: pd.DataFrame,
    top_n: int = 10,
    adjustments: dict[str, float] | None = None,
    exclude_spotify_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    return recommend_from_catalog(
        catalog=bundle.catalog,
        user_tracks_df=user_tracks_df,
        top_n=top_n,
        user_weights=DEFAULT_WEIGHTS,
        adjustments=adjustments,
        randomize_results=True,
        random_state=None,
        exclude_spotify_ids=exclude_spotify_ids,
    )


def fetch_album_art_urls(sp, spotify_ids: Iterable[str]) -> list[str | None]:
    spotify_ids = list(spotify_ids)
    art_urls: list[str | None] = []
    for spotify_id in spotify_ids:
        try:
            track = sp.track(spotify_id)
        except Exception:
            art_urls.append(None)
            continue

        images = (track or {}).get("album", {}).get("images", [])
        art_urls.append(
            images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
        )
    return art_urls


def attach_album_art(sp, recs: pd.DataFrame) -> pd.DataFrame:
    recs = recs.copy()
    recs["album_art_url"] = fetch_album_art_urls(sp, recs["spotify_id"])
    return recs.reset_index(drop=True)


def get_spotify_client_or_raise():
    try:
        return get_public_spotify_client()
    except AppError:
        raise
    except Exception as exc:
        raise SpotifyAuthenticationError(str(exc)) from exc


def get_recommendations(
    playlist_url: str,
    top_n: int = 10,
    adjustments: dict[str, float] | None = None,
    sp=None,
    public_sp=None,
    catalog_bundle: CatalogBundle | None = None,
    exclude_spotify_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    if sp is None:
        raise SpotifyAuthenticationError(
            "Connect Spotify to read a playlist you own or collaborate on."
        )
    public_sp = public_sp or get_spotify_client_or_raise()
    bundle = catalog_bundle or load_catalog_bundle()
    user_tracks = match_playlist_tracks(sp, playlist_url, bundle)
    recs = generate_recommendations(
        bundle,
        user_tracks,
        top_n=top_n,
        adjustments=adjustments,
        exclude_spotify_ids=exclude_spotify_ids,
    )
    return attach_album_art(public_sp, recs)


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
