from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import spotipy

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.catalog_store import CatalogStore
from utils.matcher import build_indexes
from utils.merge_datasets import get_merged_dataset
from utils.spotify_auth import (
    DEFAULT_EVALUATION_TOKEN_CACHE,
    get_cached_user_spotify_client,
    get_spotify_client,
)
from utils.spotify_integration import extract_playlist_id, fetch_playlist_membership

DEFAULT_OUTPUT_CSV = ROOT_DIR / "data" / "local" / "playlist_membership.csv"
DEFAULT_SUMMARY_CSV = ROOT_DIR / "data" / "local" / "playlist_match_summary.csv"


def deployment_catalog_path(root_dir: Path = ROOT_DIR) -> Path:
    """Resolve the immutable catalog artifact named by data/catalog/CURRENT."""
    manifest_path = root_dir / "data" / "catalog" / "CURRENT"
    artifact_name = manifest_path.read_text(encoding="utf-8").strip()
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ValueError(f"Invalid deployment catalog manifest: {manifest_path}")

    artifact_path = manifest_path.parent / artifact_name
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    return artifact_path


def parquet_row_count(path: str | Path) -> int:
    """Read a Parquet row count from metadata without loading the catalog."""
    return pq.ParquetFile(path).metadata.num_rows


def get_playlist_spotify_client(
    access_token_env: str = "SPOTIFY_USER_ACCESS_TOKEN",
    token_cache: str | Path = DEFAULT_EVALUATION_TOKEN_CACHE,
):
    """Use cached user OAuth, with raw-token and app-only compatibility fallbacks."""
    access_token = os.getenv(access_token_env)
    if access_token:
        return spotipy.Spotify(auth=access_token)
    try:
        return get_cached_user_spotify_client(cache_path=token_cache)
    except ValueError as exc:
        print(f"[warning] {exc}")
    print(
        "[warning] using app-only Spotify credentials. Development Mode "
        "playlist items require user authorization."
    )
    return get_spotify_client()


def load_playlist_inputs(
    playlist_urls: list[str] | None = None,
    playlist_file: str | None = None,
) -> list[str]:
    inputs = list(playlist_urls or [])
    if playlist_file:
        path = Path(playlist_file)
        file_inputs = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        inputs.extend(file_inputs)

    seen = set()
    deduped = []
    for item in inputs:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def count_csv_rows(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        line_count = sum(1 for _ in stream)
    return max(line_count - 1, 0)


def count_raw_catalog_rows(paths: list[str]) -> int:
    return sum(count_csv_rows(path) for path in paths)


def duplicate_reduction_rate(raw_rows: int | None, merged_rows: int) -> float:
    if not raw_rows:
        return 0.0
    return max(0.0, 1.0 - (merged_rows / raw_rows))


def build_evaluation_dataset(
    sp,
    catalog_df: pd.DataFrame | None = None,
    playlist_inputs: list[str] | None = None,
    min_matched_tracks: int = 10,
    raw_catalog_rows: int | None = None,
    catalog_store=None,
    catalog_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if catalog_store is None and catalog_df is None:
        raise ValueError("Provide catalog_store or catalog_df.")
    if not playlist_inputs:
        return pd.DataFrame(), pd.DataFrame()

    indexes = build_indexes(catalog_df) if catalog_store is None else None
    playlist_frames = []
    summary_rows = []
    merged_catalog_rows = catalog_rows
    if merged_catalog_rows is None:
        if catalog_df is not None:
            merged_catalog_rows = len(catalog_df)
        elif getattr(catalog_store, "path", None):
            merged_catalog_rows = parquet_row_count(catalog_store.path)
        else:
            merged_catalog_rows = 0
    reduction_rate = duplicate_reduction_rate(raw_catalog_rows, merged_catalog_rows)

    for playlist_input in playlist_inputs:
        playlist_id = extract_playlist_id(playlist_input)
        membership, stats = fetch_playlist_membership(
            sp,
            playlist_id,
            indexes=indexes,
            catalog_df=catalog_df,
            catalog_store=catalog_store,
            return_stats=True,
        )
        total_source_tracks = int(stats.get("total_source_tracks", len(membership)))
        total_unique_source_tracks = int(stats.get("total_unique_source_tracks", len(membership)))
        duplicate_tracks_removed = int(
            stats.get(
                "duplicate_tracks_removed",
                total_source_tracks - total_unique_source_tracks,
            )
        )
        matched_unique_tracks = int(
            stats.get(
                "matched_unique_tracks",
                membership["matched"].sum() if "matched" in membership else 0,
            )
        )
        match_rate = (
            matched_unique_tracks / total_unique_source_tracks
            if total_unique_source_tracks
            else 0.0
        )
        included = matched_unique_tracks >= min_matched_tracks

        summary_rows.append(
            {
                "playlist_id": playlist_id,
                "total_source_tracks": total_source_tracks,
                "total_unique_source_tracks": total_unique_source_tracks,
                "duplicate_tracks_removed": duplicate_tracks_removed,
                "matched_unique_tracks": matched_unique_tracks,
                "match_rate": match_rate,
                "included": included,
                # Backward-compatible aliases use unique-track counts.
                "total_tracks": total_unique_source_tracks,
                "matched_tracks": matched_unique_tracks,
                "raw_catalog_rows": raw_catalog_rows or 0,
                "merged_catalog_rows": merged_catalog_rows,
                "duplicate_reduction_rate": reduction_rate,
            }
        )

        if not included:
            print(
                f"[skip] {playlist_id}: matched "
                f"{matched_unique_tracks}/{total_unique_source_tracks} unique tracks "
                f"(minimum is {min_matched_tracks})"
            )
            continue

        playlist_frames.append(membership)
        print(
            f"[include] {playlist_id}: matched "
            f"{matched_unique_tracks}/{total_unique_source_tracks} unique tracks "
            f"({match_rate:.1%})"
        )

    if not playlist_frames:
        return pd.DataFrame(), pd.DataFrame(summary_rows)

    dataset = pd.concat(playlist_frames, ignore_index=True)
    dataset = dataset.sort_values(["playlist_id", "position"], kind="stable").reset_index(drop=True)
    return dataset, pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a label-only playlist membership CSV for offline recommender evaluation."
        )
    )
    parser.add_argument(
        "--playlist-url",
        action="append",
        default=[],
        help="Spotify playlist URL, URI, or raw ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--playlist-file",
        help=(
            "Text file with one Spotify playlist URL, URI, or raw ID per line. "
            "'#' comments are ignored."
        ),
    )
    catalog_group = parser.add_mutually_exclusive_group()
    catalog_group.add_argument(
        "--catalog-parquet",
        help=(
            "Deployment catalog Parquet path. Defaults to the artifact named by "
            "data/catalog/CURRENT."
        ),
    )
    catalog_group.add_argument(
        "--raw-catalog-path",
        "--catalog-path",
        dest="raw_catalog_path",
        action="append",
        help=(
            "Explicit raw catalog CSV path for the legacy in-memory matching path. "
            "Can be passed multiple times; --catalog-path is retained as an alias."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Where to write the label-only playlist membership CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(DEFAULT_SUMMARY_CSV),
        help="Where to write the per-playlist match summary CSV.",
    )
    parser.add_argument(
        "--min-matched-tracks",
        type=int,
        default=10,
        help="Skip playlists with fewer matched catalog tracks.",
    )
    parser.add_argument(
        "--force-rebuild-catalog",
        action="store_true",
        help="Rebuild the merged catalog cache when using --raw-catalog-path.",
    )
    parser.add_argument(
        "--spotify-access-token-env",
        default="SPOTIFY_USER_ACCESS_TOKEN",
        help=(
            "Environment variable containing a user access token for owned or "
            "collaborative playlists. Falls back to app-only credentials when unset."
        ),
    )
    parser.add_argument(
        "--spotify-token-cache",
        default=str(DEFAULT_EVALUATION_TOKEN_CACHE),
        help=(
            "Refreshable user-token cache created by scripts/authorize_spotify.py. "
            "Ignored when --spotify-access-token-env is set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playlist_inputs = load_playlist_inputs(args.playlist_url, args.playlist_file)
    if not playlist_inputs:
        raise SystemExit("Provide at least one --playlist-url or --playlist-file.")

    catalog_df = None
    catalog_store = None
    catalog_rows = None
    raw_catalog_rows = None
    if args.raw_catalog_path:
        catalog_paths = args.raw_catalog_path
        catalog_df = get_merged_dataset(
            catalog_paths,
            force_rebuild=args.force_rebuild_catalog,
        )
        raw_catalog_rows = count_raw_catalog_rows(catalog_paths)
        catalog_rows = len(catalog_df)
    else:
        catalog_path = (
            Path(args.catalog_parquet) if args.catalog_parquet else deployment_catalog_path()
        )
        catalog_store = CatalogStore(catalog_path)
        catalog_rows = parquet_row_count(catalog_path)

    sp = get_playlist_spotify_client(
        args.spotify_access_token_env,
        args.spotify_token_cache,
    )

    dataset, summary = build_evaluation_dataset(
        sp=sp,
        catalog_df=catalog_df,
        catalog_store=catalog_store,
        playlist_inputs=playlist_inputs,
        min_matched_tracks=args.min_matched_tracks,
        raw_catalog_rows=raw_catalog_rows,
        catalog_rows=catalog_rows,
    )
    if dataset.empty:
        if args.summary_csv and not summary.empty:
            summary_path = Path(args.summary_csv)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
        raise SystemExit("No playlists met the minimum matched-track threshold.")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    playlist_count = dataset["playlist_id"].nunique()
    print(f"[write] {len(dataset)} rows across {playlist_count} playlists -> {output_path}")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"[write] summary -> {summary_path}")


if __name__ == "__main__":
    main()
