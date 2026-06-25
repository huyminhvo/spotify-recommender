from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recommender.evaluate import DEFAULT_STRATEGIES, EvaluationConfig, evaluate_catalog_playlists
from scripts.build_evaluation_dataset import (
    build_evaluation_dataset,
    count_raw_catalog_rows,
    default_catalog_paths,
    load_playlist_inputs,
)
from utils.merge_datasets import get_merged_dataset
from utils.spotify_auth import get_spotify_client


DEFAULT_PLAYLIST_FILE = ROOT_DIR / "data" / "examples" / "playlists.txt"
DEFAULT_EVAL_CSV = ROOT_DIR / "data" / "examples" / "real_playlist_eval.csv"
DEFAULT_SUMMARY_CSV = ROOT_DIR / "data" / "examples" / "real_playlist_eval_summary.csv"


def print_match_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    cols = [
        "playlist_id",
        "total_tracks",
        "matched_tracks",
        "match_rate",
        "raw_catalog_rows",
        "merged_catalog_rows",
        "duplicate_reduction_rate",
    ]
    available_cols = [col for col in cols if col in summary.columns]
    if not available_cols:
        return
    print("\nData quality summary:")
    print(summary[available_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate recommender strategies on playlist holdouts. Provide either "
            "a prebuilt playlist-labeled --catalog-csv, or Spotify playlists via "
            "--playlist-url/--playlist-file to build the evaluation dataset first."
        )
    )
    parser.add_argument(
        "--catalog-csv",
        help="Prebuilt playlist-labeled CSV. Used directly when no Spotify playlist inputs are provided.",
    )
    parser.add_argument(
        "--playlist-url",
        action="append",
        default=[],
        help="Spotify playlist URL, URI, or raw ID to fetch and evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--playlist-file",
        default=None,
        help="Text file with one Spotify playlist URL, URI, or raw ID per line. '#' comments are ignored.",
    )
    parser.add_argument(
        "--raw-catalog-path",
        action="append",
        help="Raw catalog CSV path for playlist matching. Can be passed multiple times. Defaults to data/raw/*.csv inputs.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_EVAL_CSV),
        help="Where to save a dataset built from Spotify playlists.",
    )
    parser.add_argument(
        "--summary-csv",
        default=str(DEFAULT_SUMMARY_CSV),
        help="Optional path for a per-playlist match summary CSV when building from Spotify playlists.",
    )
    parser.add_argument(
        "--min-matched-tracks",
        type=int,
        default=10,
        help="Skip Spotify playlists with fewer matched catalog tracks.",
    )
    parser.add_argument(
        "--force-rebuild-catalog",
        action="store_true",
        help="Rebuild the merged raw catalog cache before matching Spotify playlists.",
    )
    parser.add_argument("--playlist-col", default="playlist_id", help="Column that identifies playlist membership.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--holdout-size", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--min-popularity", type=int, default=None)
    parser.add_argument("--same-artist-exclusion", action="store_true")
    parser.add_argument("--use-pca", action="store_true")
    return parser.parse_args()


def load_or_build_catalog(args: argparse.Namespace) -> pd.DataFrame:
    playlist_file = args.playlist_file
    output_path = Path(args.output_csv)

    if not args.catalog_csv and not args.playlist_url and playlist_file is None:
        if output_path.exists():
            print(f"[cache] Using saved evaluation dataset {output_path}")
            summary_path = Path(args.summary_csv) if args.summary_csv else DEFAULT_SUMMARY_CSV
            if summary_path.exists():
                print_match_summary(pd.read_csv(summary_path))
            return pd.read_csv(output_path)
        if DEFAULT_PLAYLIST_FILE.exists():
            playlist_file = str(DEFAULT_PLAYLIST_FILE)

    playlist_inputs = load_playlist_inputs(args.playlist_url, playlist_file)
    if not playlist_inputs:
        if not args.catalog_csv:
            raise SystemExit(
                "No evaluation input found. Add playlist URLs to "
                f"{DEFAULT_PLAYLIST_FILE}, or pass --catalog-csv/--playlist-url/--playlist-file."
            )
        return pd.read_csv(Path(args.catalog_csv))

    raw_catalog_paths = args.raw_catalog_path or default_catalog_paths()
    raw_catalog = get_merged_dataset(
        raw_catalog_paths,
        force_rebuild=args.force_rebuild_catalog,
    )
    raw_catalog_rows = count_raw_catalog_rows(raw_catalog_paths)
    sp = get_spotify_client()
    dataset, summary = build_evaluation_dataset(
        sp=sp,
        catalog_df=raw_catalog,
        playlist_inputs=playlist_inputs,
        min_matched_tracks=args.min_matched_tracks,
        raw_catalog_rows=raw_catalog_rows,
    )

    if args.summary_csv and not summary.empty:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"[write] summary -> {summary_path}")
    print_match_summary(summary)

    if dataset.empty:
        raise SystemExit("No playlists met the minimum matched-track threshold.")

    if args.playlist_col != "playlist_id":
        dataset = dataset.rename(columns={"playlist_id": args.playlist_col})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"[write] {len(dataset)} rows across {dataset[args.playlist_col].nunique()} playlists -> {output_path}")
    return dataset


def main() -> None:
    args = parse_args()
    catalog = load_or_build_catalog(args)
    config = EvaluationConfig(
        top_k=args.top_k,
        seed_size=args.seed_size,
        holdout_size=args.holdout_size,
        random_state=args.random_state,
        min_popularity=args.min_popularity,
        same_artist_exclusion=args.same_artist_exclusion,
        use_pca=args.use_pca,
    )
    results = evaluate_catalog_playlists(
        catalog,
        playlist_col=args.playlist_col,
        config=config,
        strategies=DEFAULT_STRATEGIES,
    )
    if results.empty:
        print("No playlists had enough unique tracks for evaluation.")
        return
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
