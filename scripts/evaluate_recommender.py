from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recommender.evaluate import DEFAULT_STRATEGIES, EvaluationConfig, evaluate_catalog_playlists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommender strategies on playlist holdouts.")
    parser.add_argument("--catalog-csv", required=True, help="CSV with catalog rows and a playlist id column.")
    parser.add_argument("--playlist-col", default="playlist_id", help="Column that identifies playlist membership.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--holdout-size", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--min-popularity", type=int, default=None)
    parser.add_argument("--same-artist-exclusion", action="store_true")
    parser.add_argument("--use-pca", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog_path = Path(args.catalog_csv)
    catalog = pd.read_csv(catalog_path)
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
