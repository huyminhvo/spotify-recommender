from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recommender.tuning import (
    WEIGHT_SEMANTICS,
    TuningConfig,
    tune_recommender_weights,
    write_tuning_result,
)
from utils.catalog_store import CatalogStore
from utils.terminal_progress import TerminalProgress

DEFAULT_OUTPUT_JSON = ROOT_DIR / "reports" / "weight_tuning.json"


def deployment_catalog_path(root_dir: Path = ROOT_DIR) -> Path:
    manifest = root_dir / "data" / "catalog" / "CURRENT"
    artifact_name = manifest.read_text(encoding="utf-8").strip()
    if not artifact_name or Path(artifact_name).name != artifact_name:
        raise ValueError(f"Invalid deployment catalog manifest: {manifest}")
    artifact = manifest.parent / artifact_name
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune audio-feature vector multipliers on a playlist-level tuning partition. "
            "The held-out test playlists are recorded but never evaluated by this command."
        ),
        epilog=(
            f"Weight semantics: {WEIGHT_SEMANTICS} After selecting weights, run the final "
            "benchmark exactly once on the JSON artifact's test_playlist_ids."
        ),
    )
    parser.add_argument(
        "--memberships-csv",
        required=True,
        help=(
            "Label-only membership CSV with playlist_id, source_spotify_id, and "
            "catalog_spotify_id columns."
        ),
    )
    parser.add_argument(
        "--catalog-parquet",
        help=(
            "Deployment catalog Parquet. Defaults to the immutable artifact named by "
            "data/catalog/CURRENT."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Reproducible tuning artifact path.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=8,
        help="Total candidates, including uniform and current hand-set defaults.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--min-playlists",
        type=int,
        default=10,
        help="Refuse tuning when fewer distinct playlists are available.",
    )
    parser.add_argument("--min-tuning-playlists", type=int, default=5)
    parser.add_argument("--min-test-playlists", type=int, default=2)
    parser.add_argument("--weight-min", type=float, default=0.25)
    parser.add_argument("--weight-max", type=float, default=4.0)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--min-popularity", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help=(
            "Optional CatalogStore sample limit for smoke runs. Omit it to use the "
            "same configured limit as deployment."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    catalog_path = Path(args.catalog_parquet) if args.catalog_parquet else deployment_catalog_path()
    catalog = CatalogStore(catalog_path, candidate_limit=args.candidate_limit)
    memberships = pd.read_csv(Path(args.memberships_csv))
    config = TuningConfig(
        num_trials=args.trials,
        test_fraction=args.test_fraction,
        num_splits=args.splits,
        top_k=args.top_k,
        seed_size=args.seed_size,
        random_state=args.random_state,
        min_playlists=args.min_playlists,
        min_tuning_playlists=args.min_tuning_playlists,
        min_test_playlists=args.min_test_playlists,
        weight_min=args.weight_min,
        weight_max=args.weight_max,
        bootstrap_samples=args.bootstrap_samples,
        min_popularity=args.min_popularity,
        pca_components=args.pca_components,
    )
    print(
        f"[tuning] starting {config.num_trials} trials x {config.num_splits} splits; "
        "progress is reported in approximately 1% increments.",
        flush=True,
    )
    result = tune_recommender_weights(
        catalog,
        memberships,
        config=config,
        progress_callback=TerminalProgress("tuning"),
    )
    output_path = write_tuning_result(result, args.output_json)

    selected = result.selected_trial
    print(f"Selected trial: {selected.candidate.name}")
    print(f"Mean NDCG@{config.top_k}: {selected.mean_ndcg_at_k:.6f}")
    print(f"Mean Recall@{config.top_k}: {selected.mean_recall_at_k:.6f}")
    print(f"Weight semantics: {WEIGHT_SEMANTICS}")
    print(f"Wrote tuning artifact: {output_path}")
    print(
        "Held-out test playlist IDs were not evaluated. Run the final benchmark once "
        "on partition.test_playlist_ids before reporting recommendation quality."
    )


if __name__ == "__main__":
    main()
