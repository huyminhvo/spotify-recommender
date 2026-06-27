from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.matcher import build_indexes
from utils.merge_datasets import get_merged_dataset
from utils.spotify_auth import get_spotify_client
from utils.spotify_integration import extract_playlist_id, fetch_playlist_profile


def default_catalog_paths(root_dir: Path = ROOT_DIR) -> list[str]:
    data_raw = root_dir / "data" / "raw"
    return [
        str(data_raw / "spotify_data.csv"),
        str(data_raw / "spotify_top_songs_audio_features.csv"),
        str(data_raw / "tracks_features.csv"),
    ]


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
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        line_count = sum(1 for _ in f)
    return max(line_count - 1, 0)


def count_raw_catalog_rows(paths: list[str]) -> int:
    return sum(count_csv_rows(path) for path in paths)


def duplicate_reduction_rate(raw_rows: int | None, merged_rows: int) -> float:
    if not raw_rows:
        return 0.0
    return max(0.0, 1.0 - (merged_rows / raw_rows))


def build_evaluation_dataset(
    sp,
    catalog_df: pd.DataFrame,
    playlist_inputs: list[str],
    min_matched_tracks: int = 10,
    raw_catalog_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = build_indexes(catalog_df)
    playlist_frames = []
    summary_rows = []
    merged_catalog_rows = len(catalog_df)
    reduction_rate = duplicate_reduction_rate(raw_catalog_rows, merged_catalog_rows)

    for playlist_input in playlist_inputs:
        playlist_id = extract_playlist_id(playlist_input)
        matched, stats = fetch_playlist_profile(
            sp,
            playlist_id,
            indexes,
            catalog_df,
            return_stats=True,
        )
        matched_count = len(matched)
        total_tracks = stats.get("total_tracks", matched_count)
        match_rate = matched_count / total_tracks if total_tracks else 0.0

        summary_rows.append(
            {
                "playlist_id": playlist_id,
                "total_tracks": total_tracks,
                "matched_tracks": matched_count,
                "match_rate": match_rate,
                "included": matched_count >= min_matched_tracks,
                "raw_catalog_rows": raw_catalog_rows or 0,
                "merged_catalog_rows": merged_catalog_rows,
                "duplicate_reduction_rate": reduction_rate,
            }
        )

        if matched_count < min_matched_tracks:
            print(
                f"[skip] {playlist_id}: matched {matched_count}/{total_tracks} tracks "
                f"(minimum is {min_matched_tracks})"
            )
            continue

        matched = matched.copy()
        matched.insert(0, "playlist_id", playlist_id)
        playlist_frames.append(matched)
        print(
            f"[include] {playlist_id}: matched {matched_count}/{total_tracks} tracks ({match_rate:.1%})"
        )

    if not playlist_frames:
        return pd.DataFrame(), pd.DataFrame(summary_rows)

    dataset = pd.concat(playlist_frames, ignore_index=True)
    dataset = dataset.drop_duplicates(subset=["playlist_id", "spotify_id"]).reset_index(drop=True)
    return dataset, pd.DataFrame(summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a playlist-labeled CSV for offline recommender evaluation."
    )
    parser.add_argument(
        "--playlist-url",
        action="append",
        default=[],
        help="Spotify playlist URL, URI, or raw ID. Can be passed multiple times.",
    )
    parser.add_argument(
        "--playlist-file",
        help="Text file with one Spotify playlist URL, URI, or raw ID per line. '#' comments are ignored.",
    )
    parser.add_argument(
        "--catalog-path",
        action="append",
        help="Raw catalog CSV path. Can be passed multiple times. Defaults to data/raw/*.csv inputs.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT_DIR / "data" / "examples" / "real_playlist_eval.csv"),
        help="Where to write the playlist-labeled evaluation CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional path for a per-playlist match summary CSV.",
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
        help="Rebuild the merged catalog cache before matching playlists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playlist_inputs = load_playlist_inputs(args.playlist_url, args.playlist_file)
    if not playlist_inputs:
        raise SystemExit("Provide at least one --playlist-url or --playlist-file.")

    catalog_paths = args.catalog_path or default_catalog_paths()
    catalog_df = get_merged_dataset(catalog_paths, force_rebuild=args.force_rebuild_catalog)
    raw_catalog_rows = count_raw_catalog_rows(catalog_paths)
    sp = get_spotify_client()

    dataset, summary = build_evaluation_dataset(
        sp=sp,
        catalog_df=catalog_df,
        playlist_inputs=playlist_inputs,
        min_matched_tracks=args.min_matched_tracks,
        raw_catalog_rows=raw_catalog_rows,
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
    print(
        f"[write] {len(dataset)} rows across {dataset['playlist_id'].nunique()} playlists -> {output_path}"
    )

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"[write] summary -> {summary_path}")


if __name__ == "__main__":
    main()
