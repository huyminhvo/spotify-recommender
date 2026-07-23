from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from recommender.evaluate import (
    DEFAULT_STRATEGIES,
    EvaluationConfig,
    EvaluationResult,
    EvaluationStrategy,
    evaluate_benchmark,
    normalize_memberships,
)
from recommender.policy import DEPLOYED_POLICY
from scripts.build_evaluation_dataset import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_SUMMARY_CSV,
    deployment_catalog_path,
    parquet_row_count,
)
from utils.catalog_store import DEFAULT_CANDIDATE_LIMIT, CatalogStore
from utils.terminal_progress import TerminalProgress

DEFAULT_MEMBERSHIP_CSV = DEFAULT_OUTPUT_CSV
EXAMPLE_MEMBERSHIP_CSV = ROOT_DIR / "data" / "examples" / "real_playlist_membership.csv"
LEGACY_EVAL_CSV = ROOT_DIR / "data" / "examples" / "real_playlist_eval.csv"
DEFAULT_MATCH_SUMMARY_CSV = DEFAULT_SUMMARY_CSV
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "evaluation.md"
DEFAULT_RESULTS_JSON = ROOT_DIR / "reports" / "evaluation_results.json"
README_PATH = ROOT_DIR / "README.md"
README_RESULTS_START = "<!-- evaluation-results:start -->"
README_RESULTS_END = "<!-- evaluation-results:end -->"


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_membership_path() -> Path | None:
    if DEFAULT_MEMBERSHIP_CSV.exists():
        return DEFAULT_MEMBERSHIP_CSV
    if EXAMPLE_MEMBERSHIP_CSV.exists():
        return EXAMPLE_MEMBERSHIP_CSV
    if LEGACY_EVAL_CSV.exists():
        return LEGACY_EVAL_CSV
    return None


def load_memberships(
    path: str | Path,
    *,
    playlist_col: str = "playlist_id",
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(path)
    raw = pd.read_csv(path)
    modern_schema = {
        "source_spotify_id",
        "catalog_spotify_id",
    }.issubset(raw.columns)
    labels = normalize_memberships(raw, playlist_col=playlist_col)
    metadata = {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "schema": "membership-v2" if modern_schema else "legacy-matched-only",
        "preserves_unmatched": modern_schema,
        "rows": len(labels),
    }
    return labels, metadata


def load_playlist_id_filter(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        try:
            values = payload["partition"]["test_playlist_ids"]
        except (KeyError, TypeError) as exc:
            raise ValueError("JSON playlist partitions need partition.test_playlist_ids.") from exc
        return {str(value) for value in values}

    values = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return set(values)


def load_selected_weights(path: str | Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    weights = payload.get("selected_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Weight JSON needs a non-empty selected_weights object.")
    return {str(feature): float(value) for feature, value in weights.items()}


def strategies_with_selected_weights(
    weights: dict[str, float] | None,
) -> tuple[EvaluationStrategy, ...]:
    if not weights:
        return DEFAULT_STRATEGIES
    deterministic_policy = replace(
        DEPLOYED_POLICY,
        user_weights=weights,
        randomize_results=False,
        random_state=0,
    )
    deployed_policy = replace(DEPLOYED_POLICY, user_weights=weights)
    return (
        *DEFAULT_STRATEGIES,
        EvaluationStrategy(
            name="tuned_weighted_cosine_pca",
            policy=deterministic_policy,
            description="Selected weights evaluated on the untouched test partition.",
        ),
        EvaluationStrategy(
            name="tuned_deployed",
            policy=deployed_policy,
            description="Selected weights with deployed top-pool randomization.",
        ),
    )


def filter_memberships(
    memberships: pd.DataFrame,
    playlist_ids: Iterable[str] | None,
) -> pd.DataFrame:
    if playlist_ids is None:
        return memberships
    selected = {str(value) for value in playlist_ids}
    filtered = memberships[memberships["playlist_id"].isin(selected)].copy()
    missing = selected - set(filtered["playlist_id"])
    if missing:
        raise ValueError(f"Playlist IDs were not found in labels: {sorted(missing)}")
    return filtered.reset_index(drop=True)


def load_match_summary(
    path: str | Path | None,
    playlist_ids: Iterable[str] | None = None,
) -> dict[str, object] | None:
    if path is None:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    if playlist_ids is not None:
        selected = {str(playlist_id) for playlist_id in playlist_ids}
        summary = summary[summary["playlist_id"].astype(str).isin(selected)].copy()
    if summary.empty:
        return None

    total_col = (
        "total_unique_source_tracks" if "total_unique_source_tracks" in summary else "total_tracks"
    )
    matched_col = (
        "matched_unique_tracks" if "matched_unique_tracks" in summary else "matched_tracks"
    )
    total = int(pd.to_numeric(summary[total_col], errors="coerce").fillna(0).sum())
    matched = int(pd.to_numeric(summary[matched_col], errors="coerce").fillna(0).sum())
    return {
        "path": portable_path(summary_path),
        "sha256": sha256_file(summary_path),
        "num_playlists": int(summary["playlist_id"].nunique()),
        "total_source_tracks": total,
        "matched_tracks": matched,
        "match_rate": matched / total if total else 0.0,
    }


def _strategy_metadata(strategy: EvaluationStrategy) -> dict[str, object]:
    policy = strategy.policy
    return {
        "name": strategy.name,
        "description": strategy.description,
        "strategy": policy.strategy,
        "weighted": policy.user_weights is not None,
        "weights": dict(policy.user_weights or {}),
        "min_popularity": policy.min_popularity,
        "use_pca": policy.use_pca,
        "pca_components": policy.pca_components,
        "randomize_results": policy.randomize_results,
        "same_artist_exclusion": policy.same_artist_exclusion,
        "adjustments": dict(strategy.adjustments or {}),
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def build_results_payload(
    result: EvaluationResult,
    *,
    config: EvaluationConfig,
    catalog_metadata: dict[str, object],
    label_metadata: dict[str, object],
    match_summary: dict[str, object] | None,
    strategies: Iterable[EvaluationStrategy],
    input_artifacts: dict[str, object] | None = None,
) -> dict[str, object]:
    return _json_safe(
        {
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": (
                "benchmark" if result.audit.get("benchmark_ready") else "engineering-smoke-test"
            ),
            "catalog": catalog_metadata,
            "labels": label_metadata,
            "match_summary": match_summary,
            "input_artifacts": input_artifacts or {},
            "config": asdict(config),
            "strategies": [_strategy_metadata(strategy) for strategy in strategies],
            "audit": result.audit,
            "summary": result.summary.to_dict("records"),
            "skipped": result.skipped.to_dict("records"),
        }
    )


def write_results_json(payload: dict[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _format_number(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _format_percent(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def _format_interval(row: pd.Series, metric: str) -> str:
    low = row.get(f"{metric}_ci_low")
    high = row.get(f"{metric}_ci_high")
    if pd.isna(low) or pd.isna(high):
        return "—"
    return f"[{float(low):.3f}, {float(high):.3f}]"


def comparison_table_markdown(
    summary: pd.DataFrame,
    *,
    top_k: int,
    include_intervals: bool = True,
    confidence_level: float = 0.95,
) -> str:
    if summary.empty:
        return "_No evaluable playlist splits were available._"

    if include_intervals:
        confidence_percent = round(confidence_level * 100)
        header = (
            f"| Strategy | Recall@{top_k} | {confidence_percent}% CI | "
            f"NDCG@{top_k} | {confidence_percent}% CI | Hit rate | "
            "Retrieval ceiling | Retrievable recall | Steering target distance | "
            "Catalog coverage |"
        )
        separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = f"| Strategy | Recall@{top_k} | NDCG@{top_k} | Hit rate | Catalog coverage |"
        separator = "|---|---:|---:|---:|---:|"

    lines = [header, separator]
    for _, row in summary.iterrows():
        if include_intervals:
            lines.append(
                "| {strategy} | {recall} | {recall_ci} | {ndcg} | {ndcg_ci} | "
                "{hit_rate} | {ceiling} | {retrievable_recall} | "
                "{target_distance} | {coverage} |".format(
                    strategy=row["strategy"],
                    recall=_format_number(row["recall_at_k"]),
                    recall_ci=_format_interval(row, "recall_at_k"),
                    ndcg=_format_number(row["ndcg_at_k"]),
                    ndcg_ci=_format_interval(row, "ndcg_at_k"),
                    hit_rate=_format_percent(row["hit_rate_at_k"]),
                    ceiling=_format_percent(row["candidate_recall_ceiling"]),
                    retrievable_recall=_format_number(row["retrievable_recall_at_k"]),
                    target_distance=_format_number(row.get("steering_target_distance")),
                    coverage=_format_percent(row["catalog_coverage"], digits=4),
                )
            )
        else:
            lines.append(
                "| {strategy} | {recall} | {ndcg} | {hit_rate} | {coverage} |".format(
                    strategy=row["strategy"],
                    recall=_format_number(row["recall_at_k"]),
                    ndcg=_format_number(row["ndcg_at_k"]),
                    hit_rate=_format_percent(row["hit_rate_at_k"]),
                    coverage=_format_percent(row["catalog_coverage"], digits=4),
                )
            )
    return "\n".join(lines)


def render_report(
    result: EvaluationResult,
    *,
    config: EvaluationConfig,
    catalog_metadata: dict[str, object],
    label_metadata: dict[str, object],
    match_summary: dict[str, object] | None,
    strategies: Iterable[EvaluationStrategy],
) -> str:
    audit = result.audit
    benchmark_ready = bool(audit.get("benchmark_ready"))
    status = (
        "Benchmark result"
        if benchmark_ready
        else "Engineering smoke test — not evidence of recommendation quality"
    )
    generated_date = datetime.now(UTC).date().isoformat()
    confidence_percent = round(config.confidence_level * 100)

    lines = [
        "# Offline evaluation report",
        "",
        f"Generated: {generated_date}",
        "",
        f"**Status: {status}.**",
        "",
    ]
    for warning in audit.get("warnings", []):
        lines.append(f"- {warning}")
    if not label_metadata.get("preserves_unmatched"):
        lines.append(
            "- This run used the legacy matched-only label file. Ranking metrics "
            f"are conditional on its {int(audit['num_memberships']):,} "
            "catalog-matched tracks; rebuild labels with "
            "`scripts/build_evaluation_dataset.py` to include unmatched positives."
        )
    if int(audit.get("num_playlists", 0)) < 2:
        lines.append(
            "- With one playlist, the playlist-clustered bootstrap interval "
            "collapses to the point estimate and is not an uncertainty estimate."
        )

    lines.extend(
        [
            "",
            "## Data and policy",
            "",
            f"- Item catalog: `{catalog_metadata['artifact']}` "
            f"({int(catalog_metadata['rows']):,} unique rows).",
            f"- Popularity-eligible catalog: "
            f"{int(catalog_metadata['eligible_rows']):,} tracks; bounded candidate "
            f"sample: {int(catalog_metadata['candidate_limit']):,}.",
            f"- Membership labels: {int(audit['num_memberships']):,} rows across "
            f"{int(audit['num_playlists']):,} "
            f"{'playlist' if int(audit['num_playlists']) == 1 else 'playlists'}.",
            f"- Repeated splits: {config.num_splits} per playlist, "
            f"{config.seed_size} matched seeds per split.",
            f"- Intervals: {confidence_percent}% playlist-clustered bootstrap, "
            f"{config.bootstrap_samples:,} resamples after averaging splits within "
            "each playlist.",
            "- The `deployed` row uses the same weighted cosine, PCA(5), "
            "minimum-popularity filter, bounded catalog sample, and weighted "
            "top-pool randomization as the first web-app request. A fixed random "
            "seed is supplied only for reproducibility.",
            "- Session history is not simulated; this is a first-request "
            "playlist-continuation benchmark.",
        ]
    )
    if match_summary:
        lines.append(
            f"- Source matching: {int(match_summary['matched_tracks']):,}/"
            f"{int(match_summary['total_source_tracks']):,} tracks "
            f"({_format_percent(match_summary['match_rate'])})."
        )

    lines.extend(
        [
            "",
            "## Strategy comparison",
            "",
            comparison_table_markdown(
                result.summary,
                top_k=config.top_k,
                include_intervals=True,
                confidence_level=config.confidence_level,
            ),
            "",
            "Recall uses every non-seed playlist item as relevant. "
            "`Retrieval ceiling` is the fraction of those positives present in "
            "the actual filtered, bounded candidate pool. `Retrievable recall` "
            "conditions on that pool to separate ranking from retrieval loss. "
            "Catalog coverage is "
            "computed once per strategy as distinct recommended items divided by "
            "the full popularity-eligible catalog, so it can distinguish "
            "strategies across repeated requests.",
            "",
            "## Ablations",
            "",
            "| Strategy | Weights | PCA | Randomized | Steering |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for strategy in strategies:
        policy = strategy.policy
        steering = (
            ", ".join(
                f"{feature}={value:+.2f}" for feature, value in (strategy.adjustments or {}).items()
            )
            or "off"
        )
        lines.append(
            f"| {strategy.name} | {'yes' if policy.user_weights else 'no'} | "
            f"{'yes' if policy.use_pca else 'no'} | "
            f"{'yes' if policy.randomize_results else 'no'} | {steering} |"
        )

    lines.extend(
        [
            "",
            "The fixed steering row is not treated as a user-preference label. "
            "It is paired with the unsteered PCA row to measure target-distance "
            "movement and any relevance cost; playlist NDCG alone cannot validate "
            "steering quality.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if benchmark_ready:
        lines.append(
            "The playlist-count gate is satisfied, but results should still be "
            "reviewed for match rate, overlap, and playlist diversity before being "
            "used as a quality claim."
        )
    else:
        lines.append(
            "This report validates the evaluation machinery only. Collect at least "
            f"{config.min_playlists_for_claim} diverse, accessible playlists, "
            "rebuild the membership labels so unmatched tracks are retained, and "
            "rerun before placing quality numbers on a resume."
        )
    lines.extend(
        [
            "",
            "The benchmark is a playlist-continuation proxy rather than a direct "
            "measure of user satisfaction. Weight selection must occur on a "
            "playlist-level tuning partition, with the reported test playlists "
            "kept untouched.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: str, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def update_readme_results(
    readme_path: str | Path,
    *,
    summary: pd.DataFrame,
    top_k: int,
    report_path: str | Path,
    benchmark_ready: bool,
    num_playlists: int,
) -> None:
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    start_index = text.find(README_RESULTS_START)
    end_index = text.find(README_RESULTS_END)
    if start_index < 0 or end_index < 0 or end_index < start_index:
        raise ValueError("README evaluation result markers are missing or invalid.")

    try:
        relative_report = Path(report_path).resolve().relative_to(ROOT_DIR.resolve())
        report_link = relative_report.as_posix()
    except ValueError:
        report_link = str(report_path)
    status = (
        f"Benchmark over {num_playlists} playlists."
        if benchmark_ready
        else (
            f"Smoke test over {num_playlists} "
            f"{'playlist' if num_playlists == 1 else 'playlists'}; "
            "do not treat as a quality claim."
        )
    )
    replacement = "\n".join(
        [
            README_RESULTS_START,
            "",
            status,
            "",
            comparison_table_markdown(
                summary,
                top_k=top_k,
                include_intervals=False,
            ),
            "",
            f"[Full methodology and confidence intervals]({report_link})",
            "",
            README_RESULTS_END,
        ]
    )
    new_text = text[:start_index] + replacement + text[end_index + len(README_RESULTS_END) :]
    path.write_text(new_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_labels = default_membership_path()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate separate playlist-membership labels against the immutable "
            "deployment catalog and generate a reproducible report."
        )
    )
    parser.add_argument(
        "--labels-csv",
        default=str(default_labels) if default_labels else None,
        help=(
            "Label-only membership CSV. Legacy matched-track CSVs are accepted but "
            "reported as matched-only smoke-test inputs."
        ),
    )
    parser.add_argument(
        "--catalog-parquet",
        default=None,
        help="Catalog Parquet path. Defaults to the artifact named by data/catalog/CURRENT.",
    )
    parser.add_argument(
        "--match-summary-csv",
        default=str(DEFAULT_MATCH_SUMMARY_CSV),
        help="Optional builder summary used to report source-to-catalog match rate.",
    )
    parser.add_argument(
        "--playlist-col",
        default="playlist_id",
        help="Membership column containing playlist IDs.",
    )
    parser.add_argument(
        "--playlist-id-file",
        default=None,
        help=(
            "Optional text list of playlist IDs, or a tuning JSON artifact whose "
            "partition.test_playlist_ids should be evaluated."
        ),
    )
    parser.add_argument(
        "--weights-json",
        default=None,
        help=(
            "Optional tuning JSON containing selected_weights. When omitted, a JSON "
            "--playlist-id-file is also used as the weight source."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-size", type=int, default=5)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--min-popularity", type=int, default=20)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        help="Bounded catalog sample size; production default is 100000.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-playlists-for-claim", type=int, default=50)
    parser.add_argument(
        "--report-path",
        default=str(DEFAULT_REPORT_PATH),
        help="Generated Markdown report path.",
    )
    parser.add_argument(
        "--results-json",
        default=str(DEFAULT_RESULTS_JSON),
        help="Generated machine-readable summary path.",
    )
    parser.add_argument(
        "--per-split-csv",
        default=None,
        help="Optional raw split-level metrics CSV.",
    )
    parser.add_argument(
        "--recommendations-csv",
        default=None,
        help="Optional raw recommendation exposure CSV.",
    )
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Replace the README comparison table between generated markers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels_csv is None:
        raise SystemExit(
            "No membership labels were found. Pass --labels-csv after running "
            "scripts/build_evaluation_dataset.py."
        )
    catalog_path = Path(args.catalog_parquet) if args.catalog_parquet else deployment_catalog_path()
    catalog = CatalogStore(catalog_path, candidate_limit=args.candidate_limit)
    memberships, label_metadata = load_memberships(
        args.labels_csv,
        playlist_col=args.playlist_col,
    )
    playlist_filter = load_playlist_id_filter(args.playlist_id_file)
    memberships = filter_memberships(memberships, playlist_filter)
    weight_path = args.weights_json
    if (
        weight_path is None
        and args.playlist_id_file
        and Path(args.playlist_id_file).suffix.lower() == ".json"
    ):
        weight_path = args.playlist_id_file
    strategies = strategies_with_selected_weights(load_selected_weights(weight_path))
    label_metadata["rows"] = len(memberships)
    label_metadata["playlists"] = int(memberships["playlist_id"].nunique())

    config = EvaluationConfig(
        top_k=args.top_k,
        seed_size=args.seed_size,
        num_splits=args.splits,
        random_state=args.random_state,
        min_popularity=args.min_popularity,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        min_playlists_for_claim=args.min_playlists_for_claim,
    )
    eligible_rows = catalog.count_candidates(min_popularity=config.min_popularity)
    catalog_metadata = {
        "path": portable_path(catalog_path),
        "artifact": catalog_path.name,
        "rows": parquet_row_count(catalog_path),
        "eligible_rows": eligible_rows,
        "candidate_limit": catalog.candidate_limit,
    }
    match_summary = load_match_summary(args.match_summary_csv, playlist_filter)
    input_artifacts: dict[str, object] = {}
    if args.playlist_id_file:
        input_artifacts["playlist_filter"] = {
            "path": portable_path(args.playlist_id_file),
            "sha256": sha256_file(args.playlist_id_file),
            "selected_playlist_ids": sorted(playlist_filter or ()),
        }
    if weight_path:
        input_artifacts["weights"] = {
            "path": portable_path(weight_path),
            "sha256": sha256_file(weight_path),
        }

    print(
        f"[evaluation] starting {label_metadata['playlists']} playlists x "
        f"{config.num_splits} splits x {len(strategies)} strategies; "
        "progress is reported in approximately 1% increments.",
        flush=True,
    )
    result = evaluate_benchmark(
        catalog,
        memberships,
        config=config,
        strategies=strategies,
        catalog_size=eligible_rows,
        progress_callback=TerminalProgress("evaluation"),
    )
    payload = build_results_payload(
        result,
        config=config,
        catalog_metadata=catalog_metadata,
        label_metadata=label_metadata,
        match_summary=match_summary,
        strategies=strategies,
        input_artifacts=input_artifacts,
    )
    report = render_report(
        result,
        config=config,
        catalog_metadata=catalog_metadata,
        label_metadata=label_metadata,
        match_summary=match_summary,
        strategies=strategies,
    )
    report_path = write_report(report, args.report_path)
    results_path = write_results_json(payload, args.results_json)

    if args.per_split_csv:
        per_split_path = Path(args.per_split_csv)
        per_split_path.parent.mkdir(parents=True, exist_ok=True)
        result.per_split.to_csv(per_split_path, index=False)
    if args.recommendations_csv:
        recommendations_path = Path(args.recommendations_csv)
        recommendations_path.parent.mkdir(parents=True, exist_ok=True)
        result.recommendations.to_csv(recommendations_path, index=False)
    if args.update_readme:
        update_readme_results(
            README_PATH,
            summary=result.summary,
            top_k=config.top_k,
            report_path=report_path,
            benchmark_ready=bool(result.audit["benchmark_ready"]),
            num_playlists=int(result.audit["num_playlists"]),
        )

    if result.summary.empty:
        print("No playlist splits were evaluable.")
    else:
        columns = [
            "strategy",
            "num_playlists",
            "recall_at_k",
            "ndcg_at_k",
            "hit_rate_at_k",
            "candidate_recall_ceiling",
            "catalog_coverage",
        ]
        print(
            result.summary[columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )
    print(f"[write] report -> {report_path}")
    print(f"[write] results -> {results_path}")


if __name__ == "__main__":
    main()
