"""Build the compact, immutable catalog consumed by the deployed web app."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import duckdb

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "catalog"

# Only columns used for matching, display, filtering, and recommendation scoring
# are shipped. Narrow numeric types and Zstandard substantially reduce the asset.
SELECT_COLUMNS = """
    CAST(spotify_id AS VARCHAR) AS spotify_id,
    CAST(title_raw AS VARCHAR) AS title_raw,
    CAST(title_canon AS VARCHAR) AS title_canon,
    artists_raw,
    CAST(artist_primary_canon AS VARCHAR) AS artist_primary_canon,
    CAST(duration_ms AS INTEGER) AS duration_ms,
    CAST(popularity AS FLOAT) AS popularity,
    CAST(release_year AS SMALLINT) AS release_year,
    CAST(danceability AS FLOAT) AS danceability,
    CAST(energy AS FLOAT) AS energy,
    CAST(valence AS FLOAT) AS valence,
    CAST(speechiness AS FLOAT) AS speechiness,
    CAST(acousticness AS FLOAT) AS acousticness,
    CAST(instrumentalness AS FLOAT) AS instrumentalness,
    CAST(liveness AS FLOAT) AS liveness,
    CAST(loudness AS FLOAT) AS loudness,
    CAST(tempo AS FLOAT) AS tempo
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_catalog(source: Path, output_dir: Path, version: str) -> Path:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not version.replace("-", "").replace("_", "").isalnum():
        raise ValueError("version may contain only letters, numbers, '-' and '_'")

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / f".catalog-{version}.tmp.parquet"
    source_sql = str(source).replace("'", "''")
    target_sql = str(temporary.resolve()).replace("'", "''")

    with duckdb.connect(":memory:") as connection:
        connection.execute(
            f"""
            COPY (
                SELECT {SELECT_COLUMNS}
                FROM read_parquet('{source_sql}')
            ) TO '{target_sql}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 122880)
            """
        )
        row_count = connection.execute(
            f"SELECT count(*) FROM read_parquet('{target_sql}')"
        ).fetchone()[0]
    if row_count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("refusing to publish an empty catalog")

    digest = sha256_file(temporary)
    artifact = output_dir / f"catalog-{version}-{digest[:16]}.parquet"
    temporary.replace(artifact)
    (output_dir / "CURRENT").write_text(f"{artifact.name}\n", encoding="utf-8")
    print(f"Built {artifact} ({row_count:,} rows, {artifact.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest}")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing merged catalog Parquet")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    build_catalog(args.source, args.output_dir, args.version)


if __name__ == "__main__":
    main()
