import pandas as pd
import pytest

from scripts.build_deployment_catalog import build_catalog, sha256_file

CATALOG_ROW = {
    "spotify_id": "track-id",
    "title_raw": "Song",
    "title_canon": "song",
    "artists_raw": ["Artist"],
    "artist_primary_canon": "artist",
    "duration_ms": 180_000,
    "popularity": 50.0,
    "release_year": 2024,
    "danceability": 0.5,
    "energy": 0.6,
    "valence": 0.7,
    "speechiness": 0.1,
    "acousticness": 0.2,
    "instrumentalness": 0.0,
    "liveness": 0.1,
    "loudness": -8.0,
    "tempo": 120.0,
}


def test_build_catalog_publishes_content_addressed_artifact_and_manifest(tmp_path):
    source = tmp_path / "source.parquet"
    pd.DataFrame([CATALOG_ROW]).to_parquet(source, index=False)
    output_dir = tmp_path / "catalog"

    artifact = build_catalog(source, output_dir, "test-v1")

    assert artifact.is_file()
    assert sha256_file(artifact)[:16] in artifact.name
    assert (output_dir / "CURRENT").read_text(encoding="utf-8") == f"{artifact.name}\n"
    assert list(output_dir.glob("*.tmp*")) == []
    assert list(output_dir.glob(".catalog-*")) == []


def test_empty_catalog_failure_preserves_manifest_and_removes_temporary_files(tmp_path):
    source = tmp_path / "empty.parquet"
    pd.DataFrame(columns=CATALOG_ROW).to_parquet(source, index=False)
    output_dir = tmp_path / "catalog"
    output_dir.mkdir()
    manifest = output_dir / "CURRENT"
    manifest.write_text("existing.parquet\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty catalog"):
        build_catalog(source, output_dir, "test-v2")

    assert manifest.read_text(encoding="utf-8") == "existing.parquet\n"
    assert list(output_dir.glob("*.tmp*")) == []
    assert list(output_dir.glob(".catalog-*")) == []
