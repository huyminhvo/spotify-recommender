from __future__ import annotations

import ast
import hashlib
import json
import logging
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from utils.matcher import canon_artist_primary, canon_title

logger = logging.getLogger(__name__)
MERGE_SCHEMA_VERSION = 3
ARTIST_SCAN_CHUNK_SIZE = 100_000

# === Audio feature list ===
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "loudness",
    "tempo",
    "key",
    "mode",
]

# === Column auto-mapper ===


def _auto_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Try to guess which columns in df map to the canonical schema.
    """
    lower = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None

    duration = next((c for c in df.columns if "duration" in c.lower()), None)
    year = next((c for c in df.columns if "year" in c.lower()), None)
    return {
        "id": pick("id", "track_id", "spotify_id", "uri"),
        "name": pick("name", "track_name", "title"),
        "artists": pick("artists", "artist_names", "artist", "artist_name"),
        "duration": duration,
        "explicit": next((c for c in df.columns if "explicit" in c.lower()), None),
        "popularity": next((c for c in df.columns if "popularity" in c.lower()), None),
        "isrc": pick("isrc"),
        "release_year": year,
        "album": pick("album", "album_name"),
        # audio features
        "danceability": pick("danceability"),
        "energy": pick("energy"),
        "valence": pick("valence"),
        "speechiness": pick("speechiness"),
        "acousticness": pick("acousticness"),
        "instrumentalness": pick("instrumentalness"),
        "liveness": pick("liveness"),
        "loudness": pick("loudness"),
        "tempo": pick("tempo"),
        "key": pick("key"),
        "mode": pick("mode"),
    }


# === Row normalization ===


def _artist_tokens(value: str) -> tuple[str, ...]:
    return tuple(part.strip().casefold() for part in value.split(",") if part.strip())


def _parse_artists(
    value: Any,
    source_column: str | None,
    known_comma_artists: dict[tuple[str, ...], str] | None = None,
) -> list[str]:
    """Normalize singular, serialized-list, and comma-delimited artist fields."""
    if isinstance(value, (list, tuple)):
        return [str(artist).strip() for artist in value if str(artist).strip()]
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(artist).strip() for artist in parsed if str(artist).strip()]

    column_name = (source_column or "").lower()
    if column_name == "artist_names":
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) < 2:
            return parts

        # This source stores collaborators as a comma-delimited string, which is
        # ambiguous for names such as "Tyler, The Creator". Prefer the longest
        # artist names observed in singular/list-valued source columns, then treat
        # all remaining segments as collaborators.
        known = known_comma_artists or {}
        artists: list[str] = []
        position = 0
        while position < len(parts):
            for end in range(len(parts), position + 1, -1):
                known_artist = known.get(tuple(part.casefold() for part in parts[position:end]))
                if known_artist is not None:
                    artists.append(known_artist)
                    position = end
                    break
            else:
                artists.append(parts[position])
                position += 1
        return artists
    return [text]


def _known_comma_artists(paths: list[str]) -> dict[tuple[str, ...], str]:
    """Index comma-bearing names from unambiguous artist columns."""
    known: dict[tuple[str, ...], str] = {}
    for path in paths:
        columns = pd.read_csv(path, nrows=0).columns
        artist_column = _auto_columns(pd.DataFrame(columns=columns))["artists"]
        if artist_column is None or artist_column.lower() == "artist_names":
            continue

        chunks = pd.read_csv(
            path,
            usecols=[artist_column],
            chunksize=ARTIST_SCAN_CHUNK_SIZE,
        )
        for chunk in chunks:
            for value in chunk[artist_column].dropna().drop_duplicates():
                for artist in _parse_artists(value, artist_column):
                    tokens = _artist_tokens(artist)
                    if len(tokens) > 1:
                        known.setdefault(tokens, artist)
    return known


def _parse_bool(value: Any) -> bool | None:
    """Parse common boolean encodings without treating the string 'False' as true."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _normalize_row(
    r: pd.Series,
    colmap: dict[str, str | None],
    known_comma_artists: dict[tuple[str, ...], str] | None = None,
) -> dict[str, Any]:
    """
    Normalize one raw row into the canonical schema + audio features.
    """

    # artists to list[str]
    artists_raw = _parse_artists(
        r.get(colmap["artists"]),
        colmap["artists"],
        known_comma_artists,
    )

    # id normalize
    raw_id = r.get(colmap["id"])
    sid = str(raw_id).strip() if raw_id is not None and pd.notna(raw_id) else None
    sid = sid or None
    if sid and ":" in sid:
        sid = sid.split(":")[-1]

    # title canon
    raw_title = r.get(colmap["name"], "")
    title_raw = "" if raw_title is None or pd.isna(raw_title) else str(raw_title).strip()
    tc = canon_title(title_raw)
    title_canon = tc[0] if isinstance(tc, tuple) else tc

    # artist canon
    artist_primary = canon_artist_primary(artists_raw)

    # duration
    dur = r.get(colmap["duration"])
    try:
        duration_ms = int(dur) if pd.notnull(dur) else None
    except (TypeError, ValueError, OverflowError):
        duration_ms = None

    # explicit
    explicit = _parse_bool(r.get(colmap["explicit"])) if colmap["explicit"] else None

    # popularity
    popularity = None
    if colmap["popularity"] is not None and pd.notnull(r.get(colmap["popularity"])):
        try:
            popularity = int(r.get(colmap["popularity"]))
        except (TypeError, ValueError, OverflowError):
            popularity = None

    # release year
    release_year = None
    if colmap["release_year"] is not None and pd.notnull(r.get(colmap["release_year"])):
        try:
            release_year = int(str(r.get(colmap["release_year"]))[:4])
        except (TypeError, ValueError, OverflowError):
            release_year = None

    # isrc / album
    isrc = (
        str(r.get(colmap["isrc"])) if colmap["isrc"] and pd.notnull(r.get(colmap["isrc"])) else None
    )
    album = (
        str(r.get(colmap["album"]))
        if colmap["album"] and pd.notnull(r.get(colmap["album"]))
        else None
    )

    # audio features
    feats: dict[str, Any] = {}
    for feat in AUDIO_FEATURES:
        col = colmap.get(feat)
        if not col:
            feats[feat] = None
            continue
        val = r.get(col)
        if pd.isnull(val):
            feats[feat] = None
            continue
        try:
            if feat in {
                "danceability",
                "energy",
                "valence",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
            }:
                v = float(val)
                feats[feat] = max(0.0, min(1.0, v))  # clamp
            elif feat == "loudness":
                feats[feat] = float(val)
            elif feat == "tempo":
                v = float(val)
                feats[feat] = v if v > 0 else None
            elif feat in {"key", "mode"}:
                feats[feat] = int(val)
            else:
                feats[feat] = float(val)
        except (TypeError, ValueError, OverflowError):
            feats[feat] = None

    return {
        "spotify_id": sid,
        "title_raw": title_raw,
        "title_canon": title_canon,
        "artists_raw": artists_raw,
        "artist_primary_canon": artist_primary,
        "duration_ms": duration_ms,
        "explicit": explicit,
        "popularity": popularity,
        "release_year": release_year,
        "isrc": isrc,
        "album": album,
        **feats,
    }


# === Field-wise merge ===


def _coalesce(a, b):
    if a in (None, "", [], {}):
        return b
    if b in (None, "", [], {}):
        return a
    if isinstance(a, str) and isinstance(b, str):
        return a if len(a) >= len(b) else b
    if isinstance(a, list) and isinstance(b, list):
        return a if len(a) >= len(b) else b
    return a


def _merge_two_rows(x: dict[str, Any], y: dict[str, Any]) -> dict[str, Any]:
    out = dict(x)
    for k in ["spotify_id", "isrc"]:
        out[k] = out.get(k) or y.get(k)

    out["title_raw"] = _coalesce(out.get("title_raw"), y.get("title_raw"))
    out["artists_raw"] = _coalesce(out.get("artists_raw"), y.get("artists_raw"))
    out["album"] = _coalesce(out.get("album"), y.get("album"))
    # Canonical fields must describe the metadata selected above, not whichever
    # input row happened to initialize the merge bucket.
    out["title_canon"] = canon_title(out.get("title_raw", ""))
    out["artist_primary_canon"] = canon_artist_primary(out.get("artists_raw", []))

    # popularity: take max
    px, py = out.get("popularity"), y.get("popularity")
    out["popularity"] = (
        max(px, py) if (px is not None and py is not None) else (px if px is not None else py)
    )

    # explicit: True if any
    ex, ey = out.get("explicit"), y.get("explicit")
    out["explicit"] = bool(ex or ey) if (ex is not None or ey is not None) else None

    # release_year: prefer newer
    rx, ry = out.get("release_year"), y.get("release_year")
    if rx is None:
        out["release_year"] = ry
    elif ry is None:
        out["release_year"] = rx
    else:
        out["release_year"] = max(rx, ry)

    # duration: prefer larger if within 2s
    dx, dy = out.get("duration_ms"), y.get("duration_ms")
    if dx is None:
        out["duration_ms"] = dy
    elif dy is None:
        out["duration_ms"] = dx
    else:
        if abs(dx - dy) <= 2000:
            out["duration_ms"] = max(dx, dy)
        else:
            out["duration_ms"] = dx

    # audio features: keep existing if present, else take from y
    for feat in AUDIO_FEATURES:
        if out.get(feat) is None and y.get(feat) is not None:
            out[feat] = y.get(feat)

    return out


# === Dedupe passes ===


def _dedupe_by_key(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    buckets = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is not None:
            buckets[k].append(r)

    merged: list[dict[str, Any]] = []
    seen = set()
    for _, items in buckets.items():
        base = items[0]
        for other in items[1:]:
            base = _merge_two_rows(base, other)
        merged.append(base)
        seen.update(id(x) for x in items)

    for r in rows:
        if id(r) not in seen:
            merged.append(r)
    return merged


# === Main merge ===


def merge_datasets(paths: list[str], conservative_duration_ms: int = 3000) -> pd.DataFrame:
    """
    Load CSV datasets, normalize rows, and dedupe in three passes:
      1) by Spotify ID
      2) by ISRC
      3) by (title_canon, artist_primary_canon, duration bucket)

    Returns
    -------
    DataFrame
        A merged dataset with canonical metadata and audio features.
    """
    if conservative_duration_ms <= 0:
        raise ValueError("conservative_duration_ms must be greater than zero.")

    known_comma_artists = _known_comma_artists(paths)
    norm_rows: list[dict[str, Any]] = []
    for p in paths:
        df = pd.read_csv(p)
        col = _auto_columns(df)
        for _, r in df.iterrows():
            norm_rows.append(_normalize_row(r, col, known_comma_artists))

    rows = _dedupe_by_key(norm_rows, key_fn=lambda r: r["spotify_id"])
    rows = _dedupe_by_key(rows, key_fn=lambda r: r["isrc"])

    def key_canon_dur(r):
        if not r["title_canon"] or not r["artist_primary_canon"] or r["duration_ms"] is None:
            return None
        bucket = round(r["duration_ms"] / conservative_duration_ms)
        return (r["title_canon"], r["artist_primary_canon"], bucket)

    rows = _dedupe_by_key(rows, key_fn=key_canon_dur)

    out_df = pd.DataFrame(
        rows,
        columns=[
            "spotify_id",
            "title_raw",
            "title_canon",
            "artists_raw",
            "artist_primary_canon",
            "duration_ms",
            "explicit",
            "popularity",
            "release_year",
            "isrc",
            "album",
            *AUDIO_FEATURES,
        ],
    )

    return out_df.drop_duplicates(
        subset=["spotify_id", "isrc", "title_canon", "artist_primary_canon", "duration_ms"],
        keep="first",
    ).reset_index(drop=True)


# === Cache wrapper ===


def _fingerprint_inputs(paths: list[str]) -> str:
    """
    Stable hash of input files based on path, size, and mtime.
    """
    items = []
    for p in paths:
        stat = Path(p).stat()
        items.append(
            {
                "path": str(Path(p).resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    fingerprint_payload = {
        "merge_schema_version": MERGE_SCHEMA_VERSION,
        "inputs": sorted(items, key=lambda item: item["path"]),
    }
    blob = json.dumps(fingerprint_payload, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def get_merged_dataset(
    paths: list[str], cache_dir: str = ".dataset_cache", force_rebuild: bool = False
) -> pd.DataFrame:
    """
    Get the merged dataset, using a cached Parquet file if available.
    The web app queries this Parquet cache directly through DuckDB.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fp = _fingerprint_inputs(paths)
    target = Path(cache_dir) / f"merged_{fp}.parquet"

    if target.exists() and not force_rebuild:
        logger.info("Using cached merged dataset %s", target.name)
        return pd.read_parquet(target)

    logger.info("Rebuilding merged dataset cache")
    df = merge_datasets(paths)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}-",
            suffix=".tmp.parquet",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        df.to_parquet(temporary_path, index=False)
        temporary_path.replace(target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return df
