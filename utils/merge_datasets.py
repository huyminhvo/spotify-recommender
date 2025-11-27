from __future__ import annotations
import ast
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
import hashlib, json

import pandas as pd
from utils.matcher import canon_title, canon_artist_primary

# === Audio feature list ===
AUDIO_FEATURES = [
    "danceability","energy","valence","speechiness","acousticness",
    "instrumentalness","liveness","loudness","tempo","key","mode"
]

# === Column auto-mapper ===

def _auto_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Try to guess which columns in df map to the canonical schema.
    """
    lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lower: return lower[c]
        return None
    duration = next((c for c in df.columns if "duration" in c.lower()), None)
    year = next((c for c in df.columns if "year" in c.lower()), None)
    return {
        "id": pick("id","track_id","spotify_id","uri"),
        "name": pick("name","track_name","title"),
        "artists": pick("artists","artist","artist_name"),
        "duration": duration,
        "explicit": next((c for c in df.columns if "explicit" in c.lower()), None),
        "popularity": next((c for c in df.columns if "popularity" in c.lower()), None),
        "isrc": "isrc" if "isrc" in lower else None,
        "release_year": year,
        "album": pick("album","album_name"),
        # audio features
        "danceability": "danceability" if "danceability" in lower else None,
        "energy": "energy" if "energy" in lower else None,
        "valence": "valence" if "valence" in lower else None,
        "speechiness": "speechiness" if "speechiness" in lower else None,
        "acousticness": "acousticness" if "acousticness" in lower else None,
        "instrumentalness": "instrumentalness" if "instrumentalness" in lower else None,
        "liveness": "liveness" if "liveness" in lower else None,
        "loudness": "loudness" if "loudness" in lower else None,
        "tempo": "tempo" if "tempo" in lower else None,
        "key": "key" if "key" in lower else None,
        "mode": "mode" if "mode" in lower else None,
    }

# === Row normalization ===

def _normalize_row(r: pd.Series, colmap: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Normalize one raw row into the canonical schema + audio features.
    """

    # artists to list[str]
    artists_raw = r.get(colmap["artists"], "")
    if isinstance(artists_raw, str):
        txt = artists_raw.strip()
        if txt.startswith("["):
            try:
                artists_raw = ast.literal_eval(txt)
            except Exception:
                artists_raw = [a.strip() for a in txt.split(",")]
        else:
            artists_raw = [a.strip() for a in txt.split(",")]
    elif not isinstance(artists_raw, list) or artists_raw is None:
        artists_raw = []

    # id normalize
    raw_id = r.get(colmap["id"])
    sid = str(raw_id) if raw_id is not None and str(raw_id) != "" else None
    if sid and ":" in sid:
        sid = sid.split(":")[-1]

    # title canon
    title_raw = str(r.get(colmap["name"], "") or "")
    tc = canon_title(title_raw)
    title_canon = tc[0] if isinstance(tc, tuple) else tc

    # artist canon
    artist_primary = canon_artist_primary(artists_raw)

    # duration
    dur = r.get(colmap["duration"])
    try:
        duration_ms = int(dur) if pd.notnull(dur) else None
    except Exception:
        duration_ms = None

    # explicit
    explicit = None
    if colmap["explicit"] is not None:
        try:
            explicit = bool(r.get(colmap["explicit"]))
        except Exception:
            explicit = None

    # popularity
    popularity = None
    if colmap["popularity"] is not None and pd.notnull(r.get(colmap["popularity"])):
        try:
            popularity = int(r.get(colmap["popularity"]))
        except Exception:
            popularity = None

    # release year
    release_year = None
    if colmap["release_year"] is not None and pd.notnull(r.get(colmap["release_year"])):
        try:
            release_year = int(str(r.get(colmap["release_year"]))[:4])
        except Exception:
            release_year = None

    # isrc / album
    isrc = str(r.get(colmap["isrc"])) if colmap["isrc"] and pd.notnull(r.get(colmap["isrc"])) else None
    album = str(r.get(colmap["album"])) if colmap["album"] and pd.notnull(r.get(colmap["album"])) else None

    # audio features
    feats: Dict[str, Any] = {}
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
            if feat in {"danceability","energy","valence","speechiness","acousticness","instrumentalness","liveness"}:
                v = float(val)
                feats[feat] = max(0.0, min(1.0, v))  # clamp
            elif feat == "loudness":
                feats[feat] = float(val)
            elif feat == "tempo":
                v = float(val)
                feats[feat] = v if v > 0 else None
            elif feat in {"key","mode"}:
                feats[feat] = int(val)
            else:
                feats[feat] = float(val)
        except Exception:
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

def _merge_two_rows(x: Dict[str, Any], y: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(x)
    for k in ["spotify_id", "isrc"]:
        out[k] = out[k] or y.get(k)

    out["title_raw"] = _coalesce(out.get("title_raw"), y.get("title_raw"))
    out["artists_raw"] = _coalesce(out.get("artists_raw"), y.get("artists_raw"))
    out["album"] = _coalesce(out.get("album"), y.get("album"))

    # popularity: take max
    px, py = out.get("popularity"), y.get("popularity")
    out["popularity"] = max(px, py) if (px is not None and py is not None) else (px if px is not None else py)

    # explicit: True if any
    ex, ey = out.get("explicit"), y.get("explicit")
    out["explicit"] = bool(ex or ey) if (ex is not None or ey is not None) else None

    # release_year: prefer newer
    rx, ry = out.get("release_year"), y.get("release_year")
    if rx is None: out["release_year"] = ry
    elif ry is None: out["release_year"] = rx
    else: out["release_year"] = max(rx, ry)

    # duration: prefer larger if within 2s
    dx, dy = out.get("duration_ms"), y.get("duration_ms")
    if dx is None: out["duration_ms"] = dy
    elif dy is None: out["duration_ms"] = dx
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

def _dedupe_by_key(rows: List[Dict[str, Any]], key_fn) -> List[Dict[str, Any]]:
    buckets = defaultdict(list)
    for r in rows:
        k = key_fn(r)
        if k is not None:
            buckets[k].append(r)

    merged: List[Dict[str, Any]] = []
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

def merge_datasets(paths: List[str], conservative_duration_ms: int = 3000) -> pd.DataFrame:
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
    norm_rows: List[Dict[str, Any]] = []
    for p in paths:
        df = pd.read_csv(p)
        col = _auto_columns(df)
        for _, r in df.iterrows():
            norm_rows.append(_normalize_row(r, col))

    rows = _dedupe_by_key(norm_rows, key_fn=lambda r: r["spotify_id"])
    rows = _dedupe_by_key(rows, key_fn=lambda r: r["isrc"])

    def key_canon_dur(r):
        if not r["title_canon"] or not r["artist_primary_canon"] or r["duration_ms"] is None:
            return None
        bucket = int(round(r["duration_ms"] / conservative_duration_ms))
        return (r["title_canon"], r["artist_primary_canon"], bucket)

    rows = _dedupe_by_key(rows, key_fn=key_canon_dur)

    out_df = pd.DataFrame(rows, columns=[
        "spotify_id","title_raw","title_canon","artists_raw","artist_primary_canon",
        "duration_ms","explicit","popularity","release_year","isrc","album"
    ] + AUDIO_FEATURES)

    out_df = out_df.drop_duplicates(
        subset=["spotify_id","isrc","title_canon","artist_primary_canon","duration_ms"],
        keep="first"
    ).reset_index(drop=True)

    return out_df

# === Cache wrapper ===

def _fingerprint_inputs(paths: list[str]) -> str:
    """
    Stable hash of input files based on path, size, and mtime.
    """
    items = []
    for p in paths:
        stat = Path(p).stat()
        items.append({"path": str(Path(p).resolve()),
                      "size": stat.st_size,
                      "mtime": int(stat.st_mtime)})
    blob = json.dumps(sorted(items, key=lambda d: d["path"]), separators=(",",":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]

def get_merged_dataset(paths: list[str],
                       cache_dir: str = ".dataset_cache",
                       force_rebuild: bool = False) -> pd.DataFrame:
    """
    Get the merged dataset, using a cached Parquet file if available.
    Also persists a pickled indexes object for fast future loading.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fp = _fingerprint_inputs(paths)
    target = Path(cache_dir) / f"merged_{fp}.parquet"
    index_target = Path(cache_dir) / f"indexes_{fp}.pkl"

    if target.exists() and index_target.exists() and not force_rebuild:
        print(f"[cache] Using cached {target.name} and {index_target.name}")
        return pd.read_parquet(target)

    print("[cache] Rebuilding merged dataset + indexes…")
    df = merge_datasets(paths)
    df.to_parquet(target, index=False)

    # build and save indexes alongside parquet
    from utils.matcher import build_indexes
    indexes = build_indexes(df)
    with open(index_target, "wb") as f:
        import pickle
        pickle.dump(indexes, f)

    return df
