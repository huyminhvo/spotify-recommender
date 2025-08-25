#!/usr/bin/env python3
"""
overlap_check.py — Compare overlap between two Spotify datasets (1.2M vs 1M, etc.)

Usage:
  python overlap_check.py --left /path/to/1p2m_tracks_features.csv --right /path/to/1m_spotify_data.csv \
      [--left-id id] [--right-id id] [--left-name name] [--right-name track_name] \
      [--left-artists artists] [--right-artists artist_names] \
      [--left-duration duration_ms] [--right-duration duration_ms] \
      [--write-csv]

What it does:
- Loads the minimal columns needed, with dtype=str where possible to avoid parsing surprises.
- Normalizes Spotify IDs (supports raw 22-char IDs, URIs like "spotify:track:ID", and URLs).
- Computes overlap by ID (fast path). If an ID column is missing in either file, falls back to a robust
  composite key: (normalized track_name, normalized primary_artist, rounded duration_seconds).
- Prints counts, overlap %, and Jaccard similarity. Optionally writes CSVs of only-in-left/right.

Notes:
- For very large CSVs, this script only reads the minimal columns to keep memory reasonable.
- If your "duration" is in seconds (not ms), the script will autodetect by magnitude and convert.
- Normalization strips diacritics/punctuation and collapses whitespace. It also removes common suffixes
  like " - remaster", " - radio edit", " (feat. ...)", etc., to improve matching across editions.
"""

import argparse
import os
import re
import sys
import unicodedata
from typing import Optional, Tuple, Set

import pandas as pd

# ---------- Utilities ----------

SPOTIFY_ID_RE = re.compile(r'(?i)(?:spotify:track:|https?://open\.spotify\.com/track/)?([A-Za-z0-9]{22})')

def extract_spotify_id(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    m = SPOTIFY_ID_RE.search(s)
    if m:
        return m.group(1)
    # Some dumps may already store id as 22-char plain
    if len(s) == 22 and s.isalnum():
        return s
    return None

def find_first(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def strip_accents_punct(text: str) -> str:
    # Remove diacritics and punctuation; keep alphanumerics and spaces
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r'[^0-9A-Za-z]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def normalize_title(raw: str) -> str:
    s = str(raw).lower().strip()
    # Remove common remix/edit qualifiers & featured segments
    # e.g., "Song - Remastered 2009", "Song (feat. X)", "Song - Radio Edit"
    s = re.sub(r'\s*-\s*(remastered.*|remaster.*|radio edit.*|single version.*|album version.*)$', '', s)
    s = re.sub(r'\s*\(feat\.?.*?\)', '', s)
    s = re.sub(r'\s*-\s*feat\.?.*$', '', s)
    return strip_accents_punct(s)

def normalize_artists(raw: str) -> str:
    s = str(raw).lower().strip()
    # Many dumps store artists as "['a','b']" or "a; b" or "a, b"
    # Extract the first/primary artist as tie-breaker for joining
    # 1) Try Python-list-like pattern
    m = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
    if m:
        first = next((a or b for (a, b) in m if (a or b)), None)
        if first:
            return strip_accents_punct(first)
    # 2) Split on common delimiters ; , & and take first
    first = re.split(r'\s*[;,/&]\s*', s)[0]
    # Remove trailing "feat. ..." if present
    first = re.sub(r'\s*\bfeat\.?.*$', '', first)
    return strip_accents_punct(first)

def coerce_duration_ms(series: pd.Series) -> pd.Series:
    # Try to coerce duration to integer ms; if values look like seconds (< 10000), assume seconds and convert to ms
    s = pd.to_numeric(series, errors='coerce')
    # Heuristic: if median is < 10000, it's likely seconds
    med = s.median(skipna=True)
    if pd.notna(med) and med < 10000:
        s = (s * 1000).round()
    return s.astype('Int64')

def build_fallback_key(df: pd.DataFrame, name_col: str, artists_col: str, duration_col: str) -> pd.Series:
    title = df[name_col].astype(str).map(normalize_title)
    artist = df[artists_col].astype(str).map(normalize_artists)
    duration_ms = coerce_duration_ms(df[duration_col])
    # Round to nearest second for stability
    dur_s = (duration_ms / 1000.0).round().astype('Int64')
    return title + '::' + artist + '::' + dur_s.astype(str)

def load_min_columns(path: str, id_col: Optional[str], name_col: Optional[str], artists_col: Optional[str], duration_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[str], Optional[str], Optional[str], Optional[str]]:
    # Read minimal columns present
    usecols = [c for c in [id_col, name_col, artists_col, duration_col] if c]
    # If any are None, read header first to auto-detect
    header = pd.read_csv(path, nrows=0)
    if id_col is None:
        id_col = find_first(header, ['id', 'track_id', 'spotify_id', 'trackid', 'uri', 'track_uri'])
    if name_col is None:
        name_col = find_first(header, ['name', 'track_name', 'song', 'title'])
    if artists_col is None:
        artists_col = find_first(header, ['artists', 'artist_names', 'artist', 'artists_name'])
    if duration_col is None:
        duration_col = find_first(header, ['duration_ms', 'duration', 'length', 'track_duration'])
    usecols = list({c for c in [id_col, name_col, artists_col, duration_col] if c})
    df = pd.read_csv(path, usecols=usecols, dtype={c: 'string' for c in usecols}, low_memory=False)
    return df, id_col, name_col, artists_col, duration_col

def summarize_sets(A: Set[str], B: Set[str], labelA: str, labelB: str):
    inter = A & B
    onlyA = A - B
    onlyB = B - A
    # Jaccard similarity
    jac = len(inter) / len(A | B) if (A or B) else 0.0
    print(f"Summary — unique IDs/keys")
    print(f"  {labelA}: {len(A):,}")
    print(f"  {labelB}: {len(B):,}")
    print(f"  Intersection: {len(inter):,}")
    print(f"  Only in {labelA}: {len(onlyA):,}")
    print(f"  Only in {labelB}: {len(onlyB):,}")
    pctA = (len(inter) / len(A) * 100) if A else 0.0
    pctB = (len(inter) / len(B) * 100) if B else 0.0
    print(f"  Overlap as % of {labelA}: {pctA:.2f}%")
    print(f"  Overlap as % of {labelB}: {pctB:.2f}%")
    print(f"  Jaccard similarity: {jac:.4f}")
    return inter, onlyA, onlyB, jac

def main():
    ap = argparse.ArgumentParser(description="Compute overlap between two Spotify datasets by ID (or fallback key).")
    ap.add_argument("--left", required=True, help="Path to left CSV (e.g., 1.2M tracks_features.csv)")
    ap.add_argument("--right", required=True, help="Path to right CSV (e.g., 1M spotify_data.csv)")
    ap.add_argument("--left-id", dest="left_id", default=None, help="ID column name for left (default: auto)")
    ap.add_argument("--right-id", dest="right_id", default=None, help="ID column name for right (default: auto)")
    ap.add_argument("--left-name", dest="left_name", default=None, help="Track title column for left (default: auto)")
    ap.add_argument("--right-name", dest="right_name", default=None, help="Track title column for right (default: auto)")
    ap.add_argument("--left-artists", dest="left_artists", default=None, help="Artists column for left (default: auto)")
    ap.add_argument("--right-artists", dest="right_artists", default=None, help="Artists column for right (default: auto)")
    ap.add_argument("--left-duration", dest="left_duration", default=None, help="Duration column for left (default: auto)")
    ap.add_argument("--right-duration", dest="right_duration", default=None, help="Duration column for right (default: auto)")
    ap.add_argument("--write-csv", action="store_true", help="Write CSVs listing only-in-left/right and intersection")
    ap.add_argument("--outdir", default="overlap_outputs", help="Directory for outputs when --write-csv is on")
    args = ap.parse_args()

    # Load minimal columns for both datasets
    left_df, L_id, L_name, L_art, L_dur = load_min_columns(args.left, args.left_id, args.left_name, args.left_artists, args.left_duration)
    right_df, R_id, R_name, R_art, R_dur = load_min_columns(args.right, args.right_id, args.right_name, args.right_artists, args.right_duration)

    labelL = os.path.basename(args.left)
    labelR = os.path.basename(args.right)

    # ---------- Try ID-based path ----------
    left_ids = None
    right_ids = None
    if L_id and R_id:
        left_ids = left_df[L_id].map(extract_spotify_id)
        right_ids = right_df[R_id].map(extract_spotify_id)
        left_ids = set(left_ids.dropna().unique())
        right_ids = set(right_ids.dropna().unique())

    if left_ids is not None and right_ids is not None and left_ids and right_ids:
        print("Matching by Spotify track ID")
        inter, onlyL, onlyR, jac = summarize_sets(left_ids, right_ids, labelL, labelR)
        if args.write_csv:
            os.makedirs(args.outdir, exist_ok=True)
            pd.Series(sorted(inter)).to_csv(os.path.join(args.outdir, "intersection_ids.csv"), index=False, header=["id"])
            pd.Series(sorted(onlyL)).to_csv(os.path.join(args.outdir, "only_in_left_ids.csv"), index=False, header=["id"])
            pd.Series(sorted(onlyR)).to_csv(os.path.join(args.outdir, "only_in_right_ids.csv"), index=False, header=["id"])
        return

    # ---------- Fallback: composite key ----------
    missing = []
    for side, nm, ar, du in (("left", L_name, L_art, L_dur), ("right", R_name, R_art, R_dur)):
        if nm is None: missing.append(f"{side}: track name")
        if ar is None: missing.append(f"{side}: artists")
        if du is None: missing.append(f"{side}: duration")
    if missing:
        print("ID columns not available for both datasets, and missing columns prevent fallback matching.")
        for m in missing:
            print(" -", m)
        print("Provide the appropriate column names via CLI flags (e.g., --left-name track_name --right-name name).")
        sys.exit(2)

    print("Matching by composite key: (normalized title, primary artist, rounded duration_s)")
    left_keys = build_fallback_key(left_df, L_name, L_art, L_dur)
    right_keys = build_fallback_key(right_df, R_name, R_art, R_dur)
    left_set = set(left_keys.dropna().unique())
    right_set = set(right_keys.dropna().unique())

    inter, onlyL, onlyR, jac = summarize_sets(left_set, right_set, labelL, labelR)

    if args.write_csv:
        os.makedirs(args.outdir, exist_ok=True)
        pd.Series(sorted(inter)).to_csv(os.path.join(args.outdir, "intersection_keys.csv"), index=False, header=["key"])
        pd.Series(sorted(onlyL)).to_csv(os.path.join(args.outdir, "only_in_left_keys.csv"), index=False, header=["key"])
        pd.Series(sorted(onlyR)).to_csv(os.path.join(args.outdir, "only_in_right_keys.csv"), index=False, header=["key"])

if __name__ == "__main__":
    main()
