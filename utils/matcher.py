import ast
import re
import unicodedata
from collections import defaultdict
import pandas as pd

# === Canonicalization ===

def normalize_ascii(s: str) -> str:
    """
    Fold accents and strip to plain ASCII.
    """
    s = str(s or "")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def canon_title(title: str) -> str:
    """
    Canonicalize a title: strip accents, lowercase, remove variant tags, normalize spacing.
    """
    t = normalize_ascii(title).strip().lower()
    STRIP_KEYWORDS = [
        "clean", "explicit", "radio edit", "remaster", "remastered",
        "instrumental", "acoustic", "live", "deluxe", "anniversary edition",
        "club mix", "extended mix"
    ]
    def contains_kw(seg: str) -> bool:
        seg = seg.lower()
        return any(re.search(rf"\b{re.escape(kw)}\b", seg) for kw in STRIP_KEYWORDS)

    # strip trailing bracket segments with keywords
    while True:
        m = re.search(r"(.*?)([\(\[\{]([^()\[\]{}]+)[\)\]\}])\s*$", t)
        if not m:
            break
        if contains_kw(m.group(3)):
            t = m.group(1).rstrip()
        else:
            break

    # strip trailing dash suffix with keywords
    m = re.search(r"^(.*?)(\s*-\s*(.+))$", t)
    if m and contains_kw(m.group(3)):
        t = m.group(1).rstrip()

    # normalize spaces/punct
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def canon_artist_primary(artists_raw) -> str:
    """
    Canonicalize the first artist in a list/string.
    """
    if isinstance(artists_raw, list) and artists_raw:
        primary = artists_raw[0]
    elif isinstance(artists_raw, str):
        txt = artists_raw.strip()
        if txt.startswith("["):
            try:
                parsed = ast.literal_eval(txt)
                primary = parsed[0] if parsed else ""
            except Exception:
                primary = txt.split(",")[0]
        else:
            primary = txt.split(",")[0]
    else:
        primary = ""
    primary = normalize_ascii(primary).strip().lower()
    primary = re.sub(r"\s+", " ", primary)
    return primary

# variant tags to deprioritize in tie-breaking
VARIANT_TAGS = ["live", "remix", "instrumental", "clean", "explicit",
                "karaoke", "cover", "demo", "edit"]

# === Index building ===

def build_indexes(df: pd.DataFrame):
    """
    Build lookup dicts for fast matching, storing only row indices
    (to keep pickled index size small).
    """
    by_id, by_key, by_artist = {}, defaultdict(list), defaultdict(list)

    for i, row in df.iterrows():
        sid = row.get("spotify_id")
        if sid:
            by_id[sid] = i  # store row index, not the row itself

        title_canon = row.get("title_canon") or canon_title(row.get("title_raw", ""))
        artist_canon = row.get("artist_primary_canon") or canon_artist_primary(row.get("artists_raw", []))

        if title_canon and artist_canon:
            by_key[(title_canon, artist_canon)].append(i)

        if artist_canon:
            by_artist[artist_canon].append(i)

    return {"by_id": by_id, "by_key": by_key, "by_artist": by_artist}


# === Match resolution ===

def match_track(track, indexes, df, duration_tol=2000):
    """
    Resolve a Spotify API track dict to a row in df using prebuilt indexes.
    """
    tid = track.get("id")
    if tid and tid in indexes["by_id"]:
        return df.iloc[indexes["by_id"][tid]].to_dict()

    title_canon = canon_title(track.get("name", ""))
    artists = track.get("artists", [])
    artist_names = [a["name"] for a in artists] if artists else []
    artist_canon = canon_artist_primary(artist_names)

    candidates_idx = indexes["by_key"].get((title_canon, artist_canon), [])
    dur = track.get("duration_ms")

    if dur and candidates_idx:
        candidates_idx = [
            i for i in candidates_idx
            if df.at[i, "duration_ms"] and abs(df.at[i, "duration_ms"] - dur) <= duration_tol
        ]

    if not candidates_idx:
        return None
    if len(candidates_idx) == 1:
        return df.iloc[candidates_idx[0]].to_dict()

    candidates = [df.iloc[i].to_dict() for i in candidates_idx]
    return _choose_best(candidates)

def _choose_best(candidates):
    """
    Prefer canonical versions over variants, then highest popularity & year.
    """
    def variant_score(row):
        name = (row.get("title_raw") or "").lower()
        for tag in VARIANT_TAGS:
            if tag in name:
                return 1  # variant
        return 0  # canonical

    sorted_cands = sorted(
        candidates,
        key=lambda r: (
            variant_score(r),
            -(r.get("popularity") or 0),
            -(r.get("release_year") or 0),
        )
    )
    return sorted_cands[0]
