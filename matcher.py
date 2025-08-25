import ast
import re
import unicodedata
from collections import defaultdict

# === Canonicalization ===

def normalize_ascii(s: str) -> str:
    """Fold accents and strip to plain ASCII."""
    s = str(s or "")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def canon_title(title: str) -> str:
    """Canonicalize a title: strip accents, lowercase, remove variant tags, normalize spacing."""
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
    """Canonicalize the first artist in a list/string."""
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

# Variant tags to deprioritize in tie-breaking
VARIANT_TAGS = ["live", "remix", "instrumental", "clean", "explicit",
                "karaoke", "cover", "demo", "edit"]

# === Index Building ===

def build_indexes(df):
    """
    Build lookup dicts for fast matching.
    Uses the normalized columns (title_canon, artist_primary_canon)
    if they exist in the DataFrame, otherwise recomputes.
    """
    by_id, by_key, by_artist = {}, defaultdict(list), defaultdict(list)

    for _, row in df.iterrows():
        sid = row.get("spotify_id")
        if sid:
            by_id[sid] = row

        title_canon = row.get("title_canon") or canon_title(row.get("title_raw", ""))
        artist_canon = row.get("artist_primary_canon") or canon_artist_primary(row.get("artists_raw", []))

        if title_canon and artist_canon:
            by_key[(title_canon, artist_canon)].append(row)

        if artist_canon:
            by_artist[artist_canon].append(row)

    return {"by_id": by_id, "by_key": by_key, "by_artist": by_artist}

# === Match Resolution ===

def match_track(track, indexes, duration_tol=2000):
    """Try to resolve a Spotify API track dict to a dataset row."""
    tid = track.get("id")
    if tid and tid in indexes["by_id"]:
        return indexes["by_id"][tid]

    title_canon = canon_title(track.get("name", ""))
    artists = track.get("artists", [])
    artist_names = [a["name"] for a in artists] if artists else []
    artist_canon = canon_artist_primary(artist_names)

    candidates = indexes["by_key"].get((title_canon, artist_canon), [])
    dur = track.get("duration_ms")

    if dur and candidates:
        candidates = [
            c for c in candidates
            if c.get("duration_ms") and abs(c["duration_ms"] - dur) <= duration_tol
        ]

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    return _choose_best(candidates)

def _choose_best(candidates):
    """Prefer canonical versions over variants, then highest popularity & year."""
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
