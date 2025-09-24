"""
schema.py
---------
Defines constants and schema information for features used in the recommender.
"""

from typing import List

# Ordered list of numerical features for modeling (your requested 10)
# Note: Spotify datasets typically use 'duration_ms'. If your merged catalog
# has 'duration' in seconds/minutes instead, preprocess.py will auto-detect it.
FEATURE_COLS: List[str] = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
    "tempo",
    "loudness",
    "duration_ms",   # prefer ms; preprocess handles 'duration' fallback
]

# Columns that may need special transforms
# - tempo: light log1p compression (skewed, positive)
# - duration_ms: convert to minutes then log1p (broad range, skewed)
SPECIAL_COLS = {
    "tempo": "log1p",        # assumes tempo >= 0
    "duration_ms": "log1p_min",  # convert ms -> minutes, then log1p
}

# Reasonable numeric bounds for sanity clipping (very mild; optional)
# We keep these wide to avoid distorting data but kill obvious outliers.
CLIP_BOUNDS = {
    "tempo": (0.0, 300.0),        # bpm
    "loudness": (-60.0, 0.0),     # dBFS typically ~[-60, 0]
    "duration_ms": (30_000, 1_200_000),  # 0.5 min to 20 min
}
