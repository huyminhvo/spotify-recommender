from typing import List

# List of numerical features for modeling
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

# columns that may need special transforms
# - tempo: light log1p compression (skewed, positive)
# - duration_ms: convert to minutes then log1p (broad range, skewed)
SPECIAL_COLS = {
    "tempo": "log1p",        # assumes tempo >= 0
    "duration_ms": "log1p_min",  # convert ms -> minutes, then log1p
}

# reasonable numeric bounds for sanity clipping (very mild; optional)
# keep these wide to avoid distorting data but remove obvious outliers.
CLIP_BOUNDS = {
    "tempo": (0.0, 300.0),        # bpm
    "loudness": (-60.0, 0.0),     # dBFS typically ~[-60, 0]
    "duration_ms": (30_000, 1_200_000),  # 0.5 min to 20 min
}
