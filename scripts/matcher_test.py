# scripts/matcher_test.py
from matcher import canon_title, canon_artist_primary, build_indexes, match_track

# --- 1. Test canonicalization ---
print("=== Canonicalization Tests ===")
titles = [
    "Song Title (Remastered 2011)",
    "Song Title - Live at Wembley",
    "  Béyoncé  ",
    "Track Name - Explicit",
]
for t in titles:
    print(f"{t!r} -> {canon_title(t)}")

artists_cases = [
    ["Beyoncé", "Jay-Z"],
    "Beyoncé, Jay-Z",
    "[\"Beyoncé\"]",
    None,
]
for a in artists_cases:
    print(f"{a!r} -> {canon_artist_primary(a)}")

# --- 2. Build a tiny mock dataset ---
mock_rows = [
    {
        "spotify_id": "abc123",
        "title_raw": "Halo (Remastered)",
        "title_canon": canon_title("Halo (Remastered)"),
        "artists_raw": ["Beyoncé"],
        "artist_primary_canon": canon_artist_primary(["Beyoncé"]),
        "duration_ms": 260000,
        "popularity": 80,
        "release_year": 2009,
    },
    {
        "spotify_id": "def456",
        "title_raw": "99 Problems - Live",
        "title_canon": canon_title("99 Problems - Live"),
        "artists_raw": ["Jay-Z"],
        "artist_primary_canon": canon_artist_primary(["Jay-Z"]),
        "duration_ms": 230000,
        "popularity": 70,
        "release_year": 2004,
    },
]
import pandas as pd
df = pd.DataFrame(mock_rows)

indexes = build_indexes(df)
print("\n=== Indexes built ===")
print("by_id keys:", list(indexes["by_id"].keys()))
print("by_key keys:", list(indexes["by_key"].keys()))
print("by_artist keys:", list(indexes["by_artist"].keys()))

# --- 3. Test matching a track dict ---
track = {
    "id": "abc123",
    "name": "Halo - Remastered 2011",
    "artists": [{"name": "Beyoncé"}],
    "duration_ms": 260123,
}
print("\n=== Match Track Test ===")
result = match_track(track, indexes)
print("Input track:", track)
print("Matched row:", dict(result) if result is not None else None)

# --- 4. Negative test: unmatched track ---
bad_track = {
    "id": "zzz999",
    "name": "Nonexistent Song",
    "artists": [{"name": "Unknown Artist"}],
    "duration_ms": 200000,
}
print("\nUnmatched track result:", match_track(bad_track, indexes))
