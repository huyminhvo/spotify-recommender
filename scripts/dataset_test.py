# scripts/dataset_test.py
from merge_datasets import get_merged_dataset, AUDIO_FEATURES
import pandas as pd

# --- 1. Load merged dataset (cached if available) ---
paths = ["tracks_features.csv", "spotify_data.csv"]  # adjust filenames if needed
df = get_merged_dataset(paths)

print("\n=== Basic Info ===")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# --- 2. Verify core columns exist ---
expected_cols = [
    "spotify_id","title_raw","title_canon","artists_raw","artist_primary_canon",
    "duration_ms","explicit","popularity","release_year","isrc","album"
] + AUDIO_FEATURES

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("❌ Missing expected columns:", missing)
else:
    print("✅ All expected columns present")

# --- 3. Check for duplicate IDs ---
dupe_ids = df[df.duplicated(subset=["spotify_id"], keep=False)]
print("\nDuplicate spotify_id rows:", len(dupe_ids))

# --- 4. Sample rows ---
print("\n=== Sample rows ===")
print(df.head(10)[["spotify_id","title_raw","artist_primary_canon","duration_ms"] + AUDIO_FEATURES[:3]])

# --- 5. Audio feature sanity checks ---
print("\n=== Audio feature ranges ===")
for feat in AUDIO_FEATURES:
    if feat in df:
        v = pd.to_numeric(df[feat], errors="coerce")
        print(f"{feat:15} min={v.min(skipna=True)}  max={v.max(skipna=True)}")

# --- 6. Spot check: how many rows have no spotify_id but do have a canon key ---
no_id = df[df["spotify_id"].isna()]
print("\nRows without spotify_id:", len(no_id))
if not no_id.empty:
    print(no_id.head(5)[["title_raw","artist_primary_canon","duration_ms"]])
