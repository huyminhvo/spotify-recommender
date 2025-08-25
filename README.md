# Spotify Recommender

A personal project to build a recommendation engine on top of Spotify user data and large public track datasets. The goal is to recommend new songs using audio features, artist/title metadata, and similarity metrics.

## Project Status

**Done**
- Merged ~2M tracks from multiple datasets
- Normalized metadata (IDs, titles, artists, albums, ISRC)
- Canonicalization functions for titles & primary artists
- Deduplication in three passes: by Spotify ID, by ISRC, and by (title + artist + duration bucket)
- Integrated audio features (danceability, energy, valence, speechiness, acousticness, instrumentalness, liveness, loudness, tempo, key, mode)
- Added caching: merged dataset saved as Parquet for fast reloads

**Next**
- Feature vector construction
- Cosine similarity (with optional feature weights)
- Candidate selection + diversification
- Simple API / web UI

## Setup

1) Create and activate a virtual environment

    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

2) Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

    (Currently: `pandas`, `pyarrow`, optionally `spotipy` for API tests.)

3) Run the dataset sanity test

    ```bash
    python scripts/dataset_test.py
    ```

    The first run builds the merged cache (minutes). Subsequent runs load from Parquet (seconds).

## Repository Structure

```text
spotify-recommender/
├─ merge_datasets.py       # merge + dedupe pipeline (with caching helpers)
├─ matcher.py              # canonicalization + matching logic
├─ scripts/
│  ├─ dataset_test.py      # tests merge_datasets and cache
│  └─ matcher_test.py      # tests matcher canonicalization & matching
└─ .dataset_cache/         # cached Parquet files (auto-created)
```

## Roadmap

- [x] Merge + normalize datasets
- [x] Audio features integrated
- [x] Cache merged DataFrame to Parquet
- [ ] Build feature vectors
- [ ] Similarity scoring (cosine, weighted)
- [ ] Recommendation API
- [ ] Web UI (optional stretch goal)

## Notes

This project emphasizes clarity and reproducibility over premature optimization. The merge step is treated as an offline ETL job and cached to avoid user-facing delays.


