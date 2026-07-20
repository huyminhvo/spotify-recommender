# Spotify Recommender

A full-stack music recommendation system that generates personalized playlists from any Spotify playlist.
Built in Python, powered by pandas, scikit-learn, and Streamlit, and integrated with the Spotify Web API.


## Features
- Paste any Spotify playlist URL to get personalized recommendations
- Album art, artist names, and direct Spotify links
- Optional mood, energy, danceability, and acousticness steering
- Full Spotify API integration (auth, playlist parsing, album art)
- Streamlit web interface
- Cached dataset merging for fast repeat runs


## Run Locally
```bash
git clone https://github.com/<your-username>/spotify-recommender.git
cd spotify-recommender
python -m venv venv
venv\Scripts\activate    # (Windows)
source venv/bin/activate   # (Mac/Linux)
pip install -r requirements.txt
streamlit run webapp/streamlit_app.py
```

## Development

Install the development dependencies and run the same checks used by CI:

```bash
pip install -r requirements-dev.txt
black --check .
ruff check .
pytest
```

To apply formatting and import fixes locally:

```bash
black .
ruff check --fix .
```

GitHub Actions runs formatting, linting, and tests on every push and pull request.


## Environment Variables
Create a .env file for local use:
```bash
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8501/
# Optional override for the deployment catalog and recommendation sample size:
CATALOG_PARQUET_PATH=/path/to/merged_catalog.parquet
CATALOG_CANDIDATE_LIMIT=100000
```

Register that exact redirect URI in the Spotify developer dashboard. For a
hosted deployment, set `SPOTIPY_REDIRECT_URI` to the app's exact HTTPS URL
(including its path and trailing slash, if any) and add the same value to
Streamlit secrets alongside the client ID and secret.

Album-art requests use Spotify's client-credentials flow. Spotify's current
Development Mode API exposes playlist items only when the authorized user owns
or collaborates on the playlist, so the app asks the user to connect before
generating recommendations. The same authorization is reused if they choose
**Add These Songs to Spotify Playlist**. User access and refresh tokens are kept
in that user's Streamlit session and are never written to a shared `.cache`
file.

Spotify Development Mode apps are limited to five allowlisted users, and the
app owner must have Spotify Premium. Plan for those restrictions before sharing
a hosted instance.

The Streamlit app queries the Parquet catalog with DuckDB instead of loading the
entire catalog and a pickled lookup index. Each recommendation request loads a
deterministic, dtype-optimized sample of at most `CATALOG_CANDIDATE_LIMIT`
eligible tracks; the default is 100,000.

The deployed app reads the immutable artifact named by `data/catalog/CURRENT`;
it never merges the raw CSV dataset during startup. To publish a new catalog:

```bash
git lfs install
python scripts/build_deployment_catalog.py .dataset_cache/merged_<fingerprint>.parquet --version v1
git add .gitattributes data/catalog/CURRENT data/catalog/*.parquet
git commit -m "Publish deployment catalog"
```

The artifact name includes a SHA-256 content prefix. Parquet files under
`data/catalog/` are tracked by Git LFS. After cloning, `git lfs pull` must leave
the artifact itself (not an LFS pointer) in the checkout. Raw inputs and
`.dataset_cache/` remain local development files and are not needed in Cloud.


## Architecture Overview
- webapp/streamlit_app.py : User interface and visualization layer
- webapp/interface.py : End-to-end orchestration (playlist to recommendations)
- recommender/ : Core recommendation logic (feature processing and cosine similarity)
- utils/ : Spotify API integration, caching, and dataset merging


## Recommendation and Evaluation

The recommender builds a median taste profile from seed tracks, applies
catalog-fitted preprocessing, and ranks a bounded deployment-catalog sample. The
deployed policy is centralized in `recommender/policy.py`: weighted cosine,
PCA(5), minimum popularity 20, a deterministic 100,000-track catalog sample,
and relevance-weighted random selection from the top candidate pool. The
feature multipliers are hand-set defaults, not claimed as benchmark-tuned.

The offline evaluator now keeps the item catalog and playlist membership labels
separate. It uses every non-seed playlist item as a positive, preserves
unmatched items when labels are built with the current schema, repeats splits,
and computes playlist-clustered bootstrap confidence intervals. It compares:

- random and popularity baselines;
- unweighted and weighted cosine without PCA;
- weighted cosine with PCA;
- the exact first-request deployed policy, with fixed random seeds for reproducibility;
- a fixed steering intervention, including target-distance diagnostics.

Catalog coverage is aggregated across all requests as distinct recommended
items divided by the eligible catalog, rather than `top_k / catalog_size`.
Reports also show the retrieval ceiling imposed by catalog matching,
popularity filtering, and the bounded candidate sample.

### Current result

<!-- evaluation-results:start -->

Smoke test over 9 playlists; do not treat as a quality claim.

| Strategy | Recall@10 | NDCG@10 | Hit rate | Catalog coverage |
|---|---:|---:|---:|---:|
| weighted_cosine_pca | 0.000 | 0.001 | 1.11% | 0.3503% |
| weighted_cosine | 0.000 | 0.001 | 1.67% | 0.3421% |
| unweighted_cosine | 0.000 | 0.001 | 1.11% | 0.3387% |
| deployed | 0.000 | 0.001 | 0.56% | 0.3577% |
| tuned_deployed | 0.000 | 0.001 | 0.56% | 0.3600% |
| tuned_weighted_cosine_pca | 0.000 | 0.001 | 0.56% | 0.3545% |
| weighted_cosine_pca_steered | 0.000 | 0.000 | 0.56% | 0.3324% |
| popularity | 0.000 | 0.000 | 0.00% | 0.0021% |
| random | 0.000 | 0.000 | 0.00% | 0.3760% |

[Full methodology and confidence intervals](reports/evaluation.md)

<!-- evaluation-results:end -->

The committed report is an engineering smoke test until the label set contains
at least 50–100 diverse playlists. Playlist count alone is not enough: inspect
match rate, track-set overlap, playlist length, and feature/era distributions
before making recommendation-quality claims.

### Build membership labels

Use playlists the authorized Spotify account owns or collaborates on. First,
authorize once in your browser; the helper stores a refreshable token in the
gitignored `.spotify_cache/` directory:

```bash
python scripts/authorize_spotify.py
```

After approval, copy the complete redirected URL from the browser address bar
back into the terminal. Then run the builder; expired access tokens are
refreshed automatically:

```bash
python scripts/build_evaluation_dataset.py \
  --playlist-file data/local/playlist_ids.txt \
  --output-csv data/local/playlist_membership.csv \
  --summary-csv data/local/playlist_match_summary.csv
```

The label-only output contains playlist/source IDs, the matched catalog ID,
position, and match status; it does not double as a retrieval catalog.

### Run the benchmark

```bash
python scripts/evaluate_recommender.py \
  --labels-csv data/local/playlist_membership.csv \
  --match-summary-csv data/local/playlist_match_summary.csv \
  --splits 20 \
  --bootstrap-samples 2000 \
  --update-readme
```

This writes [the Markdown report](reports/evaluation.md) and a machine-readable
`reports/evaluation_results.json`. Raw split and recommendation CSVs are
optional CLI outputs and remain local by default.

### Tune feature weights without test leakage

Weight search partitions whole playlists, evaluates candidates only on the
tuning partition, and records untouched test playlist IDs:

```bash
python scripts/tune_recommender_weights.py \
  --memberships-csv data/local/playlist_membership.csv \
  --trials 20 \
  --output-json reports/weight_tuning.json

python scripts/evaluate_recommender.py \
  --labels-csv data/local/playlist_membership.csv \
  --playlist-id-file reports/weight_tuning.json \
  --weights-json reports/weight_tuning.json
```

Weights are vector multipliers applied to both the profile and candidate
vectors, so their direct contribution inside cosine similarity is squared. The
held-out test results must not be used to revise the selected weights.


## Roadmap
- Streamlit Cloud deployment
- ML-based recommendation enhancements
- Visual analytics panel


## License
MIT License © 2026 Huy Vo
