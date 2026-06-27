# Spotify Recommender

A full-stack music recommendation system that generates personalized playlists from any Spotify playlist.
Built in Python, powered by pandas, scikit-learn, and Streamlit, and integrated with the Spotify Web API.


## Features
- Paste any Spotify playlist URL to get personalized recommendations
- Album art, artist names, and direct Spotify links
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
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```


## Architecture Overview
- webapp/streamlit_app.py : User interface and visualization layer
- webapp/interface.py : End-to-end orchestration (playlist to recommendations)
- recommender/ : Core recommendation logic (feature processing and cosine similarity)
- utils/ : Spotify API integration, caching, and dataset merging


## Recommendation and Evaluation
The main recommender builds a median taste profile from seed tracks, scales audio
features with catalog-fitted preprocessing, and ranks candidate tracks with cosine
similarity. The recommendation module also includes comparison baselines:

- `random`: random eligible candidates
- `popularity`: most popular eligible candidates
- `unweighted_cosine`: cosine similarity without feature weights
- `weighted_cosine`: cosine similarity with tuned audio-feature weights

Same-artist exclusion can be enabled to force discovery outside the seed artists.

To compare strategies on a playlist-labeled CSV, run:

```bash
python scripts/evaluate_recommender.py --catalog-csv path/to/playlists.csv --playlist-col playlist_id --top-k 10 --seed-size 5 --holdout-size 5
```

To evaluate real Spotify playlists with defaults, add playlist URLs to
`data/examples/playlists.txt`, then run:

```bash
python scripts/evaluate_recommender.py
```

By default, the script builds `data/examples/real_playlist_eval.csv` from
`data/examples/playlists.txt` if needed. On later runs, it reuses that saved CSV.
For custom runs, pass explicit options:

```bash
python scripts/evaluate_recommender.py --catalog-csv data/examples/real_playlist_eval.csv --playlist-col playlist_id --top-k 10 --seed-size 5 --holdout-size 5
```

The evaluator performs a seed/holdout split per playlist and reports
ranking metrics (`precision_at_k`, `recall_at_k`, `hit_rate_at_k`, `ndcg_at_k`),
recommendation diagnostics (`coverage_rate`, `avg_similarity`, `artist_diversity`,
`artist_duplication_rate`, average recommendation popularity), and build-quality
metrics (`match_rate`, raw/merged catalog size, duplicate reduction rate).


## Roadmap
- Streamlit Cloud deployment
- Mood and energy sliders
- ML-based recommendation enhancements
- Visual analytics panel


## License
MIT License © 2026 Huy Vo
