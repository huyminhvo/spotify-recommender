# Spotify Recommender

A full-stack music recommendation system that generates personalized playlists from any Spotify playlist.  
Built in **Python**, powered by **pandas**, **scikit-learn**, and **Streamlit**, and integrated with the **Spotify Web API**.

---

## Web App MVP (October 2025)

Paste any Spotify playlist URL and instantly generate personalized track recommendations, complete with album art, artist info, and direct Spotify links.

**Highlights**
- Full Spotify API integration (auth, playlist parsing, album art)
- Recommendation engine using cosine similarity on track feature vectors
- Streamlit web interface (interactive and deployable)
- Cached dataset merge for fast repeat runs

**Run locally**
```bash
pip install -r webapp/requirements.txt
streamlit run webapp/streamlit_app.py
```

---

## Features
- Merge + deduplicate multiple Spotify datasets (millions of rows)
- Normalize and compare audio features across tracks
- Generate top-N recommendations from any Spotify playlist
- Lightweight, modular codebase ready for scaling and experimentation

---

## Setup

### 1. Clone & Create Environment
```bash
git clone https://github.com/<your-username>/spotify-recommender.git
cd spotify-recommender
python -m venv venv
venv\Scripts\activate    # (Windows)
source venv/bin/activate # (Mac/Linux)
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```bash
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```

---

## CLI Usage <a name="cli-usage"></a>
You can also (alternatively) run the recommender directly from the command line.

```bash
python cli.py --playlist "https://open.spotify.com/playlist/4bvPBOdMcU0dVJQqP86upR" --top_n 20
```

Example output:
```bash
[cache] Using cached merged_xxx.parquet
Top 20 recommended tracks:
1. Artist – Track
2. Artist – Track
...
```
---

## Roadmap
- Web app MVP (October 2025)
- Cloud deployment (Streamlit Cloud / Render)
- Mood & energy sliders
- Multi-profile recommendations
- Demo video + hosted link

---

## License
For educational and portfolio purposes only.
