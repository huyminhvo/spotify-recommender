# Spotify Recommender

A full-stack music recommendation system that generates personalized playlists from any Spotify playlist.
Built in Python, powered by pandas, scikit-learn, and Streamlit, and integrated with the Spotify Web API.


## Features
- Paste any Spotify playlist URL to get personalized recommendations
- Album art, artist names, and direct Spotify links
- Full Spotify API integration (auth, playlist parsing, album art)
- Streamlit web interface with clean dark theme
- Cached dataset merging for fast repeat runs


## Run Locally
```bash
git clone https://github.com/<your-username>/spotify-recommender.git
cd spotify-recommender
python -m venv venv
venv\Scripts\activate    # (Windows)
source venv/bin/activate   # (Mac/Linux)
pip install -r webapp/requirements.txt
streamlit run webapp/streamlit_app.py
```


## Environment Variables
Create a .env file for local use (or set these as Streamlit secrets on deployment):
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


## Roadmap
- Cloud deployment (Streamlit Cloud / Render)
- Mood and energy sliders
- Save-to-Spotify playlist export
- Visual analytics panel


## License
MIT License © 2025 Huy Vo
