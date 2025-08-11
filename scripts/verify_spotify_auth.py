import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load credentials from .env
load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    scope="user-top-read user-read-recently-played",
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI")
))

# Get current user's display name
user = sp.current_user()
print(f"Authenticated as: {user['display_name']} ({user['id']})")

# Sample top tracks
top_tracks = sp.current_user_top_tracks(limit=5, time_range="medium_term")
print("Top track sample:", [t['name'] for t in top_tracks['items']])

# Sample recently played
recent = sp.current_user_recently_played(limit=5)
print("Recently played sample:", [t['track']['name'] for t in recent['items']])
