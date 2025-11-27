import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_spotify_client() -> spotipy.Spotify:
    load_dotenv()  # load from .env

    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope = (
            "playlist-modify-public "
            "playlist-modify-private "
            "playlist-read-private "
            "user-read-private"
            ),
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        )
    )
