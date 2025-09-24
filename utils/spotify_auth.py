# utils/spotify_auth.py
"""
spotify_auth.py
---------------
Central place to create an authenticated Spotipy client.
"""

import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

def get_spotify_client() -> spotipy.Spotify:
    load_dotenv()  # load from .env

    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope="user-library-read playlist-read-private playlist-read-collaborative",
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        )
    )
