from __future__ import annotations


class AppError(Exception):
    """Base exception for expected app failures with user-facing messages."""

    user_message = "Something went wrong. Please try again."

    def __init__(self, detail: str | None = None):
        self.detail = detail
        super().__init__(detail or self.user_message)


class InvalidPlaylistURLError(AppError):
    user_message = "That does not look like a valid Spotify playlist URL, URI, or ID."


class SpotifyAuthenticationError(AppError):
    user_message = "Spotify authentication failed. Check your Spotify API credentials and sign in again."


class SpotifyPlaylistAccessError(AppError):
    user_message = "Spotify could not access that playlist. Make sure it exists and is public or shared with your account."


class SpotifyRateLimitError(AppError):
    user_message = "Spotify rate-limited this request. Wait a bit, then try again."


class NoCatalogMatchesError(AppError):
    user_message = "No tracks from this playlist matched the local recommendation catalog."


class MissingDatasetError(AppError):
    user_message = "The local music dataset or cache is missing. Rebuild the catalog data before running recommendations."


class NoRecommendationTracksError(AppError):
    user_message = "There are no valid Spotify track IDs to add to a playlist."


class SpotifyServiceError(AppError):
    user_message = "Spotify returned an unexpected error. Please try again later."


def classify_spotify_error(exc: Exception) -> AppError:
    status = getattr(exc, "http_status", None)
    if status == 401:
        return SpotifyAuthenticationError(str(exc))
    if status == 429:
        return SpotifyRateLimitError(str(exc))
    if status in {403, 404}:
        return SpotifyPlaylistAccessError(str(exc))
    return SpotifyServiceError(str(exc))
