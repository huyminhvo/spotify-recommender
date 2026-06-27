from __future__ import annotations

try:
    from services import (
        AppError,
        add_recommendations_to_spotify,
        get_recommendations,
        get_spotify_client_or_raise,
    )
except ModuleNotFoundError:
    from webapp.services import (
        AppError,
        add_recommendations_to_spotify,
        get_recommendations,
        get_spotify_client_or_raise,
    )

__all__ = [
    "AppError",
    "add_recommendations_to_spotify",
    "get_recommendations",
    "get_spotify_client_or_raise",
]
