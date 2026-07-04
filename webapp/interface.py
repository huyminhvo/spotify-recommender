from __future__ import annotations

try:
    from services import (
        AppError,
        add_recommendations_to_spotify,
        get_recommendations,
        get_spotify_client_or_raise,
        load_catalog_bundle,
    )
except ModuleNotFoundError:
    from webapp.services import (
        AppError,
        add_recommendations_to_spotify,
        get_recommendations,
        get_spotify_client_or_raise,
        load_catalog_bundle,
    )

from recommender.steering import setting_scale_to_adjustment

__all__ = [
    "AppError",
    "add_recommendations_to_spotify",
    "get_recommendations",
    "get_spotify_client_or_raise",
    "load_catalog_bundle",
    "setting_scale_to_adjustment",
]
