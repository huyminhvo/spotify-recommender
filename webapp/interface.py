from __future__ import annotations

if __package__:
    from webapp.services import (
        AppError,
        add_recommendations_to_spotify,
        format_artist_names,
        get_recommendations,
        get_spotify_client_or_raise,
        load_catalog_bundle,
    )
else:
    from services import (
        AppError,
        add_recommendations_to_spotify,
        format_artist_names,
        get_recommendations,
        get_spotify_client_or_raise,
        load_catalog_bundle,
    )

from recommender.steering import setting_scale_to_adjustment

__all__ = [
    "AppError",
    "add_recommendations_to_spotify",
    "format_artist_names",
    "get_recommendations",
    "get_spotify_client_or_raise",
    "load_catalog_bundle",
    "setting_scale_to_adjustment",
]
