from __future__ import annotations

try:
    from services import add_recommendations_to_spotify, get_recommendations
except ModuleNotFoundError:
    from webapp.services import add_recommendations_to_spotify, get_recommendations

__all__ = ["add_recommendations_to_spotify", "get_recommendations"]
