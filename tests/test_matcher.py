import pandas as pd

from utils.matcher import (
    build_indexes,
    canon_artist_primary,
    canon_title,
    match_track,
)


def test_canon_title_strips_accents_variant_tags_and_punctuation():
    assert canon_title("Cafe del Mar - Radio Edit") == "cafe del mar"
    assert canon_title("Halo (Remastered 2011)") == "halo"
    assert canon_title("Sweetest Thing!!!") == "sweetest thing"


def test_canon_artist_primary_handles_lists_and_stringified_lists():
    assert canon_artist_primary(["Beyonce", "JAY-Z"]) == "beyonce"
    assert canon_artist_primary("['Beyonce', 'JAY-Z']") == "beyonce"
    assert canon_artist_primary("Beyonce, JAY-Z") == "beyonce"


def test_match_track_prefers_exact_spotify_id():
    df = pd.DataFrame(
        [
            {
                "spotify_id": "known-id",
                "title_raw": "Known Song",
                "artists_raw": ["Known Artist"],
                "duration_ms": 200_000,
                "popularity": 10,
                "release_year": 2020,
            }
        ]
    )
    indexes = build_indexes(df)

    match = match_track({"id": "known-id", "name": "Anything Else"}, indexes, df)

    assert match["spotify_id"] == "known-id"
    assert match["title_raw"] == "Known Song"


def test_match_track_indexes_support_non_range_dataframe_index():
    df = pd.DataFrame(
        [
            {
                "spotify_id": "first-id",
                "title_raw": "First Song",
                "artists_raw": ["Artist"],
                "duration_ms": 180_000,
            },
            {
                "spotify_id": "second-id",
                "title_raw": "Second Song",
                "artists_raw": ["Artist"],
                "duration_ms": 200_000,
            },
        ],
        index=[10, 20],
    )
    indexes = build_indexes(df)

    match = match_track({"id": "second-id"}, indexes, df)

    assert match["title_raw"] == "Second Song"


def test_match_track_resolves_canonical_title_artist_and_duration():
    df = pd.DataFrame(
        [
            {
                "spotify_id": "catalog-id",
                "title_raw": "Cafe del Mar",
                "artists_raw": ["Energy 52"],
                "duration_ms": 198_500,
                "popularity": 60,
                "release_year": 1993,
            }
        ]
    )
    indexes = build_indexes(df)
    spotify_track = {
        "id": "unseen-id",
        "name": "Cafe del Mar - Radio Edit",
        "artists": [{"name": "Energy 52"}],
        "duration_ms": 199_000,
    }

    match = match_track(spotify_track, indexes, df)

    assert match["spotify_id"] == "catalog-id"


def test_match_track_rejects_duration_outside_tolerance():
    df = pd.DataFrame(
        [
            {
                "spotify_id": "catalog-id",
                "title_raw": "Same Song",
                "artists_raw": ["Same Artist"],
                "duration_ms": 180_000,
                "popularity": 60,
                "release_year": 2020,
            }
        ]
    )
    indexes = build_indexes(df)
    spotify_track = {
        "id": "different-id",
        "name": "Same Song",
        "artists": [{"name": "Same Artist"}],
        "duration_ms": 240_000,
    }

    assert match_track(spotify_track, indexes, df) is None


def test_match_track_breaks_ties_toward_canonical_version():
    df = pd.DataFrame(
        [
            {
                "spotify_id": "canonical",
                "title_raw": "Blue Monday",
                "artists_raw": ["New Order"],
                "duration_ms": 210_000,
                "popularity": 40,
                "release_year": 1983,
            },
            {
                "spotify_id": "live",
                "title_raw": "Blue Monday - Live",
                "artists_raw": ["New Order"],
                "duration_ms": 210_000,
                "popularity": 95,
                "release_year": 2021,
            },
        ]
    )
    indexes = build_indexes(df)
    spotify_track = {
        "id": "different-id",
        "name": "Blue Monday",
        "artists": [{"name": "New Order"}],
        "duration_ms": 210_000,
    }

    match = match_track(spotify_track, indexes, df)

    assert match["spotify_id"] == "canonical"
