import numpy as np
import pandas as pd
import pytest

from recommender.preprocess import _extract_feature_matrix, fit_scaler, transform
from recommender.schema import FEATURE_COLS


def _feature_row(**overrides):
    row = {
        "danceability": 0.5,
        "energy": 0.7,
        "valence": 0.6,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "speechiness": 0.05,
        "tempo": 120.0,
        "loudness": -8.0,
        "duration_ms": 210_000,
    }
    row.update(overrides)
    return row


def test_fit_and_transform_return_finite_matrix_in_feature_order():
    df = pd.DataFrame(
        [
            _feature_row(spotify_id="a", tempo=90.0),
            _feature_row(spotify_id="b", tempo=180.0, loudness=-4.0),
            _feature_row(spotify_id="c", tempo=None, duration_ms=None),
        ]
    )

    scaler = fit_scaler(df)
    X = transform(df, scaler)

    assert X.shape == (3, len(FEATURE_COLS))
    assert X.dtype == np.float32
    assert np.isfinite(X).all()


def test_transform_uses_training_medians_for_imputation():
    catalog = pd.DataFrame(
        [
            _feature_row(danceability=0.2, energy=0.2),
            _feature_row(danceability=0.8, energy=0.8),
            _feature_row(danceability=None, energy=0.5),
        ]
    )
    user_tracks = pd.DataFrame(
        [
            _feature_row(danceability=None, energy=0.2),
            _feature_row(danceability=0.9, energy=0.8),
        ]
    )
    scaler = fit_scaler(catalog)

    transformed = transform(user_tracks, scaler)
    expected_raw = _extract_feature_matrix(user_tracks, FEATURE_COLS)
    expected_raw.loc[0, "danceability"] = 0.5
    expected_raw = expected_raw.fillna(pd.Series(scaler.impute_values_, index=FEATURE_COLS))
    expected = scaler.transform(expected_raw.values.astype(np.float32)).astype(np.float32)

    assert scaler.impute_values_["danceability"] == pytest.approx(0.5)
    assert transformed[0, FEATURE_COLS.index("danceability")] == pytest.approx(
        expected[0, FEATURE_COLS.index("danceability")]
    )


def test_extract_feature_matrix_converts_duration_seconds_fallback():
    df = pd.DataFrame([_feature_row(duration=210.0)])
    df = df.drop(columns=["duration_ms"])

    Xdf = _extract_feature_matrix(df, FEATURE_COLS)

    assert list(Xdf.columns) == FEATURE_COLS
    assert Xdf.loc[0, "duration_ms"] == pytest.approx(np.log1p(3.5))


def test_extract_feature_matrix_clips_and_transforms_special_columns():
    df = pd.DataFrame([_feature_row(tempo=500.0, loudness=5.0, duration_ms=2_400_000)])

    Xdf = _extract_feature_matrix(df, FEATURE_COLS)

    assert Xdf.loc[0, "tempo"] == pytest.approx(np.log1p(300.0))
    assert Xdf.loc[0, "loudness"] == 0.0
    assert Xdf.loc[0, "duration_ms"] == pytest.approx(np.log1p(20.0))


def test_extract_feature_matrix_raises_for_missing_required_columns():
    df = pd.DataFrame([_feature_row()]).drop(columns=["energy"])

    with pytest.raises(KeyError, match="Missing required feature columns"):
        _extract_feature_matrix(df, FEATURE_COLS)
