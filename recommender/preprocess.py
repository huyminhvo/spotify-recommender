from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import schema


def _resolve_duration_column(df: pd.DataFrame) -> str:
    """
    Prefer 'duration_ms' if present; otherwise accept 'duration' (assumed seconds or ms).
    If only 'duration' exists, we assume SECONDS. Adjust here if your catalog differs.
    """
    if "duration_ms" in df.columns:
        return "duration_ms"
    if "duration" in df.columns:
        return "duration"
    return "duration_ms"


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _apply_special_transforms(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """
    Apply mild, interpretable transforms to reduce skew:
      - tempo: log1p
      - duration_ms: convert to minutes then log1p
    Handles 'duration' fallback automatically.
    """
    out = df.copy()

    # handle duration column name
    dur_col = _resolve_duration_column(out)
    if dur_col != "duration_ms" and dur_col in out.columns and "duration_ms" in feature_order:
        # create a duration_ms surrogate (assume 'duration' is in seconds)
        out["duration_ms"] = out[dur_col] * 1000.0

    # tempo
    if "tempo" in out.columns and "tempo" in feature_order:
        # clip first (optional), then log1p
        lo, hi = schema.CLIP_BOUNDS.get("tempo", (None, None))
        if lo is not None:
            out["tempo"] = np.maximum(out["tempo"], lo)
        if hi is not None:
            out["tempo"] = np.minimum(out["tempo"], hi)
        out["tempo"] = np.log1p(out["tempo"].clip(lower=0.0))

    # duration_ms -> minutes then log1p
    if "duration_ms" in out.columns and "duration_ms" in feature_order:
        lo, hi = schema.CLIP_BOUNDS.get("duration_ms", (None, None))
        if lo is not None:
            out["duration_ms"] = np.maximum(out["duration_ms"], lo)
        if hi is not None:
            out["duration_ms"] = np.minimum(out["duration_ms"], hi)
        # convert ms -> minutes
        dur_min = out["duration_ms"] / 60000.0
        out["duration_ms"] = np.log1p(dur_min.clip(lower=0.0))

    # loudness clipping (dBFS)
    if "loudness" in out.columns and "loudness" in feature_order:
        lo, hi = schema.CLIP_BOUNDS.get("loudness", (None, None))
        if lo is not None:
            out["loudness"] = np.maximum(out["loudness"], lo)
        if hi is not None:
            out["loudness"] = np.minimum(out["loudness"], hi)

    return out


def _extract_feature_matrix(df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame with only the requested features, in the exact order.
    Will create 'duration_ms' surrogate if only 'duration' exists.
    """
    # ensure duration fallbacks are handled BEFORE selecting columns
    df2 = _apply_special_transforms(
        _coerce_numeric(df, [*feature_order, "duration"]), feature_order
    )

    missing = [c for c in feature_order if c not in df2.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    return df2[feature_order].copy()


def fit_scaler(df: pd.DataFrame, feature_cols: list[str] | None = None) -> StandardScaler:
    """
    Fit a StandardScaler on catalog features (after transforms).

    Parameters
    ----------
    df : pd.DataFrame
        Catalog DataFrame containing required feature columns.
    feature_cols : list[str] | None
        If None, uses schema.FEATURE_COLS.

    Returns
    -------
    StandardScaler
        Fitted scaler to be re-used for user seeds and candidates.
    """
    feats = feature_cols or schema.FEATURE_COLS
    Xdf = _extract_feature_matrix(df, feats)

    # Median-impute from the training/catalog distribution and retain those
    # values so user tracks and candidates are transformed consistently.
    impute_values = Xdf.median(numeric_only=True).reindex(feats).fillna(0.0)
    Xdf = Xdf.fillna(impute_values)

    scaler = StandardScaler()
    scaler.fit(Xdf.values.astype(np.float32))
    scaler.impute_values_ = impute_values.astype(np.float32)
    scaler.impute_feature_cols_ = list(feats)
    return scaler


def transform(
    df: pd.DataFrame, scaler: StandardScaler, feature_cols: list[str] | None = None
) -> np.ndarray:
    """
    Transform features into a scaled numpy array using a pre-fitted scaler.

    Notes
    -----
    - Applies the same transforms as fit (log1p on tempo/duration, clipping).
    - Uses median imputation learned by fit_scaler from the catalog/training df.
    """
    feats = feature_cols or schema.FEATURE_COLS
    Xdf = _extract_feature_matrix(df, feats)
    if hasattr(scaler, "impute_values_"):
        if getattr(scaler, "impute_feature_cols_", list(feats)) != list(feats):
            raise ValueError("feature_cols must match the columns used to fit the scaler.")
        impute_values = pd.Series(scaler.impute_values_, index=feats)
    else:
        # Backward compatibility for callers with an older persisted scaler.
        impute_values = Xdf.median(numeric_only=True).reindex(feats).fillna(0.0)
    Xdf = Xdf.fillna(impute_values)

    X = scaler.transform(Xdf.values.astype(np.float32))
    # ensure finite
    X = np.where(np.isfinite(X), X, 0.0)
    return X.astype(np.float32)
