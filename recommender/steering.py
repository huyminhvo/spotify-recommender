from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

STEERABLE_FEATURES = ("energy", "valence", "danceability", "acousticness")
DEFAULT_STEERING_STRENGTH = 0.7
DEFAULT_STEERING_WEIGHT = 0.2
MAX_ADJUSTMENT = 0.5
SETTING_MIN = 1.0
SETTING_MAX = 10.0
SETTING_NEUTRAL = (SETTING_MIN + SETTING_MAX) / 2


def setting_scale_to_adjustment(value: float) -> float:
    """Map the UI's 1-10 scale to a relative feature adjustment in [-0.5, 0.5]."""
    setting = float(value)
    if not np.isfinite(setting):
        raise ValueError("Recommendation setting must be finite.")
    setting = float(np.clip(setting, SETTING_MIN, SETTING_MAX))
    return (setting - SETTING_NEUTRAL) / (SETTING_MAX - SETTING_MIN)


def normalize_adjustments(adjustments: Mapping[str, float] | None) -> dict[str, float]:
    """Validate and clamp relative audio-feature shifts to the supported range."""
    if not adjustments:
        return {}

    unknown = set(adjustments) - set(STEERABLE_FEATURES)
    if unknown:
        raise ValueError(f"Unsupported steering features: {sorted(unknown)}")

    normalized = {}
    for feature, delta in adjustments.items():
        value = float(delta)
        if not np.isfinite(value):
            raise ValueError(f"Adjustment for {feature} must be finite.")
        if value != 0.0:
            normalized[feature] = float(np.clip(value, -MAX_ADJUSTMENT, MAX_ADJUSTMENT))
    return normalized


def rerank_with_adjustments(
    candidates: pd.DataFrame,
    user_tracks: pd.DataFrame,
    similarities: np.ndarray,
    adjustments: Mapping[str, float] | None,
    *,
    strength: float = DEFAULT_STEERING_STRENGTH,
    steering_weight: float = DEFAULT_STEERING_WEIGHT,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply interpretable feature-distance penalties to cosine similarity scores."""
    active = normalize_adjustments(adjustments)
    scores = np.asarray(similarities, dtype=float).copy()
    targets: dict[str, float] = {}

    for feature, delta in active.items():
        seed_values = pd.to_numeric(user_tracks[feature], errors="coerce")
        baseline = float(seed_values.median())
        if not np.isfinite(baseline):
            continue

        target = float(np.clip(baseline + delta * strength, 0.0, 1.0))
        candidate_values = pd.to_numeric(candidates[feature], errors="coerce").fillna(baseline)
        scores -= steering_weight * np.abs(candidate_values.to_numpy(dtype=float) - target)
        targets[feature] = target

    return scores, targets
