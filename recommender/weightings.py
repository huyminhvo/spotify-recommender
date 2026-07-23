from collections.abc import Mapping, Sequence

import numpy as np

from recommender.schema import FEATURE_COLS

# Hand-set vector multipliers used by the deployed policy. They have not yet
# been selected on a locked playlist-level tuning/test benchmark. Because the
# multipliers are applied to both the profile and candidate vectors, their
# effective contribution inside cosine similarity is squared.
DEFAULT_WEIGHTS = {
    "danceability": 1.5,
    "energy": 2.0,
    "valence": 0.5,
    "acousticness": 2.0,
    "instrumentalness": 2.0,
    "liveness": 0.5,
    "speechiness": 0.25,
    "tempo": 1.0,
    "loudness": 0.75,
    "duration_ms": 0.25,
}

_SUPPORTED_FEATURES = frozenset(FEATURE_COLS)


def validate_feature_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Return finite, nonnegative weights for supported recommender features."""
    unknown = set(weights) - _SUPPORTED_FEATURES
    if unknown:
        raise ValueError(f"Unsupported weight features: {sorted(unknown, key=str)}")

    validated: dict[str, float] = {}
    for feature, raw_weight in weights.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight for {feature!r} must be numeric.") from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Weight for {feature!r} must be finite and nonnegative.")
        validated[feature] = weight
    return validated


def apply_weights(
    X: np.ndarray,
    weights: Mapping[str, float],
    feature_order: Sequence[str],
) -> np.ndarray:
    """
    Apply feature weights to a vector or matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix or vector. Shape (n, d) or (d,).
    weights : dict[str, float]
        Mapping from feature name -> weight multiplier.
        Any features not in this dict get weight 1.0.
    feature_order : sequence of str
        Ordered list of features corresponding to columns in X.

    Returns
    -------
    np.ndarray
        Weighted features, same shape as input.
    """
    if X.ndim not in {1, 2}:
        raise ValueError("X must be a 1D or 2D numpy array.")

    features = tuple(feature_order)
    if len(features) != len(set(features)):
        raise ValueError("feature_order must not contain duplicate features.")
    unknown_features = set(features) - _SUPPORTED_FEATURES
    if unknown_features:
        raise ValueError(f"Unsupported features in feature_order: {sorted(unknown_features)}")

    feature_count = X.shape[0] if X.ndim == 1 else X.shape[1]
    if feature_count != len(features):
        raise ValueError(
            "The feature dimension of X must match the number of entries in feature_order."
        )

    validated_weights = validate_feature_weights(weights)
    # Build a weight vector aligned with feature_order.
    w = np.array([validated_weights.get(feature, 1.0) for feature in features], dtype=np.float32)

    if X.ndim == 1:  # single vector
        return X * w
    return X * w[None, :]
