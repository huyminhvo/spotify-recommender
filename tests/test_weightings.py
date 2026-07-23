import numpy as np
import pytest

from recommender.weightings import apply_weights


def test_apply_weights_aligns_valid_weights_with_vectors_and_matrices():
    weights = {"energy": 2.0}
    feature_order = ["energy", "valence"]

    np.testing.assert_allclose(
        apply_weights(np.array([1.0, 3.0]), weights, feature_order),
        [2.0, 3.0],
    )
    np.testing.assert_allclose(
        apply_weights(np.array([[1.0, 3.0], [2.0, 4.0]]), weights, feature_order),
        [[2.0, 3.0], [4.0, 4.0]],
    )


@pytest.mark.parametrize(
    "weights",
    [
        {"energgy": 1.0},
        {"energy": -1.0},
        {"energy": float("nan")},
        {"energy": float("inf")},
    ],
)
def test_apply_weights_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        apply_weights(np.ones(2), weights, ["energy", "valence"])


def test_apply_weights_rejects_misaligned_feature_dimensions():
    with pytest.raises(ValueError, match="feature dimension"):
        apply_weights(np.ones((2, 3)), {"energy": 1.0}, ["energy", "valence"])


def test_apply_weights_allows_zero_and_ignores_supported_weights_outside_feature_order():
    result = apply_weights(
        np.ones(2),
        {"energy": 0.0, "danceability": 2.0},
        ["energy", "valence"],
    )

    np.testing.assert_allclose(result, [0.0, 1.0])
