import numpy as np
import pytest

from recommender.explain import explain_feature_similarity


def test_explain_feature_similarity_selects_closest_interpretable_traits():
    feature_order = ["energy", "valence", "acousticness", "tempo", "loudness"]
    user_profile = np.zeros(5)
    candidates = np.array(
        [
            [0.1, 0.2, 2.0, 0.3, 0.0],
            [3.0, 0.1, 0.3, 0.2, 0.0],
        ]
    )

    explanations = explain_feature_similarity(user_profile, candidates, feature_order)

    assert explanations == [
        "Recommended because it is similar to your playlist in energy, valence, and tempo.",
        "Recommended because it is similar to your playlist in valence, tempo, and acousticness.",
    ]


def test_explain_feature_similarity_validates_feature_shapes():
    with pytest.raises(ValueError, match="align"):
        explain_feature_similarity(
            np.zeros(2),
            np.zeros((1, 3)),
            ["energy", "valence"],
        )
