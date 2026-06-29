import numpy as np
import pandas as pd
import pytest

from recommender.steering import normalize_adjustments, rerank_with_adjustments


def test_adjustments_are_clamped_and_unknown_features_rejected():
    assert normalize_adjustments({"energy": 1.0, "valence": -1.0}) == {
        "energy": 0.3,
        "valence": -0.3,
    }

    with pytest.raises(ValueError, match="Unsupported steering"):
        normalize_adjustments({"tempo": 0.1})


def test_reranking_clamps_target_to_audio_feature_bounds():
    candidates = pd.DataFrame({"energy": [0.9, 1.0]})
    seeds = pd.DataFrame({"energy": [0.95]})

    scores, targets = rerank_with_adjustments(
        candidates,
        seeds,
        np.array([0.8, 0.8]),
        {"energy": 0.3},
    )

    assert targets == {"energy": 1.0}
    assert scores[1] > scores[0]
