from dataclasses import FrozenInstanceError

import pytest

from recommender.policy import DEPLOYED_POLICY, RecommendationPolicy
from recommender.weightings import DEFAULT_WEIGHTS


def test_deployed_policy_encodes_the_webapp_algorithm():
    assert DEPLOYED_POLICY.strategy == "weighted_cosine"
    assert dict(DEPLOYED_POLICY.user_weights) == DEFAULT_WEIGHTS
    assert DEPLOYED_POLICY.min_popularity == 20
    assert DEPLOYED_POLICY.use_pca is True
    assert DEPLOYED_POLICY.pca_components == 5
    assert DEPLOYED_POLICY.same_artist_exclusion is False
    assert DEPLOYED_POLICY.randomize_results is True
    assert DEPLOYED_POLICY.random_state is None


def test_recommendation_policy_is_deeply_immutable():
    weights = {"energy": 2.0}
    policy = RecommendationPolicy(user_weights=weights)
    weights["energy"] = 3.0

    assert policy.user_weights["energy"] == 2.0
    with pytest.raises(FrozenInstanceError):
        policy.use_pca = False
    with pytest.raises(TypeError):
        policy.user_weights["energy"] = 1.0


@pytest.mark.parametrize(
    "weights",
    [
        {"energgy": 1.0},
        {"energy": -1.0},
        {"energy": float("nan")},
        {"energy": float("inf")},
    ],
)
def test_recommendation_policy_rejects_invalid_feature_weights(weights):
    with pytest.raises(ValueError):
        RecommendationPolicy(user_weights=weights)


def test_recommendation_policy_allows_disabling_a_feature_with_zero_weight():
    policy = RecommendationPolicy(user_weights={"energy": 0.0})

    assert policy.user_weights["energy"] == 0.0


def test_policy_exposes_candidate_and_scoring_kwargs():
    assert DEPLOYED_POLICY.candidate_kwargs() == {
        "min_popularity": 20,
        "year_range": None,
        "same_artist_exclusion": False,
    }
    assert DEPLOYED_POLICY.scoring_kwargs()["strategy"] == "weighted_cosine"
    assert DEPLOYED_POLICY.recommendation_kwargs()["randomize_results"] is True
