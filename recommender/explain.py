from collections.abc import Sequence

import numpy as np

EXPLAINABLE_FEATURES = ("energy", "valence", "acousticness", "tempo")


def explain_feature_similarity(
    user_profile: np.ndarray,
    candidate_features: np.ndarray,
    feature_order: Sequence[str],
    max_features: int = 3,
) -> list[str]:
    """Describe each candidate's closest standardized traits to the user profile."""
    user_profile = np.asarray(user_profile)
    candidate_features = np.asarray(candidate_features)

    if user_profile.ndim != 1:
        raise ValueError("user_profile must be a one-dimensional feature vector.")
    if candidate_features.ndim != 2:
        raise ValueError("candidate_features must be a two-dimensional feature matrix.")
    if candidate_features.shape[1] != len(feature_order) or len(user_profile) != len(feature_order):
        raise ValueError("Feature vectors must align with feature_order.")
    if max_features < 1:
        raise ValueError("max_features must be at least 1.")

    available = [
        (index, feature)
        for index, feature in enumerate(feature_order)
        if feature in EXPLAINABLE_FEATURES
    ]
    if not available:
        return ["Recommended because its overall audio profile is similar to your playlist."] * len(
            candidate_features
        )

    explanations = []
    for candidate in candidate_features:
        ranked = sorted(
            available,
            key=lambda item: (
                abs(float(candidate[item[0]] - user_profile[item[0]])),
                item[0],
            ),
        )
        traits = [feature for _, feature in ranked[:max_features]]
        explanations.append(_format_explanation(traits))
    return explanations


def _format_explanation(traits: Sequence[str]) -> str:
    if len(traits) == 1:
        trait_text = traits[0]
    elif len(traits) == 2:
        trait_text = f"{traits[0]} and {traits[1]}"
    else:
        trait_text = f"{', '.join(traits[:-1])}, and {traits[-1]}"
    return f"Recommended because it is similar to your playlist in {trait_text}."
