from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

from recommender.weightings import DEFAULT_WEIGHTS

RecommendationStrategy = Literal[
    "weighted_cosine",
    "unweighted_cosine",
    "popularity",
    "random",
]


@dataclass(frozen=True)
class RecommendationPolicy:
    """Immutable algorithm settings shared by deployment and evaluation."""

    strategy: RecommendationStrategy = "weighted_cosine"
    user_weights: Mapping[str, float] | None = None
    min_popularity: int | None = 20
    year_range: tuple[int, int] | None = None
    use_pca: bool = True
    pca_components: int = 5
    same_artist_exclusion: bool = False
    randomize_results: bool = False
    random_state: int | None = 0

    def __post_init__(self) -> None:
        if self.user_weights is not None:
            frozen_weights = MappingProxyType(
                {feature: float(weight) for feature, weight in self.user_weights.items()}
            )
            object.__setattr__(self, "user_weights", frozen_weights)
        if self.pca_components < 1:
            raise ValueError("pca_components must be at least 1.")
        if self.year_range is not None:
            lo, hi = self.year_range
            if lo > hi:
                raise ValueError("year_range lower bound must not exceed upper bound.")

    def candidate_kwargs(self) -> dict:
        """Arguments used to construct the eligible, bounded candidate pool."""
        return {
            "min_popularity": self.min_popularity,
            "year_range": self.year_range,
            "same_artist_exclusion": self.same_artist_exclusion,
        }

    def scoring_kwargs(self) -> dict:
        """Arguments used to score and select from a prepared candidate pool."""
        return {
            "user_weights": self.user_weights,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "strategy": self.strategy,
            "random_state": self.random_state,
            "randomize_results": self.randomize_results,
        }

    def recommendation_kwargs(self) -> dict:
        """All policy-controlled arguments accepted by recommend_from_catalog."""
        return {**self.candidate_kwargs(), **self.scoring_kwargs()}


DEPLOYED_POLICY = RecommendationPolicy(
    strategy="weighted_cosine",
    user_weights=DEFAULT_WEIGHTS,
    min_popularity=20,
    use_pca=True,
    pca_components=5,
    same_artist_exclusion=False,
    randomize_results=True,
    random_state=None,
)
