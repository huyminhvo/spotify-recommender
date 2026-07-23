import warnings

import numpy as np
import pytest

from recommender.similarity import cosine


def test_cosine_handles_zero_candidate_vectors_without_runtime_warnings():
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        similarities = cosine(query, candidates)

    np.testing.assert_allclose(similarities, [0.0, 1.0])


def test_cosine_rejects_mismatched_feature_dimensions_before_zero_shortcut():
    with pytest.raises(ValueError, match="same number of features"):
        cosine(np.zeros(2, dtype=np.float32), np.zeros((3, 4), dtype=np.float32))
