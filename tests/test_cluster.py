import numpy as np
import pytest

from recommender.cluster import fit_pca, transform_pca


def test_fit_pca_caps_components_by_samples_and_features():
    X = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    pca = fit_pca(X, n_components=10)
    transformed = transform_pca(X, pca)

    assert pca.n_components == 3
    assert transformed.shape == (3, 3)


def test_fit_pca_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        fit_pca(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="at least one sample"):
        fit_pca(np.empty((0, 3)))

    with pytest.raises(ValueError, match="at least 1"):
        fit_pca(np.ones((3, 2)), n_components=0)
