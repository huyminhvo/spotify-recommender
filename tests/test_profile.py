import numpy as np
import pytest

from recommender.profile import build_user_profile


def test_build_user_profile_supports_mean_and_median():
    X_user = np.array(
        [
            [0.0, 1.0, 10.0],
            [2.0, 3.0, 20.0],
            [100.0, 5.0, 30.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(build_user_profile(X_user, method="mean"), [34.0, 3.0, 20.0])
    np.testing.assert_allclose(build_user_profile(X_user, method="median"), [2.0, 3.0, 20.0])


def test_build_user_profile_rejects_empty_or_unknown_method():
    with pytest.raises(ValueError, match="empty"):
        build_user_profile(np.array([], dtype=np.float32))

    with pytest.raises(ValueError, match="Unknown method"):
        build_user_profile(np.ones((2, 3), dtype=np.float32), method="clustered")


def test_build_user_profile_requires_a_2d_finite_result():
    with pytest.raises(ValueError, match="two-dimensional"):
        build_user_profile(np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="finite"):
        build_user_profile(np.array([[1.0, np.inf], [2.0, 3.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="finite"):
        build_user_profile(np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32))
