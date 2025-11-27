import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def fit_pca(X: np.ndarray, n_components: int = 12) -> PCA:
    """
    Fit PCA on the catalog feature matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix (e.g., output of preprocess.transform).
    n_components : int
        Maximum number of principal components to keep.

    Returns
    -------
    PCA
        Fitted PCA object that can be reused for candidates and user vectors.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features); got shape {X.shape}")

    # do not request more components than available features
    n_feats = X.shape[1]
    n_used = min(n_components, n_feats)

    pca = PCA(n_components=n_used, random_state=0)
    pca.fit(X)
    return pca


def transform_pca(X: np.ndarray, pca: PCA) -> np.ndarray:
    """
    Transform features using a fitted PCA.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix in the original feature space.
    pca : PCA
        Fitted PCA object from fit_pca.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        PCA-transformed features.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features); got shape {X.shape}")
    return pca.transform(X)


def fit_kmeans(X: np.ndarray, k: int = 8) -> KMeans:
    """Fit k-means on PCA-transformed features."""
    pass


def predict_kmeans(X: np.ndarray, kmeans: KMeans) -> np.ndarray:
    """Assign cluster labels to rows of X."""
    pass
