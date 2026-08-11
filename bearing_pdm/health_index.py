"""Fuse many per-window features into one degradation-trend number per bearing.

By the time features_time.py and features_freq.py have both run, each
window has ~15 numbers describing it -- too many to eyeball as "healthy or
not," and no single one is reliable enough to trust alone (see
features_time.py's RMS-vs-kurtosis discussion). This module answers the
central question -- "how far has this window drifted from what healthy
looks like?" -- as one scalar per window, which is what health_index.py's
consumers (rul_model.py, evaluation.py, cost_threshold.py) actually need.

The metric is Mahalanobis distance from a healthy baseline: fit a mean
vector and covariance matrix on a bearing's own early-life windows (known
healthy -- every bearing in this dataset starts healthy and degrades),
then score every later window by how far its feature vector sits from
that mean, in units of the baseline's own natural variability.

Why Mahalanobis over the simpler alternatives:

- Plain Euclidean distance from the healthy mean would be dominated by
  whichever feature happens to have the largest raw numeric scale
  (spectral energy is O(1e3-1e6); kurtosis is O(1-10)) -- not by which
  feature is actually most anomalous. Mahalanobis distance divides out
  each feature's own scale and its correlations with the others, using
  the healthy baseline's covariance matrix.
- PCA-based Hotelling's T^2 / SPE is the textbook alternative for very
  high-dimensional, highly collinear feature sets where estimating a full
  covariance matrix directly is unstable. With ~15 features here and
  hundreds of healthy windows to fit on, direct estimation is feasible;
  Mahalanobis distance on the full covariance *is* Hotelling's T^2
  without the dimensionality-reduction step.
- An autoencoder's reconstruction error needs far more training data than
  a few hundred early-life windows per bearing provide, and trades away
  interpretability this dataset doesn't have enough data to earn back.
"""

from __future__ import annotations

import numpy as np

# Small ridge term added to the healthy covariance before inverting. With
# ~15 correlated features and a few hundred healthy windows, the raw
# covariance matrix can be near-singular; this is standard shrinkage
# regularisation (Tikhonov/ridge), not an arbitrary magic number.
COVARIANCE_RIDGE = 1e-6


def stack_features(features: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Turn a {name: array} feature dict into a (n_windows, n_features) matrix.

    Column order follows the dict's insertion order. The SAME order must
    be used both when fitting the healthy baseline and when scoring new
    data -- this function exists so that ordering is built once, not
    reconstructed by hand in two places where it could drift apart.
    """
    names = list(features.keys())
    matrix = np.column_stack([features[name] for name in names])
    return matrix, names


def fit_healthy_baseline(healthy_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a healthy-population mean and inverse covariance.

    `healthy_matrix` must be windows drawn only from a bearing's known
    early-life (healthy) period -- never from a held-out bearing being
    evaluated, and never from later in the same bearing's life.
    """
    mean = healthy_matrix.mean(axis=0)
    cov = np.cov(healthy_matrix, rowvar=False)
    cov += COVARIANCE_RIDGE * np.eye(cov.shape[0])
    inv_cov = np.linalg.inv(cov)
    return mean, inv_cov


def mahalanobis_health_index(matrix: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Score each row of `matrix` by its Mahalanobis distance from the healthy baseline."""
    diff = matrix - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv_cov, diff))