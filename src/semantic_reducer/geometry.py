"""Embedding geometry: anisotropy correction and normalization.

Pure numpy, no heavy dependencies, so the geometry can be tested on its own.
"""

from __future__ import annotations

import numpy as np

__all__ = ["correct_anisotropy", "fit_anisotropy_correction",
          "apply_anisotropy_correction", "l2_normalize", "mean_offdiagonal_cosine"]


def fit_anisotropy_correction(X: np.ndarray, n_abtt: int = 2, center: bool = True):
    """Mean-center, then remove the top ``n_abtt`` principal components.

    Contextual embeddings from a Transformer encoder are strongly anisotropic:
    they occupy a narrow cone, so the cosine between two *unrelated* words is
    typically around 0.9. A cosine threshold applied to raw vectors therefore
    measures the cone rather than semantic relatedness, and is not interpretable.
    Mean-centering removes the shared component; "all-but-the-top" removes the
    dominant residual directions (Mu & Viswanath, 2018; Ethayarajh, 2019).

    The transformation is estimated from the *population* of type vectors, so it
    is only meaningful for a reasonably sized vocabulary -- on a handful of types
    it is degenerate, since two centered points are always exactly antipodal.

    Unlike :func:`correct_anisotropy`, this also returns the mean and principal
    directions the correction estimated, so the identical transform can later be
    applied to a NEW vector (e.g. an out-of-vocabulary word encoded at inference
    time) via :func:`apply_anisotropy_correction`. Comparing a freshly-encoded
    query against corpus vectors that went through this correction, without
    applying the same correction to the query first, compares two different
    coordinate systems and produces meaningless similarities.

    Args:
        X: ``(n_types, dim)`` array, not yet L2-normalized.
        n_abtt: number of principal directions to remove; 0 centers only.
        center: whether to mean-center. ``center=False, n_abtt=0`` disables
            correction entirely, which is the ablation baseline.

    Returns:
        ``(corrected, mean, principal_components)``: ``corrected`` is
        ``(n_types, dim)`` float32, not yet normalized. ``mean`` is
        ``(1, dim)`` float32 (zeros if ``center=False``). ``principal_components``
        is ``(k, dim)`` float32 with ``k = n_abtt`` (or ``0`` if it was skipped).
    """
    X = np.asarray(X, dtype=np.float32).copy()
    dim = X.shape[1]
    mean = X.mean(axis=0, keepdims=True).astype(np.float32) if center \
        else np.zeros((1, dim), dtype=np.float32)
    X = X - mean
    if n_abtt > 0 and X.shape[0] > n_abtt:
        # Economy SVD of the centered matrix; rows of Vt are the principal
        # directions ordered by explained variance.
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = Vt[:n_abtt].astype(np.float32)     # (n_abtt, dim)
        X = X - (X @ pcs.T) @ pcs                # subtract those projections
    else:
        pcs = np.zeros((0, dim), dtype=np.float32)
    return X.astype(np.float32), mean, pcs


def apply_anisotropy_correction(X: np.ndarray, mean: np.ndarray,
                                principal_components: np.ndarray) -> np.ndarray:
    """Apply an ALREADY-FITTED correction (from :func:`fit_anisotropy_correction`)
    to new vectors, e.g. out-of-vocabulary words at inference time.

    Using the corpus-estimated mean and directions rather than re-estimating
    from the (possibly single) new vector is the point: a lone vector has no
    population to estimate anisotropy from, and the whole reason the query and
    the corpus vectors need to be compared at all is that they should live in
    the same corrected space.
    """
    X = np.asarray(X, dtype=np.float32) - mean
    if principal_components.shape[0] > 0:
        X = X - (X @ principal_components.T) @ principal_components
    return X.astype(np.float32)


def correct_anisotropy(X: np.ndarray, n_abtt: int = 2, center: bool = True) -> np.ndarray:
    """Mean-center, then remove the top ``n_abtt`` principal components.

    Thin wrapper over :func:`fit_anisotropy_correction` that discards the
    fitted mean/directions; kept for callers that only need the corrected
    matrix itself and were not passing a query vector through separately.
    See :func:`fit_anisotropy_correction` for the full docstring.
    """
    corrected, _, _ = fit_anisotropy_correction(X, n_abtt=n_abtt, center=center)
    return corrected


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit norm so that inner product equals cosine.

    The denominator is clamped, so a row that collapses to zero (which anisotropy
    correction can produce) yields zeros rather than NaNs.
    """
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


def mean_offdiagonal_cosine(X: np.ndarray, sample: int | None = 2000,
                            seed: int = 0) -> float:
    """Average cosine between distinct rows -- a direct read of anisotropy.

    Near 1.0 means the vectors sit in a narrow cone and no cosine threshold will
    separate related from unrelated words. Reported by
    :meth:`SemanticReducer.geometry_report` before and after correction.

    Args:
        X: ``(n, dim)`` array; normalized internally.
        sample: cap on rows used, since the full matrix is O(n^2). ``None`` uses all.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    if n < 2:
        return float("nan")
    if sample is not None and n > sample:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(n, size=sample, replace=False)]
        n = sample
    Xn = l2_normalize(X)
    S = Xn @ Xn.T
    return float((S.sum() - np.trace(S)) / (n * n - n))
