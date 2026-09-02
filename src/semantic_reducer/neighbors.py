"""Exact neighbour search over unit-norm type vectors.

Every backend here is EXACT: each returns all pairs whose cosine is at or above
the threshold, never an approximation. They differ only in speed and hardware.

A note on FAISS, since earlier versions of this package used it: ``IndexFlatIP``
is itself brute force, so it was never providing an approximate index -- only an
optimized matrix multiply. A chunked matmul is mathematically identical, runs on
whatever device the rest of the pipeline uses, and lets the threshold be applied
with exact ``>= tau`` semantics. FAISS remains available as an optional backend.
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_edges", "resolve_backend"]

# Decimal places retained when ordering edges. Coarse enough to absorb
# backend-to-backend float32 accumulation differences (~1e-7), fine enough that
# genuinely different similarities never collide.
_SORT_DECIMALS = 5


def resolve_backend(backend: str = "auto") -> str:
    """Resolve ``auto`` to the fastest backend actually installed."""
    if backend != "auto":
        return backend
    try:
        import torch  # noqa: F401
        return "torch"
    except ImportError:
        return "numpy"


def find_edges(
    X: np.ndarray,
    threshold: float,
    backend: str = "auto",
    chunk: int = 1024,
    device: "object | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find every pair of rows with cosine >= ``threshold``.

    Args:
        X: ``(n, dim)`` L2-normalized float32 array, so inner product is cosine.
        threshold: inclusive cutoff tau.
        backend: ``auto`` | ``torch`` | ``numpy`` | ``faiss``.
        chunk: rows per block; controls peak memory (``chunk * n`` floats).
        device: torch device for the ``torch`` backend; ignored otherwise.

    Returns:
        ``(sims, rows, cols)`` with ``rows < cols``, ordered by descending
        similarity and then by index, so the ordering is fully deterministic.
    """
    backend = resolve_backend(backend)
    if backend == "torch":
        sims, rows, cols = _edges_torch(X, threshold, chunk, device)
    elif backend == "numpy":
        sims, rows, cols = _edges_numpy(X, threshold, chunk)
    elif backend == "faiss":
        sims, rows, cols = _edges_faiss(X, threshold)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # Deterministic ordering: strongest first, ties broken by index. np.lexsort
    # treats the LAST key as primary.
    #
    # The similarity is quantized before sorting. Different backends -- and even
    # different chunk sizes on one backend -- accumulate the same dot product in
    # different orders and can land ~1e-7 apart, which would silently reorder
    # near-tied edges. Since agglomeration consumes edges strongest-first, that
    # would make the merge order depend on hardware. Rounding to a tolerance far
    # coarser than the noise, then breaking ties by index, gives one canonical
    # order everywhere.
    key = np.round(sims.astype(np.float64), _SORT_DECIMALS)
    order = np.lexsort((cols, rows, -key))
    return sims[order], rows[order], cols[order]


# --------------------------------------------------------------------------- #
def _edges_numpy(X, threshold, chunk):
    n = X.shape[0]
    sims_out, rows_out, cols_out = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        S = X[start:stop] @ X.T                       # (block, n)
        local_i, j = np.nonzero(S >= threshold)
        i = local_i + start
        keep = j > i                                  # upper triangle only
        if np.any(keep):
            i, j = i[keep], j[keep]
            sims_out.append(S[i - start, j])
            rows_out.append(i)
            cols_out.append(j)
    return _concat(sims_out, rows_out, cols_out)


def _edges_torch(X, threshold, chunk, device):
    import torch

    if device is None:
        device = torch.device("cpu")
    # float32 throughout: the threshold comparison must not depend on precision.
    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    n = Xt.shape[0]

    sims_out, rows_out, cols_out = [], [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        S = Xt[start:stop] @ Xt.T                     # (block, n)
        local_i, j = torch.nonzero(S >= threshold, as_tuple=True)
        i = local_i + start
        keep = j > i
        if bool(keep.any()):
            local_i, i, j = local_i[keep], i[keep], j[keep]
            sims_out.append(S[local_i, j].detach().cpu().numpy())
            rows_out.append(i.detach().cpu().numpy())
            cols_out.append(j.detach().cpu().numpy())
        del S
    return _concat(sims_out, rows_out, cols_out)


def _edges_faiss(X, threshold):
    import faiss

    index = faiss.IndexFlatIP(X.shape[1])
    index.add(np.ascontiguousarray(X))

    # FAISS range_search on an inner-product index is STRICT (ip > radius) and
    # compares in float32, so querying at exactly tau silently drops pairs on the
    # boundary. Query a hair below -- which returns a superset -- then apply the
    # documented inclusive test here, so tau means the same thing on every backend.
    eps = 1e-6
    lims, D, I = index.range_search(np.ascontiguousarray(X), float(threshold) - eps)

    sims_out, rows_out, cols_out = [], [], []
    for i in range(X.shape[0]):
        lo, hi = lims[i], lims[i + 1]
        if hi <= lo:
            continue
        j = I[lo:hi].astype(np.int64)
        d = D[lo:hi].astype(np.float32)
        keep = (j > i) & (d >= threshold)
        if np.any(keep):
            sims_out.append(d[keep])
            rows_out.append(np.full(int(keep.sum()), i, dtype=np.int64))
            cols_out.append(j[keep])
    return _concat(sims_out, rows_out, cols_out)


def _concat(sims_out, rows_out, cols_out):
    if not sims_out:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.concatenate(sims_out).astype(np.float32),
        np.concatenate(rows_out).astype(np.int64),
        np.concatenate(cols_out).astype(np.int64),
    )
