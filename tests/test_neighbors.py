"""Tests for exact neighbour retrieval.

Two things must hold no matter which backend runs: the threshold is inclusive
(``>= tau``), and every backend returns exactly the same edge set. The FAISS
backend needs special care because ``range_search`` on an inner-product index is
strict and compares in float32, so a pair sitting exactly at tau would otherwise
be dropped depending on hardware.
"""

import numpy as np
import pytest

from conftest import unit

from semantic_reducer import find_edges
from semantic_reducer.neighbors import resolve_backend

BACKENDS = ["numpy", "torch"]
try:  # optional dependency
    import faiss  # noqa: F401
    BACKENDS.append("faiss")
except ImportError:
    pass


def brute_force_edges(X, tau):
    """Reference implementation: every pair, checked directly."""
    S = X @ X.T
    return {
        (i, j)
        for i in range(X.shape[0])
        for j in range(i + 1, X.shape[0])
        if S[i, j] >= tau
    }


@pytest.fixture
def spread():
    rng = np.random.default_rng(11)
    angles = np.sort(rng.uniform(0, 100, size=40))
    return np.stack([unit(a, dim=4) for a in angles])


class TestBackendAgreement:
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("tau", [0.3, 0.6, 0.9, 0.99])
    def test_matches_brute_force(self, backend, tau, spread):
        sims, rows, cols = find_edges(spread, threshold=tau, backend=backend)
        assert {(int(r), int(c)) for r, c in zip(rows, cols)} == brute_force_edges(spread, tau)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_reported_similarities_are_correct(self, backend, spread):
        sims, rows, cols = find_edges(spread, threshold=0.5, backend=backend)
        S = spread @ spread.T
        for sim, i, j in zip(sims, rows, cols):
            assert sim == pytest.approx(S[i, j], abs=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_only_upper_triangle_is_returned(self, backend, spread):
        _, rows, cols = find_edges(spread, threshold=0.5, backend=backend)
        assert np.all(rows < cols), "each undirected edge must appear exactly once"

    def test_all_backends_find_the_same_edges(self, spread):
        results = {
            backend: find_edges(spread, threshold=0.7, backend=backend)
            for backend in BACKENDS
        }
        reference = brute_force_edges(spread, 0.7)
        for backend, (_, rows, cols) in results.items():
            edges = {(int(r), int(c)) for r, c in zip(rows, cols)}
            assert edges == reference, f"{backend} disagrees on the edge set"

    def test_all_backends_agree_on_edge_ordering(self, spread):
        """Ordering matters: agglomeration consumes edges strongest-first.

        Backends accumulate dot products in different orders and land ~1e-7
        apart, so the sort key is quantized to keep near-tied edges in one
        canonical order regardless of hardware.
        """
        results = {
            backend: find_edges(spread, threshold=0.7, backend=backend)
            for backend in BACKENDS
        }
        reference = results[BACKENDS[0]]
        for backend, (sims, rows, cols) in results.items():
            assert np.array_equal(rows, reference[1]), f"{backend} disagrees on rows"
            assert np.array_equal(cols, reference[2]), f"{backend} disagrees on cols"
            assert np.allclose(sims, reference[0], atol=1e-5), f"{backend} disagrees on sims"

    def test_all_backends_produce_the_same_clusters(self, spread):
        """The property that actually matters downstream."""
        from semantic_reducer import agglomerate

        reference = None
        for backend in BACKENDS:
            sims, rows, cols = find_edges(spread, threshold=0.7, backend=backend)
            components = agglomerate(
                spread, sims, rows, cols, threshold=0.7, linkage=1.0
            ).components
            if reference is None:
                reference = components
            assert components == reference, f"{backend} produced different clusters"


class TestThresholdBoundary:
    """A pair exactly at tau must be included on every backend."""

    @staticmethod
    def exact_pair():
        # cos = 0.6 exactly: both rows are unit vectors and 0.6/0.8 is exact in binary.
        X = np.zeros((2, 4), dtype=np.float32)
        X[0, 0] = 1.0
        X[1, 0], X[1, 1] = 0.6, 0.8
        return X

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_boundary_is_inclusive(self, backend):
        X = self.exact_pair()
        _, rows, _ = find_edges(X, threshold=0.6, backend=backend)
        assert len(rows) == 1, f"{backend} dropped the pair sitting exactly at tau"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_just_above_the_boundary_excludes(self, backend):
        X = self.exact_pair()
        _, rows, _ = find_edges(X, threshold=0.6001, backend=backend)
        assert len(rows) == 0


class TestOrderingAndChunking:
    def test_edges_are_sorted_by_descending_similarity(self, spread):
        """Descending to within the quantization tolerance, exactly.

        The sort key is rounded so that near-tied edges order identically on
        every backend, so raw similarities may differ by up to one quantum in
        either direction; the quantized key must be strictly non-increasing.
        """
        from semantic_reducer.neighbors import _SORT_DECIMALS

        sims, _, _ = find_edges(spread, threshold=0.4, backend="numpy")
        quantum = 10.0 ** (-_SORT_DECIMALS)

        key = np.round(sims.astype(np.float64), _SORT_DECIMALS)
        assert np.all(np.diff(key) <= 0), "quantized sort key is not monotone"
        assert np.all(np.diff(sims) <= quantum)

    def test_ordering_is_deterministic(self, spread):
        a = find_edges(spread, threshold=0.4, backend="numpy")
        b = find_edges(spread, threshold=0.4, backend="numpy")
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])

    @pytest.mark.parametrize("chunk", [1, 3, 7, 1000])
    def test_chunk_size_does_not_change_results(self, chunk, spread):
        """Chunking is a memory strategy; it must not affect the answer."""
        reference = find_edges(spread, threshold=0.6, backend="numpy", chunk=10_000)
        result = find_edges(spread, threshold=0.6, backend="numpy", chunk=chunk)
        assert np.array_equal(result[1], reference[1])
        assert np.array_equal(result[2], reference[2])
        assert np.allclose(result[0], reference[0], atol=1e-5)


class TestEdgeCases:
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_no_qualifying_pairs_returns_empty_arrays(self, backend):
        X = np.stack([unit(0, dim=4), unit(90, dim=4), unit(180, dim=4)])
        sims, rows, cols = find_edges(X, threshold=0.99, backend=backend)
        assert len(sims) == len(rows) == len(cols) == 0
        assert rows.dtype == np.int64

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_single_row_has_no_edges(self, backend):
        X = np.ones((1, 4), dtype=np.float32)
        X = X / np.linalg.norm(X)
        assert len(find_edges(X, threshold=0.0, backend=backend)[0]) == 0

    def test_auto_backend_resolves_to_something_installed(self):
        assert resolve_backend("auto") in {"torch", "numpy"}

    def test_unknown_backend_is_rejected(self):
        X = np.eye(2, dtype=np.float32)
        with pytest.raises(ValueError, match="backend"):
            find_edges(X, threshold=0.5, backend="nonsense")


@pytest.mark.skipif("torch" not in BACKENDS, reason="torch backend unavailable")
class TestTorchDevice:
    def test_cpu_and_cuda_agree(self, spread):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        cpu = find_edges(spread, threshold=0.6, backend="torch",
                         device=torch.device("cpu"))
        gpu = find_edges(spread, threshold=0.6, backend="torch",
                         device=torch.device("cuda"))
        assert np.array_equal(cpu[1], gpu[1])
        assert np.array_equal(cpu[2], gpu[2])
        assert np.allclose(cpu[0], gpu[0], atol=1e-5)
