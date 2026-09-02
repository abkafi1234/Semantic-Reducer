"""Tests for anisotropy correction and normalization.

The threshold only means something if unrelated words are actually far apart
after correction. That is what these pin down.
"""

import numpy as np
import pytest

from semantic_reducer import correct_anisotropy, l2_normalize, mean_offdiagonal_cosine


def anisotropic_fixture(n=200, dim=32, scale=0.3, offset=5.0, seed=1):
    """Simulated Transformer geometry: a large shared direction plus small noise."""
    rng = np.random.default_rng(seed)
    return (np.ones(dim, dtype=np.float32) * offset
            + rng.normal(scale=scale, size=(n, dim)).astype(np.float32))


class TestCorrectAnisotropy:
    def test_mean_centering_removes_the_common_component(self):
        X = anisotropic_fixture()
        assert np.allclose(correct_anisotropy(X, n_abtt=0).mean(axis=0), 0.0, atol=1e-4)

    def test_correction_collapses_the_cone(self):
        """Without this, a cosine threshold measures the cone rather than meaning."""
        X = anisotropic_fixture()
        before = mean_offdiagonal_cosine(X)
        after = mean_offdiagonal_cosine(correct_anisotropy(X, n_abtt=2))
        assert before > 0.95, "fixture should reproduce the anisotropy cone"
        assert after < 0.2, f"correction failed to decorrelate: {after:.3f}"

    def test_abtt_removes_the_requested_directions(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 12)).astype(np.float32)
        out = correct_anisotropy(X, n_abtt=3)
        assert np.linalg.matrix_rank(out, tol=1e-4) <= 12 - 3

    def test_center_false_leaves_geometry_untouched(self):
        """The ablation baseline must be a genuine no-op."""
        X = anisotropic_fixture(n=20)
        out = correct_anisotropy(X, n_abtt=0, center=False)
        assert np.array_equal(out, X.astype(np.float32))

    def test_more_abtt_components_decorrelate_at_least_as_much(self):
        X = anisotropic_fixture()
        scores = [
            mean_offdiagonal_cosine(correct_anisotropy(X, n_abtt=k))
            for k in (0, 1, 2, 3)
        ]
        assert scores[0] >= scores[1] - 1e-3 >= scores[2] - 2e-3

    def test_does_not_mutate_its_input(self):
        X = anisotropic_fixture(n=10)
        original = X.copy()
        correct_anisotropy(X, n_abtt=1)
        assert np.array_equal(X, original)

    def test_handles_fewer_rows_than_components(self):
        X = np.ones((2, 5), dtype=np.float32)
        correct_anisotropy(X, n_abtt=10)   # must not raise

    def test_returns_float32(self):
        X = np.ones((10, 4), dtype=np.float64)
        assert correct_anisotropy(X, n_abtt=1).dtype == np.float32


class TestL2Normalize:
    def test_rows_become_unit_norm(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 6)).astype(np.float32) * 10
        assert np.allclose(np.linalg.norm(l2_normalize(X), axis=1), 1.0, atol=1e-5)

    def test_zero_row_yields_zeros_not_nan(self):
        """Anisotropy correction can collapse a row to zero; NaNs would poison search."""
        X = np.zeros((2, 4), dtype=np.float32)
        X[1, 0] = 3.0
        out = l2_normalize(X)
        assert not np.isnan(out).any()
        assert np.allclose(out[0], 0.0)
        assert np.allclose(np.linalg.norm(out[1]), 1.0)

    def test_direction_is_preserved(self):
        X = np.array([[3.0, 4.0]], dtype=np.float32)
        assert np.allclose(l2_normalize(X), [[0.6, 0.8]], atol=1e-6)


class TestMeanOffdiagonalCosine:
    def test_identical_vectors_score_one(self):
        X = np.tile(np.array([1.0, 0.0], dtype=np.float32), (10, 1))
        assert mean_offdiagonal_cosine(X) == pytest.approx(1.0, abs=1e-5)

    def test_antipodal_pair_scores_minus_one(self):
        X = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        assert mean_offdiagonal_cosine(X) == pytest.approx(-1.0, abs=1e-5)

    def test_single_row_is_undefined(self):
        assert np.isnan(mean_offdiagonal_cosine(np.ones((1, 3), dtype=np.float32)))

    def test_sampling_is_deterministic(self):
        X = anisotropic_fixture(n=500)
        assert mean_offdiagonal_cosine(X, sample=100, seed=0) == \
               mean_offdiagonal_cosine(X, sample=100, seed=0)
