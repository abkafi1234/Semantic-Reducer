"""Tests for cluster formation -- the method's central guarantees.

The headline case is a three-word chain: A~B and B~C both reach tau while A~C
does not. Single linkage merges all three and puts unrelated words in one class;
complete linkage refuses and holds the diameter bound. That contrast is the
whole argument for the linkage parameter, so it is pinned here.
"""

import numpy as np
import pytest

from conftest import build_from_vectors, name, unit

from semantic_reducer import UnionFind, agglomerate, find_edges, min_internal_similarity


# --------------------------------------------------------------------------- #
class TestUnionFind:
    def test_starts_fully_disjoint(self):
        uf = UnionFind(5)
        assert len({uf.find(i) for i in range(5)}) == 5

    def test_union_is_transitive(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)
        assert uf.find(0) != uf.find(3)

    def test_union_reports_whether_it_merged(self):
        uf = UnionFind(3)
        assert uf.union(0, 1) is True
        assert uf.union(0, 1) is False

    def test_find_is_stable_under_path_compression(self):
        uf = UnionFind(6)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            uf.union(a, b)
        roots = [uf.find(i) for i in range(5)]
        assert len(set(roots)) == 1
        assert [uf.find(i) for i in range(5)] == roots


# --------------------------------------------------------------------------- #
class TestChainingAndLinkage:
    """A~B~C with A and C unrelated: the drift scenario the method must handle."""

    TAU = 0.7

    @staticmethod
    def chain_edges():
        # 0deg / 40deg / 80deg: cos(40)=0.766 >= tau, cos(80)=0.174 < tau.
        X = np.stack([unit(0), unit(40), unit(80)])
        sims, rows, cols = find_edges(X, threshold=TestChainingAndLinkage.TAU, backend="numpy")
        return X, sims, rows, cols

    def test_fixture_geometry_is_a_genuine_chain(self):
        X, sims, rows, cols = self.chain_edges()
        pairs = {(int(r), int(c)) for r, c in zip(rows, cols)}
        assert pairs == {(0, 1), (1, 2)}, "A~C must not be an edge"

    def test_single_linkage_chains_unrelated_words_together(self):
        X, sims, rows, cols = self.chain_edges()
        result = agglomerate(X, sims, rows, cols, threshold=self.TAU, linkage=0.0)
        assert result.n_clusters == 1, "single linkage should merge the whole chain"

    def test_complete_linkage_refuses_the_drifting_merge(self):
        X, sims, rows, cols = self.chain_edges()
        result = agglomerate(X, sims, rows, cols, threshold=self.TAU, linkage=1.0)
        assert result.n_clusters == 2
        assert result.blocked_linkage == 1
        assert [0, 1] in result.components and [2] in result.components

    def test_single_linkage_violates_the_diameter_bound(self):
        """The failure complete linkage exists to prevent, stated numerically."""
        X, sims, rows, cols = self.chain_edges()
        result = agglomerate(X, sims, rows, cols, threshold=self.TAU, linkage=0.0)
        worst = min(min_internal_similarity(X, c) for c in result.components)
        assert worst < self.TAU, "single linkage should admit sub-threshold pairs"
        assert worst == pytest.approx(np.cos(np.deg2rad(80)), abs=1e-5)

    def test_complete_linkage_holds_the_diameter_bound(self):
        X, sims, rows, cols = self.chain_edges()
        result = agglomerate(X, sims, rows, cols, threshold=self.TAU, linkage=1.0)
        for component in result.components:
            assert min_internal_similarity(X, component) >= self.TAU

    @pytest.mark.parametrize(
        "lam,expected_clusters",
        [
            (0.0, 1),    # any edge suffices
            (0.5, 1),    # 1 of 2 cross-pairs qualifies -> exactly meets 0.5
            (0.6, 2),    # 0.5 < 0.6 -> refused
            (1.0, 2),    # all cross-pairs required -> refused
        ],
    )
    def test_lambda_interpolates_between_single_and_complete(self, lam, expected_clusters):
        """Merging {A,B} with {C}: cross-pairs are (A,C) below tau and (B,C) above."""
        X, sims, rows, cols = self.chain_edges()
        result = agglomerate(X, sims, rows, cols, threshold=self.TAU, linkage=lam)
        assert result.n_clusters == expected_clusters


# --------------------------------------------------------------------------- #
class TestCompleteLinkageInvariant:
    """The clique property must hold on a larger, denser graph, not just a chain."""

    @staticmethod
    def dense_fixture(n=60, spread=120.0, seed=3):
        rng = np.random.default_rng(seed)
        angles = np.sort(rng.uniform(0, spread, size=n))
        return np.stack([unit(a, dim=4) for a in angles])

    @pytest.mark.parametrize("tau", [0.5, 0.7, 0.9, 0.95])
    def test_every_cluster_is_a_clique_in_the_tau_graph(self, tau):
        X = self.dense_fixture()
        sims, rows, cols = find_edges(X, threshold=tau, backend="numpy")
        result = agglomerate(X, sims, rows, cols, threshold=tau, linkage=1.0)
        for component in result.components:
            if len(component) > 1:
                S = X[component] @ X[component].T
                assert S.min() >= tau - 1e-6, (
                    f"cluster {component} contains a pair below tau={tau}"
                )

    @pytest.mark.parametrize("tau", [0.5, 0.7, 0.9])
    def test_single_linkage_compresses_at_least_as_hard(self, tau):
        """Compression and the bound trade off; lambda is the dial between them."""
        X = self.dense_fixture()
        sims, rows, cols = find_edges(X, threshold=tau, backend="numpy")
        loose = agglomerate(X, sims, rows, cols, threshold=tau, linkage=0.0)
        strict = agglomerate(X, sims, rows, cols, threshold=tau, linkage=1.0)
        assert loose.n_clusters <= strict.n_clusters

    def test_every_type_appears_in_exactly_one_component(self):
        X = self.dense_fixture()
        sims, rows, cols = find_edges(X, threshold=0.8, backend="numpy")
        result = agglomerate(X, sims, rows, cols, threshold=0.8, linkage=1.0)
        seen = [i for c in result.components for i in c]
        assert sorted(seen) == list(range(X.shape[0]))


# --------------------------------------------------------------------------- #
class TestConstraints:
    @staticmethod
    def pair():
        X = np.stack([unit(0), unit(5), unit(10)])
        sims, rows, cols = find_edges(X, threshold=0.9, backend="numpy")
        return X, sims, rows, cols

    def test_max_cluster_size_caps_growth(self):
        X, sims, rows, cols = self.pair()
        unrestricted = agglomerate(X, sims, rows, cols, threshold=0.9, linkage=0.0)
        assert unrestricted.n_clusters == 1

        capped = agglomerate(X, sims, rows, cols, threshold=0.9, linkage=0.0,
                             max_cluster_size=2)
        assert max(len(c) for c in capped.components) <= 2
        assert capped.blocked_size >= 1

    def test_unmergeable_types_stay_singletons(self):
        X, sims, rows, cols = self.pair()
        mergeable = np.array([True, False, True])
        result = agglomerate(X, sims, rows, cols, threshold=0.9, linkage=0.0,
                             mergeable=mergeable)
        assert [1] in result.components, "a protected type must not be absorbed"
        assert result.blocked_protected >= 1

    def test_cannot_link_is_enforced_through_chains(self):
        """A and C are forbidden partners, so B must not unite them transitively."""
        X = np.stack([unit(0), unit(5), unit(10)])
        sims, rows, cols = find_edges(X, threshold=0.9, backend="numpy")
        result = agglomerate(
            X, sims, rows, cols, threshold=0.9, linkage=0.0,
            cannot_link={0: {2}, 2: {0}},
        )
        groups = {frozenset(c) for c in result.components}
        for group in groups:
            assert not ({0, 2} <= group), "forbidden pair ended up in one class"

    def test_no_edges_leaves_every_type_alone(self):
        X = np.stack([unit(0), unit(90), unit(180)])
        sims, rows, cols = find_edges(X, threshold=0.9, backend="numpy")
        result = agglomerate(X, sims, rows, cols, threshold=0.9, linkage=1.0)
        assert result.n_clusters == 3


# --------------------------------------------------------------------------- #
class TestMinInternalSimilarity:
    def test_singleton_is_maximally_tight(self):
        X = np.stack([unit(0), unit(40)])
        assert min_internal_similarity(X, [0]) == 1.0

    def test_reports_the_worst_pair(self):
        X = np.stack([unit(0), unit(40), unit(80)])
        expected = float(np.cos(np.deg2rad(80)))
        assert min_internal_similarity(X, [0, 1, 2]) == pytest.approx(expected, abs=1e-5)

    def test_large_clusters_fall_back_to_sampling(self):
        X = np.stack([unit(a, dim=4) for a in np.linspace(0, 30, 100)])
        rng = np.random.default_rng(0)
        sampled = min_internal_similarity(X, list(range(100)), max_exact=10, rng=rng)
        exact = min_internal_similarity(X, list(range(100)), max_exact=1000)
        # A subset can only ever overestimate tightness.
        assert sampled >= exact - 1e-6


# --------------------------------------------------------------------------- #
class TestDeterminism:
    def test_same_input_gives_identical_components(self):
        X = TestCompleteLinkageInvariant.dense_fixture()
        sims, rows, cols = find_edges(X, threshold=0.8, backend="numpy")
        first = agglomerate(X, sims, rows, cols, threshold=0.8, linkage=1.0).components
        second = agglomerate(X, sims, rows, cols, threshold=0.8, linkage=1.0).components
        assert first == second

    def test_reduction_map_is_order_independent(self):
        """Vocabulary is sorted internally, so dict insertion order cannot matter."""
        vectors = {name(i): unit(i * 4, dim=4) for i in range(20)}
        counts = {w: (i * 7) % 11 + 1 for i, w in enumerate(vectors)}
        forward = build_from_vectors(vectors, counts, threshold=0.85)
        backward = build_from_vectors(
            dict(reversed(list(vectors.items()))), counts, threshold=0.85
        )
        assert forward.reduction_map == backward.reduction_map
        # Guard against the assertion being vacuous: merges must actually happen.
        assert len(forward.clusters) < len(forward.vocab)
