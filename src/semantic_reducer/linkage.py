"""Cluster formation: union-find agglomeration with a linkage-strictness knob.

This is the heart of the method. Vocabulary reduction by embedding similarity
has one dominant failure mode -- *transitive semantic drift*. Plain connected
components use single-link semantics, so A~B and B~C place A and C in the same
class even when A and C are unrelated. Chains of such merges can swallow an
entire vocabulary and silently destroy meaning.

The linkage parameter lambda controls exactly that. Two clusters merge only when
at least a lambda fraction of the pairs across them reach tau:

* ``lambda = 0`` -- single linkage; merge on any qualifying edge. Maximum
  compression, no guarantee at all.
* ``lambda = 1`` -- complete linkage; every cross pair must reach tau.

Complete linkage carries a provable bound. Singletons trivially have all
internal pairs at or above tau. If two clusters that each satisfy the property
merge only when every cross pair also reaches tau, the union satisfies it too.
By induction every cluster is a clique in the tau-graph, so for any two types
u, v in a cluster, ``cos(u, v) >= tau`` and the cluster diameter is at most
``1 - tau``. Drift is therefore bounded by the threshold itself rather than left
to chance -- which is what makes tau a meaningful parameter instead of a knob
whose effect depends on the shape of the corpus.
"""

from __future__ import annotations

import numpy as np

__all__ = ["UnionFind", "agglomerate", "ClusterResult", "min_internal_similarity"]


class UnionFind:
    """Disjoint-set forest with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:           # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the sets holding ``a`` and ``b``; True if they were distinct."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


class ClusterResult:
    """Components plus a record of what was refused and why."""

    def __init__(self, components, blocked_linkage=0, blocked_size=0, blocked_protected=0):
        self.components: list[list[int]] = components
        self.blocked_linkage = blocked_linkage
        self.blocked_size = blocked_size
        self.blocked_protected = blocked_protected

    @property
    def n_clusters(self) -> int:
        return len(self.components)

    def as_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "merges_blocked_by_linkage": self.blocked_linkage,
            "merges_blocked_by_size_cap": self.blocked_size,
            "merges_blocked_by_protection": self.blocked_protected,
        }


def agglomerate(
    X: np.ndarray,
    sims: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    threshold: float,
    linkage: float = 1.0,
    max_cluster_size: int | None = None,
    mergeable: np.ndarray | None = None,
    cannot_link: dict[int, set[int]] | None = None,
) -> ClusterResult:
    """Grow clusters from the strongest edge down, honouring the linkage rule.

    Edges are consumed in descending similarity so the most reliable merges
    commit first and the outcome does not depend on input ordering.

    Args:
        X: ``(n, dim)`` L2-normalized type vectors.
        sims, rows, cols: edges sorted by descending similarity (see
            :func:`semantic_reducer.neighbors.find_edges`).
        threshold: tau, the same cutoff used to build the edges.
        linkage: lambda in [0, 1]; see the module docstring.
        max_cluster_size: refuse merges producing a cluster larger than this.
        mergeable: optional boolean mask; False types stay singletons.
        cannot_link: optional index -> forbidden indices, enforced at the
            COMPONENT level so a forbidden pair cannot be united even
            transitively through a chain.

    Returns:
        :class:`ClusterResult` whose ``components`` are sorted index lists.
    """
    n = X.shape[0]
    uf = UnionFind(n)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    forbidden: dict[int, set[int]] = {i: set() for i in range(n)}
    if cannot_link:
        for i, others in cannot_link.items():
            if 0 <= i < n:
                forbidden[i].update(o for o in others if 0 <= o < n)

    blocked_linkage = blocked_size = blocked_protected = 0

    for sim, i, j in zip(sims, rows, cols):
        i, j = int(i), int(j)

        if mergeable is not None and not (mergeable[i] and mergeable[j]):
            blocked_protected += 1
            continue

        ra, rb = uf.find(i), uf.find(j)
        if ra == rb:
            continue

        A, B = members[ra], members[rb]

        if max_cluster_size is not None and len(A) + len(B) > max_cluster_size:
            blocked_size += 1
            continue

        # Component-level cannot-link: would this union put a forbidden pair
        # into the same class, whether directly or through the chain?
        if cannot_link and (forbidden[ra] & set(B) or forbidden[rb] & set(A)):
            blocked_protected += 1
            continue

        if linkage > 0.0 and not _linkage_satisfied(X, A, B, threshold, linkage):
            blocked_linkage += 1
            continue

        uf.union(i, j)
        root = uf.find(i)
        other = rb if root == ra else ra
        members[root] = members[root] + members.pop(other)
        forbidden[root] |= forbidden.pop(other)

    components = [sorted(m) for m in members.values()]
    components.sort(key=lambda c: c[0])          # deterministic ordering
    return ClusterResult(components, blocked_linkage, blocked_size, blocked_protected)


def _linkage_satisfied(X, A, B, threshold, linkage) -> bool:
    """Does the fraction of cross-pairs reaching tau meet the linkage rule?"""
    S = X[A] @ X[B].T
    if linkage >= 1.0:
        # Complete linkage: one failing pair is enough to refuse the merge.
        return bool(S.min() >= threshold)
    return bool((S >= threshold).mean() >= linkage)


def min_internal_similarity(
    X: np.ndarray,
    component: list[int],
    max_exact: int = 256,
    rng: np.random.Generator | None = None,
) -> float:
    """Smallest cosine between any two members -- the cluster's tightness.

    With ``linkage=1.0`` this is guaranteed to be at or above tau; the gap
    between this and tau under looser linkage is precisely the drift the
    parameter is there to control.

    Clusters larger than ``max_exact`` are estimated from a random subset, since
    the exact computation is quadratic. Singletons return 1.0.
    """
    if len(component) < 2:
        return 1.0
    idx = component
    if len(idx) > max_exact:
        rng = rng or np.random.default_rng(0)
        idx = sorted(rng.choice(idx, size=max_exact, replace=False).tolist())
    V = X[idx]
    S = V @ V.T
    np.fill_diagonal(S, np.inf)
    return float(S.min())
