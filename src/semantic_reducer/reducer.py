"""Corpus-grounded semantic vocabulary reduction.

Pipeline
--------
1. Encode full sentences and average each word type's *contextual* occurrence
   vectors into one sense-averaged type vector (:mod:`.encoder`).
2. Correct anisotropy, so a cosine threshold means something (:mod:`.geometry`).
3. Retrieve every pair at or above tau, exactly (:mod:`.neighbors`).
4. Agglomerate under a linkage rule that bounds intra-cluster drift
   (:mod:`.linkage`).
5. Collapse each cluster to its most frequent member and compile a plain dict.

Inference is then a dictionary lookup: O(1) per token, with no encoder in the
loop, and idempotent because every member of a class -- the representative
included -- maps to that representative.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from .config import ReducerConfig
from .encoder import ContextualEncoder, describe_device, tokenize
from .geometry import (apply_anisotropy_correction, fit_anisotropy_correction,
                       l2_normalize, mean_offdiagonal_cosine)
from .linkage import agglomerate, min_internal_similarity
from .neighbors import find_edges
from .protect import build_mergeable_mask, protection_reasons

__all__ = ["SemanticReducer"]

FORMAT_VERSION = 3


class SemanticReducer:
    """Reduce a vocabulary to semantic equivalence classes.

    Example::

        from semantic_reducer import SemanticReducer

        reducer = SemanticReducer(threshold=0.6, linkage=1.0)
        reducer.fit(corpus)
        reducer.reduce("The rapid fox leaped.")
        reducer.save("my_corpus.json")
    """

    def __init__(self, config: ReducerConfig | None = None, **overrides):
        """
        Args:
            config: a full :class:`ReducerConfig`.
            **overrides: individual config fields, e.g. ``threshold=0.55``.
        """
        if config is not None and overrides:
            raise TypeError("pass either a ReducerConfig or keyword overrides, not both")
        self.config = config or ReducerConfig(**overrides)

        self.vocab: list[str] = []
        self.counts: Counter = Counter()
        self.concentration: np.ndarray = np.empty(0, dtype=np.float32)
        self.vectors: np.ndarray | None = None
        self.reduction_map: dict[str, str] = {}
        self.clusters: dict[str, list[str]] = {}

        self._encoder: ContextualEncoder | None = None
        self._raw_type_vectors: np.ndarray | None = None
        self._aniso_mean: np.ndarray | None = None
        self._aniso_pcs: np.ndarray | None = None
        self._cluster_result = None
        self._geometry: dict = {}
        self._protection: dict[str, str] = {}
        self._cannot_link_applied = 0
        self._n_observed_types = 0
        self._truncated = 0
        self._unencodable = 0
        # Diagnostics computed at fit time and persisted, so a system restored
        # by load() can still report them without the type vectors.
        self._stored_diagnostics: dict = {}

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #
    def fit(self, corpus, progress: bool = True) -> "SemanticReducer":
        """Learn equivalence classes from a corpus.

        Args:
            corpus: iterable of raw strings. Sentences or short documents; each
                is encoded whole so occurrence vectors are genuinely contextual.
            progress: show a progress bar during encoding.
        """
        cfg = self.config
        corpus = list(corpus)
        self._encoder = ContextualEncoder(cfg)
        if cfg.finetune:
            finetune_result = self._encoder.finetune_on_corpus(corpus, progress=progress)
            self._stored_diagnostics["finetune"] = finetune_result.to_dict()
        stats = self._encoder.encode_corpus(corpus, progress=progress)

        self._n_observed_types = len(stats.counts)
        self._truncated = stats.truncated
        self._unencodable = stats.unencodable

        # ---- vocabulary (sorted, so nothing depends on dict insertion order)
        self.vocab = sorted(w for w, c in stats.counts.items() if c >= cfg.min_count)
        if not self.vocab:
            raise ValueError(
                f"vocabulary is empty after the min_count>={cfg.min_count} filter "
                f"({self._n_observed_types} types were observed). Lower min_count "
                f"or supply a larger corpus."
            )
        self.counts = Counter({w: stats.counts[w] for w in self.vocab})

        # ---- type vectors and the concentration (polysemy) statistic
        raw = np.zeros((len(self.vocab), stats.dim), dtype=np.float32)
        conc = np.zeros(len(self.vocab), dtype=np.float32)
        for i, word in enumerate(self.vocab):
            raw[i] = stats.type_vector(word)
            conc[i] = stats.concentration(word)
        self.concentration = conc

        self._fit_from_type_vectors(raw)
        return self

    def _fit_from_type_vectors(self, raw: np.ndarray) -> "SemanticReducer":
        """Geometry, search, clustering, and map compilation.

        Split out from :meth:`fit` so the clustering half can be exercised with
        injected vectors, without an encoder.
        """
        cfg = self.config

        # Retained so a threshold or linkage sweep can re-cluster the same
        # corpus without paying to encode it again, and so the anisotropy
        # ablation can compare corrected against uncorrected geometry directly.
        self._raw_type_vectors = raw

        # ---- geometry
        before = mean_offdiagonal_cosine(raw, seed=cfg.seed)
        if cfg.anisotropy:
            if len(self.vocab) < 50:
                warnings.warn(
                    f"anisotropy correction is being applied to only "
                    f"{len(self.vocab)} types. It is estimated from the population "
                    f"of type vectors and is degenerate on tiny vocabularies, so "
                    f"the resulting cosines -- and therefore the threshold -- are "
                    f"not meaningful. Use a larger corpus or set anisotropy=False.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            corrected, aniso_mean, aniso_pcs = fit_anisotropy_correction(raw, n_abtt=cfg.n_abtt)
        else:
            corrected = raw
            aniso_mean = np.zeros((1, raw.shape[1]), dtype=np.float32)
            aniso_pcs = np.zeros((0, raw.shape[1]), dtype=np.float32)
        # Stored so assign_oov() can put a freshly-encoded OOV query through the
        # SAME correction before comparing it against `self.vectors` -- without
        # this, the query stays in the raw (uncorrected) coordinate system while
        # the corpus vectors are corrected, and the resulting similarities do not
        # mean what a threshold fitted on corrected-space cosines assumes.
        self._aniso_mean = aniso_mean
        self._aniso_pcs = aniso_pcs
        self.vectors = l2_normalize(corrected)
        self._geometry = {
            "mean_cosine_before_correction": round(before, 4),
            "mean_cosine_after_correction": round(
                mean_offdiagonal_cosine(self.vectors, seed=cfg.seed), 4
            ),
            "anisotropy_correction": cfg.anisotropy,
            "n_abtt": cfg.n_abtt if cfg.anisotropy else 0,
        }

        # ---- which types may merge at all
        self._protection = protection_reasons(
            self.vocab,
            protect=cfg.protect,
            protect_punctuation=cfg.protect_punctuation,
            protect_numerals=cfg.protect_numerals,
            protect_pattern=cfg.protect_pattern,
            protect_capitalized=cfg.protect_capitalized,
            concentration=self.concentration,
            min_concentration=cfg.min_concentration,
        )
        mergeable = build_mergeable_mask(
            self.vocab,
            protect=cfg.protect,
            protect_punctuation=cfg.protect_punctuation,
            protect_numerals=cfg.protect_numerals,
            protect_pattern=cfg.protect_pattern,
            protect_capitalized=cfg.protect_capitalized,
            concentration=self.concentration,
            min_concentration=cfg.min_concentration,
        )

        # ---- exact neighbour retrieval, then linkage-constrained agglomeration
        device = self._encoder.device if self._encoder is not None else None
        sims, rows, cols = find_edges(
            self.vectors,
            threshold=cfg.threshold,
            backend=cfg.backend,
            chunk=cfg.search_chunk,
            device=device,
        )
        self._cluster_result = agglomerate(
            self.vectors,
            sims, rows, cols,
            threshold=cfg.threshold,
            linkage=cfg.linkage,
            max_cluster_size=cfg.max_cluster_size,
            mergeable=mergeable,
            cannot_link=self._cannot_link_indices(),
        )

        self._compile_map()
        return self

    def _cannot_link_indices(self) -> dict[int, set[int]] | None:
        """Translate the configured word pairs into vocabulary indices.

        Symmetrized, so naming a pair in one direction is enough, and pairs
        mentioning words outside the vocabulary are dropped (they cannot be
        merged in any case). The count of applicable pairs is recorded for
        :meth:`constraint_report`.
        """
        pairs = self.config.cannot_link
        self._cannot_link_applied = 0
        if not pairs:
            return None

        position = {word: i for i, word in enumerate(self.vocab)}
        forbidden: dict[int, set[int]] = {}
        for word, others in pairs.items():
            i = position.get(word)
            if i is None:
                continue
            for other in others:
                j = position.get(other)
                if j is None or j == i:
                    continue
                forbidden.setdefault(i, set()).add(j)
                forbidden.setdefault(j, set()).add(i)
                self._cannot_link_applied += 1

        if not forbidden:
            warnings.warn(
                f"cannot_link names {len(pairs)} word(s), but none of the pairs "
                f"survive: both members must be in the vocabulary after the "
                f"min_count>={self.config.min_count} filter. The constraint had "
                f"no effect.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return forbidden

    def constraint_report(self) -> dict:
        """What the cannot-link constraint was asked to do, and what it did.

        Both counts are of unordered pairs. ``pairs_applicable`` is lower than
        ``pairs_configured`` when a named word is missing from the vocabulary,
        usually because ``min_count`` dropped it.
        """
        return {
            "pairs_configured": sum(len(v) for v in self.config.cannot_link.values()),
            "pairs_applicable": getattr(self, "_cannot_link_applied", 0),
            "merges_blocked_by_protection": (
                self._cluster_result.blocked_protected if self._cluster_result else 0
            ),
        }

    def _compile_map(self) -> None:
        """Choose a representative per cluster and flatten to a lookup dict."""
        self.reduction_map = {}
        self.clusters = {}

        for component in self._cluster_result.components:
            # Representative: highest corpus frequency, ties broken
            # lexicographically. Deterministic, and free of any orthographic
            # criterion such as string length.
            ordered = sorted(
                component,
                key=lambda m: (-self.counts[self.vocab[m]], self.vocab[m]),
            )
            rep = self.vocab[ordered[0]]
            self.clusters[rep] = [self.vocab[m] for m in ordered]
            for m in ordered:
                self.reduction_map[self.vocab[m]] = rep

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def reduce(self, text: str) -> str:
        """Reduce one string. O(1) per token; no encoder involved."""
        toks = tokenize(text, self.config.lowercase)
        return " ".join(self.reduction_map.get(t, t) for t in toks)

    def reduce_batch(self, texts) -> list[str]:
        return [self.reduce(t) for t in texts]

    def assign_oov(self, words: list[str]) -> dict[str, str]:
        """Map unseen words onto existing classes by encoding them directly.

        Out-of-vocabulary words pass through :meth:`reduce` unchanged by design,
        which keeps inference free of the encoder. This method is the opt-in
        alternative, and it has real costs: it loads the encoder, so inference is
        no longer O(1); and the words are encoded WITHOUT context, so their
        vectors are not comparable in kind to the corpus-derived type vectors
        that were averaged over many contexts. Use it deliberately.

        Measured cost of that gap: on a 200-word cross-corpus test, isolated
        encodings reached at most 0.40 cosine similarity to their OWN corpus
        type vector (median 0.29) -- well below a typical corpus-internal
        threshold. Lowering the threshold to compensate does not help: it
        surfaces a domain-shift hubness effect where a handful of corpus
        vectors, unremarkable in the corpus's own internal structure, absorb
        the majority of OOV queries regardless of meaning. Leaving
        out-of-vocabulary words unchanged (the default) is therefore the safer
        choice, not merely the cheaper one. Details:
        ``benchmark/reports/final/oov_demo/SUMMARY.md``.

        Returns:
            ``{word: representative}`` for words whose nearest class reaches tau.
            Words with no class above the threshold are omitted.
        """
        if self.vectors is None:
            raise RuntimeError(
                "assign_oov needs the type vectors, which are not kept by "
                "save(). Refit, or save with include_vectors=True."
            )
        if self._encoder is None:
            self._encoder = ContextualEncoder(self.config)

        unknown = [w for w in words if w not in self.reduction_map]
        if not unknown:
            return {}

        raw = self._encoder.encode_words(unknown)
        # Put the query through the SAME anisotropy correction the corpus
        # vectors were fitted with (see the comment in _fit_from_type_vectors).
        # Skipping this compares a raw, uncorrected query against corrected
        # corpus vectors -- two different coordinate systems -- and silently
        # suppresses every similarity, since correction is not a small nudge:
        # it re-centers on the corpus mean and removes the dominant shared
        # directions the raw vectors sit in.
        if self._aniso_mean is not None:
            raw = apply_anisotropy_correction(raw, self._aniso_mean, self._aniso_pcs)
        query = l2_normalize(raw)
        sims = query @ self.vectors.T

        assignments = {}
        for row, word in enumerate(unknown):
            best = int(np.argmax(sims[row]))
            if float(sims[row, best]) >= self.config.threshold:
                assignments[word] = self.reduction_map[self.vocab[best]]
        return assignments

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    def cluster_stats(self) -> dict:
        """Size and compression summary of the induced classes."""
        if not self.clusters:
            return {}
        sizes = np.array([len(m) for m in self.clusters.values()])
        merged = sizes[sizes > 1]
        return {
            "n_types": len(self.vocab),
            "n_clusters": len(self.clusters),
            "vocab_reduction_pct": round(100 * (1 - len(self.clusters) / len(self.vocab)), 2),
            "n_clusters_gt1": int((sizes > 1).sum()),
            "largest_cluster": int(sizes.max()),
            "mean_merged_cluster_size": round(float(merged.mean()), 2) if merged.size else 0.0,
            "types_protected": len(self._protection),
            **(self._cluster_result.as_dict() if self._cluster_result else {}),
        }

    def drift_report(self, max_exact: int = 256) -> dict:
        """How much semantic drift the clusters actually contain.

        ``min_internal_similarity`` is the smallest cosine between any two
        members of a cluster. Under ``linkage=1.0`` every cluster is a clique in
        the tau-graph, so this cannot fall below tau; the gap that opens under
        looser linkage is the drift the parameter exists to control.

        On a system restored by :meth:`load`, the report computed at fit time is
        returned instead: recomputing it needs the type vectors, which are not
        saved by default because they dominate the artifact's size.
        """
        if not self.clusters:
            return {}

        if self._cluster_result is None or self.vectors is None:
            stored = self._stored_diagnostics.get("drift")
            if stored is not None:
                return dict(stored)
            return {
                "unavailable": (
                    "drift cannot be recomputed on a loaded system without the "
                    "type vectors; save with include_vectors=True, or read the "
                    "report stored at fit time"
                )
            }

        rng = np.random.default_rng(self.config.seed)
        tightness = [
            min_internal_similarity(self.vectors, c, max_exact=max_exact, rng=rng)
            for c in self._cluster_result.components
            if len(c) > 1
        ]
        if not tightness:
            return {
                "threshold": self.config.threshold,
                "linkage": self.config.linkage,
                "n_merged_clusters": 0,
                "bound_holds": True,
            }
        tightness = np.array(tightness)
        tau = self.config.threshold
        return {
            "threshold": tau,
            "linkage": self.config.linkage,
            "n_merged_clusters": int(tightness.size),
            "min_internal_similarity": round(float(tightness.min()), 4),
            "mean_internal_similarity": round(float(tightness.mean()), 4),
            "worst_diameter": round(float(1 - tightness.min()), 4),
            "clusters_below_threshold": int((tightness < tau - 1e-6).sum()),
            "bound_holds": bool((tightness >= tau - 1e-6).all()),
        }

    def polysemy_report(self, top_k: int = 20) -> list[tuple[str, float]]:
        """The least consistently used types -- the likeliest polysemes.

        The score is the mean resultant length of the type's unit occurrence
        vectors, in [0, 1]. Low means the word's contexts pull its vector in many
        directions, so collapsing it to a single average discards a real
        distinction. Set ``ReducerConfig.min_concentration`` to keep such types
        out of merges entirely.
        """
        if not len(self.concentration):
            return []
        order = np.argsort(self.concentration)
        return [
            (self.vocab[i], round(float(self.concentration[i]), 4))
            for i in order[:top_k]
        ]

    def geometry_report(self) -> dict:
        """Mean off-diagonal cosine before and after anisotropy correction.

        Before correction this sits near 0.9 on Transformer embeddings, which is
        the cone, not meaning. A threshold applied at that point is not
        interpretable; the drop after correction is what makes tau usable.
        """
        return dict(self._geometry)

    def finetune_report(self) -> dict:
        """Per-epoch mean MLM loss from continued pretraining, if ``finetune=True``.

        Empty if fitting used the default frozen encoder.
        """
        return dict(self._stored_diagnostics.get("finetune", {}))

    def protection_report(self) -> dict[str, int]:
        """Count of protected types by the rule that protected them."""
        if not self._protection and self._stored_diagnostics.get("protection"):
            return dict(self._stored_diagnostics["protection"])
        counts: Counter = Counter(self._protection.values())
        return dict(sorted(counts.items()))

    def sample_merges(self, k: int = 30) -> list[tuple[str, str]]:
        """Up to k ``(word, representative)`` pairs where the word actually moved."""
        moved = [(w, r) for w, r in sorted(self.reduction_map.items()) if w != r]
        return moved[:k]

    def verify_guarantees(self) -> dict:
        """Check the properties the method claims, on this fitted map.

        Returns a dict of property -> bool. Every value must be True; the paper's
        propositions are exactly these checks.
        """
        idempotent = all(
            self.reduction_map[self.reduction_map[w]] == self.reduction_map[w]
            for w in self.reduction_map
        )
        reps_fixed = all(self.reduction_map.get(rep) == rep for rep in self.clusters)
        closed = all(
            self.reduction_map[m] == rep
            for rep, members in self.clusters.items()
            for m in members
        )
        results = {
            "idempotent": idempotent,
            "representatives_are_fixed_points": reps_fixed,
            "classes_are_closed": closed,
        }
        if self.config.linkage >= 1.0:
            drift = self.drift_report()
            # Do not claim the bound holds when it simply could not be checked.
            if "bound_holds" in drift:
                results["diameter_bound_holds"] = drift["bound_holds"]
            elif "unavailable" in drift:
                results["diameter_bound_holds"] = None
            else:
                results["diameter_bound_holds"] = True
        return results

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path, include_vectors: bool = False) -> Path:
        """Write the fitted system to JSON.

        JSON rather than pickle: the artifact is inspectable, diffable, safe to
        load from an untrusted source, and readable without this package -- which
        matters when it is the supplement to a paper.

        Args:
            path: destination ``.json`` path.
            include_vectors: also write ``<path>.vectors.npy``, needed only for
                :meth:`assign_oov`. Adds ``n_types * dim`` float32s.
        """
        if not self.reduction_map:
            raise RuntimeError("nothing to save: call fit() first")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "provenance": {
                "n_observed_types": self._n_observed_types,
                "truncated_occurrences": self._truncated,
                "unencodable_occurrences": getattr(self, "_unencodable", 0),
                "device": describe_device(self._encoder.device) if self._encoder else None,
                "geometry": self._geometry,
            },
            "vocab": self.vocab,
            "counts": {w: int(self.counts[w]) for w in self.vocab},
            "concentration": [round(float(c), 6) for c in self.concentration],
            "reduction_map": self.reduction_map,
            "clusters": self.clusters,
            "stats": self.cluster_stats(),
            # Persisted so a reloaded system can still report them: recomputing
            # drift needs the type vectors, which are not saved by default.
            "diagnostics": {
                "drift": self.drift_report(),
                "protection": self.protection_report(),
                "guarantees": self.verify_guarantees(),
                "most_polysemous": self.polysemy_report(20),
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if include_vectors:
            if self.vectors is None:
                raise RuntimeError("no type vectors are held in memory")
            np.save(path.with_suffix(".vectors.npy"), self.vectors)
            # The anisotropy transform travels WITH the vectors: assign_oov()
            # needs both together, and one without the other silently produces
            # the mismatched-coordinate-system bug described in assign_oov's
            # docstring. Persisted even when empty (anisotropy=False fits),
            # so load() can always tell "no correction was used" apart from
            # "the sidecar is simply missing."
            mean = self._aniso_mean if self._aniso_mean is not None \
                else np.zeros((1, self.vectors.shape[1]), dtype=np.float32)
            pcs = self._aniso_pcs if self._aniso_pcs is not None \
                else np.zeros((0, self.vectors.shape[1]), dtype=np.float32)
            np.savez(path.with_suffix(".aniso.npz"), mean=mean, pcs=pcs)
        return path

    @classmethod
    def load(cls, path) -> "SemanticReducer":
        """Load a system written by :meth:`save`."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        version = payload.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{path} has format_version {version!r}, but this build of "
                f"semantic-reducer writes and reads version {FORMAT_VERSION}. "
                f"Refit the corpus with this version."
            )

        reducer = cls(config=ReducerConfig.from_dict(payload["config"]))
        reducer.vocab = payload["vocab"]
        reducer.counts = Counter(payload["counts"])
        reducer.concentration = np.array(payload["concentration"], dtype=np.float32)
        reducer.reduction_map = payload["reduction_map"]
        reducer.clusters = payload["clusters"]
        reducer._geometry = payload.get("provenance", {}).get("geometry", {})
        reducer._stored_diagnostics = payload.get("diagnostics", {})

        vector_path = path.with_suffix(".vectors.npy")
        if vector_path.exists():
            reducer.vectors = np.load(vector_path)
            aniso_path = path.with_suffix(".aniso.npz")
            if aniso_path.exists():
                with np.load(aniso_path) as stored:
                    reducer._aniso_mean = stored["mean"]
                    reducer._aniso_pcs = stored["pcs"]
        return reducer

    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        if not self.clusters:
            return f"SemanticReducer(model={self.config.model_name!r}, unfitted)"
        return (
            f"SemanticReducer(model={self.config.model_name!r}, "
            f"tau={self.config.threshold}, lambda={self.config.linkage}, "
            f"{len(self.vocab)} types -> {len(self.clusters)} classes)"
        )
