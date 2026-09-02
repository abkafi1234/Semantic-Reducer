"""Tests for the SemanticReducer API: guarantees, diagnostics, persistence."""

import json

import numpy as np
import pytest

from conftest import build_from_vectors, cosine_matrix, name, unit

from semantic_reducer import ReducerConfig, SemanticReducer


# --------------------------------------------------------------------------- #
class TestGuarantees:
    @pytest.fixture
    def chain(self):
        """A~B~C with A and C unrelated, under single linkage.

        Deliberately the permissive setting: even when drift is allowed, the map
        must still be a well-formed idempotent canonicalization.
        """
        vectors = {"A": unit(0), "B": unit(40), "C": unit(80)}
        counts = {"A": 10, "B": 20, "C": 30}
        return build_from_vectors(vectors, counts, threshold=0.7, linkage=0.0)

    def test_map_is_idempotent(self, chain):
        for word in chain.vocab:
            once = chain.reduction_map[word]
            assert chain.reduction_map[once] == once, f"{word} -> {once} -> ..."

    def test_reduce_text_is_idempotent(self, chain):
        once = chain.reduce("A B C")
        assert chain.reduce(once) == once

    def test_classes_are_closed(self, chain):
        for rep, members in chain.clusters.items():
            for m in members:
                assert chain.reduction_map[m] == rep

    def test_verify_guarantees_passes(self, chain):
        assert all(chain.verify_guarantees().values())

    def test_verify_guarantees_checks_the_bound_under_complete_linkage(self):
        vectors = {"A": unit(0), "B": unit(40), "C": unit(80)}
        counts = {"A": 10, "B": 20, "C": 30}
        reducer = build_from_vectors(vectors, counts, threshold=0.7, linkage=1.0)
        results = reducer.verify_guarantees()
        assert results["diameter_bound_holds"] is True
        assert all(results.values())


# --------------------------------------------------------------------------- #
class TestRepresentativeSelection:
    def test_most_frequent_member_wins(self):
        vectors = {"tiny": unit(0), "small": unit(2), "little": unit(4)}
        counts = {"tiny": 5, "small": 99, "little": 20}
        reducer = build_from_vectors(vectors, counts, threshold=0.9)
        assert set(reducer.reduction_map.values()) == {"small"}

    def test_string_length_is_never_a_criterion(self):
        """Guards against reintroducing an orthographic rule into a semantic method."""
        vectors = {"xl": unit(0), "enormous": unit(2)}
        counts = {"xl": 1, "enormous": 50}
        reducer = build_from_vectors(vectors, counts, threshold=0.9)
        assert set(reducer.reduction_map.values()) == {"enormous"}

    def test_frequency_ties_break_lexicographically(self):
        vectors = {"beta": unit(0), "alpha": unit(2)}
        counts = {"beta": 7, "alpha": 7}
        reducer = build_from_vectors(vectors, counts, threshold=0.9)
        assert set(reducer.reduction_map.values()) == {"alpha"}

    def test_unrelated_types_are_left_alone(self):
        vectors = {"cat": unit(0), "algebra": unit(120)}
        counts = {"cat": 10, "algebra": 10}
        reducer = build_from_vectors(vectors, counts, threshold=0.7)
        assert reducer.reduction_map == {"cat": "cat", "algebra": "algebra"}


# --------------------------------------------------------------------------- #
class TestInference:
    @pytest.fixture
    def fitted(self):
        vectors = {"fast": unit(0), "quick": unit(5), "dog": unit(90)}
        counts = {"fast": 10, "quick": 3, "dog": 7}
        return build_from_vectors(vectors, counts, threshold=0.9)

    def test_reduce_replaces_merged_words(self, fitted):
        assert fitted.reduce("quick") == "fast"

    def test_punctuation_is_tokenized_the_same_way_as_training(self, fitted):
        """A train/inference tokenizer mismatch would silently break every lookup."""
        assert fitted.reduce("quick.") == "fast ."

    def test_unknown_words_pass_through(self, fitted):
        assert fitted.reduce("zebra") == "zebra"

    def test_reduce_batch_matches_reduce(self, fitted):
        texts = ["quick dog", "fast zebra"]
        assert fitted.reduce_batch(texts) == [fitted.reduce(t) for t in texts]

    def test_lowercase_config_is_applied_at_inference(self):
        vectors = {"fast": unit(0), "quick": unit(5)}
        counts = {"fast": 10, "quick": 3}
        reducer = build_from_vectors(vectors, counts, threshold=0.9, lowercase=True)
        assert reducer.reduce("QUICK") == "fast"


# --------------------------------------------------------------------------- #
class TestDiagnostics:
    @pytest.fixture
    def fitted(self):
        vectors = {"A": unit(0), "B": unit(40), "C": unit(80), "far": unit(200)}
        counts = {"A": 10, "B": 20, "C": 30, "far": 5}
        return build_from_vectors(vectors, counts, threshold=0.7, linkage=0.0)

    def test_cluster_stats_reports_compression(self, fitted):
        stats = fitted.cluster_stats()
        assert stats["n_types"] == 4
        assert stats["n_clusters"] == 2
        assert stats["largest_cluster"] == 3
        assert stats["vocab_reduction_pct"] == pytest.approx(50.0)

    def test_drift_report_detects_a_violated_bound(self, fitted):
        """Single linkage admits sub-threshold pairs, and the report must say so."""
        report = fitted.drift_report()
        assert report["bound_holds"] is False
        assert report["clusters_below_threshold"] == 1
        assert report["min_internal_similarity"] < 0.7

    def test_drift_report_confirms_the_bound_under_complete_linkage(self):
        vectors = {"A": unit(0), "B": unit(40), "C": unit(80)}
        counts = {"A": 10, "B": 20, "C": 30}
        reducer = build_from_vectors(vectors, counts, threshold=0.7, linkage=1.0)
        report = reducer.drift_report()
        assert report["bound_holds"] is True
        assert report["clusters_below_threshold"] == 0

    def test_geometry_report_records_both_sides_of_the_correction(self):
        rng = np.random.default_rng(4)
        vectors = {
            name(i): (np.ones(16, dtype=np.float32) * 5.0
                      + rng.normal(scale=0.4, size=16).astype(np.float32))
            for i in range(80)
        }
        counts = {w: 10 for w in vectors}
        reducer = build_from_vectors(vectors, counts, threshold=0.9, anisotropy=True)
        report = reducer.geometry_report()
        assert report["mean_cosine_before_correction"] > 0.9
        assert report["mean_cosine_after_correction"] < 0.5
        assert report["anisotropy_correction"] is True

    def test_sample_merges_excludes_fixed_points(self, fitted):
        for word, rep in fitted.sample_merges():
            assert word != rep

    def test_stats_are_empty_before_fitting(self):
        assert SemanticReducer().cluster_stats() == {}
        assert SemanticReducer().drift_report() == {}
        assert SemanticReducer().polysemy_report() == []


# --------------------------------------------------------------------------- #
class TestPolysemyAndProtection:
    def test_polysemy_report_ranks_least_concentrated_first(self):
        vectors = {"clear": unit(0), "vague": unit(90)}
        counts = {"clear": 10, "vague": 10}
        reducer = build_from_vectors(vectors, counts, threshold=0.99)
        reducer.concentration = np.array([0.95, 0.20], dtype=np.float32)
        ranked = reducer.polysemy_report(top_k=2)
        assert ranked[0][0] == "vague"
        assert ranked[0][1] == pytest.approx(0.20, abs=1e-4)

    def test_low_concentration_types_are_held_out_of_merges(self):
        vectors = {"bank": unit(0), "shore": unit(2)}
        counts = {"bank": 50, "shore": 10}
        config = ReducerConfig(
            threshold=0.9, anisotropy=False, min_count=1, min_concentration=0.5
        )
        reducer = SemanticReducer(config=config)
        reducer.vocab = ["bank", "shore"]
        from collections import Counter
        reducer.counts = Counter(counts)
        reducer.concentration = np.array([0.10, 0.95], dtype=np.float32)  # bank is polysemous
        reducer._fit_from_type_vectors(np.stack([vectors["bank"], vectors["shore"]]))

        assert reducer.reduction_map["bank"] == "bank"
        assert reducer.protection_report() == {"low_concentration": 1}

    def test_numerals_and_punctuation_are_protected_by_default(self):
        vectors = {"5": unit(0), "6": unit(1), ".": unit(2), "five": unit(3)}
        counts = {"5": 10, "6": 10, ".": 10, "five": 10}
        reducer = build_from_vectors(vectors, counts, threshold=0.9)
        assert reducer.reduction_map["5"] == "5"
        assert reducer.reduction_map["."] == "."
        report = reducer.protection_report()
        assert report["numeral"] == 2 and report["punctuation"] == 1

    def test_cannot_link_keeps_a_pair_apart(self):
        vectors = {"good": unit(0), "bad": unit(2)}
        counts = {"good": 50, "bad": 40}
        reducer = build_from_vectors(
            vectors, counts, threshold=0.9, cannot_link={"good": ("bad",)}
        )
        assert reducer.reduction_map["good"] != reducer.reduction_map["bad"]
        assert reducer.constraint_report()["pairs_applicable"] == 1   # unordered pairs

    def test_cannot_link_is_symmetric(self):
        """Naming the pair in one direction must be enough."""
        vectors = {"hot": unit(0), "cold": unit(2)}
        counts = {"hot": 10, "cold": 99}          # cold would otherwise be the rep
        reducer = build_from_vectors(
            vectors, counts, threshold=0.9, cannot_link={"hot": ("cold",)}
        )
        assert reducer.reduction_map["hot"] != reducer.reduction_map["cold"]

    def test_cannot_link_survives_a_transitive_chain(self):
        """The component-level guarantee: no intermediary may unite the pair."""
        vectors = {"aaa": unit(0), "mid": unit(5), "zzz": unit(10)}
        counts = {"aaa": 30, "mid": 20, "zzz": 10}
        reducer = build_from_vectors(
            vectors, counts, threshold=0.9, linkage=0.0,
            cannot_link={"aaa": ("zzz",)},
        )
        assert reducer.reduction_map["aaa"] != reducer.reduction_map["zzz"]

    def test_cannot_link_naming_absent_words_warns(self):
        vectors = {"one": unit(0), "two": unit(2)}
        counts = {"one": 5, "two": 5}
        with pytest.warns(RuntimeWarning, match="cannot_link"):
            build_from_vectors(
                vectors, counts, threshold=0.9,
                cannot_link={"absent": ("missing",)},
            )

    def test_no_cannot_link_leaves_merging_untouched(self):
        vectors = {"good": unit(0), "bad": unit(2)}
        counts = {"good": 50, "bad": 40}
        reducer = build_from_vectors(vectors, counts, threshold=0.9)
        assert reducer.reduction_map["bad"] == "good"
        assert reducer.constraint_report()["pairs_configured"] == 0

    def test_explicit_protect_set_is_honoured(self):
        vectors = {"paris": unit(0), "london": unit(2)}
        counts = {"paris": 5, "london": 50}
        reducer = build_from_vectors(
            vectors, counts, threshold=0.9, protect=frozenset({"paris"})
        )
        assert reducer.reduction_map["paris"] == "paris"


# --------------------------------------------------------------------------- #
class TestVocabularyConstruction:
    def test_min_count_drops_rare_types(self):
        rng = np.random.default_rng(9)
        vectors = {name(i): rng.normal(size=8).astype(np.float32) for i in range(60)}
        counts = {w: (10 if i % 2 else 1) for i, w in enumerate(vectors)}

        config = ReducerConfig(threshold=0.9, anisotropy=False, min_count=5)
        reducer = SemanticReducer(config=config)
        from collections import Counter
        stats_counts = Counter(counts)
        reducer.vocab = sorted(w for w, c in stats_counts.items() if c >= 5)
        reducer.counts = Counter({w: stats_counts[w] for w in reducer.vocab})
        reducer.concentration = np.ones(len(reducer.vocab), dtype=np.float32)
        raw = np.stack([vectors[w] for w in reducer.vocab])
        reducer._fit_from_type_vectors(raw)

        assert len(reducer.vocab) == 30
        assert all(reducer.counts[w] >= 5 for w in reducer.vocab)

    def test_type_vectors_are_unit_norm(self):
        vectors = {f"w{i}": unit(i * 20) for i in range(6)}
        counts = {f"w{i}": 3 for i in range(6)}
        reducer = build_from_vectors(vectors, counts, threshold=0.9)
        assert np.allclose(np.linalg.norm(reducer.vectors, axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
class TestPersistence:
    @pytest.fixture
    def fitted(self):
        vectors = {"A": unit(0), "B": unit(40), "C": unit(80)}
        counts = {"A": 10, "B": 20, "C": 30}
        return build_from_vectors(vectors, counts, threshold=0.7)

    def test_round_trip_preserves_map_and_clusters(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json")
        loaded = SemanticReducer.load(path)
        assert loaded.reduction_map == fitted.reduction_map
        assert loaded.clusters == fitted.clusters
        assert loaded.vocab == fitted.vocab

    def test_loaded_system_reduces_identically(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json")
        loaded = SemanticReducer.load(path)
        assert loaded.reduce("A B C") == fitted.reduce("A B C")

    def test_artifact_records_its_configuration(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config"]["threshold"] == fitted.config.threshold
        assert payload["config"]["linkage"] == fitted.config.linkage
        assert payload["config"]["model_name"] == fitted.config.model_name
        assert "geometry" in payload["provenance"]

    def test_artifact_is_plain_readable_json(self, fitted, tmp_path):
        """The supplement to a paper should be inspectable without this package."""
        path = fitted.save(tmp_path / "sys.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["reduction_map"]["A"] == fitted.reduction_map["A"]

    def test_vectors_are_optional_and_round_trip(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json", include_vectors=True)
        assert path.with_suffix(".vectors.npy").exists()
        loaded = SemanticReducer.load(path)
        assert np.allclose(loaded.vectors, fitted.vectors)

    def test_vectors_are_not_written_by_default(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json")
        assert not path.with_suffix(".vectors.npy").exists()

    def test_diagnostics_survive_a_round_trip(self, fitted, tmp_path):
        """A reloaded system must still report drift without the type vectors.

        Recomputing drift needs the vectors, which save() omits by default, so
        the fit-time report is persisted. Before this, every cache hit during a
        sweep crashed with AttributeError on a None cluster result.
        """
        path = fitted.save(tmp_path / "sys.json")
        loaded = SemanticReducer.load(path)

        drift = loaded.drift_report()
        assert "unavailable" not in drift
        assert drift["bound_holds"] == fitted.drift_report()["bound_holds"]
        assert loaded.protection_report() == fitted.protection_report()
        assert all(v for v in loaded.verify_guarantees().values() if v is not None)

    def test_loaded_system_reports_stats_without_crashing(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json")
        loaded = SemanticReducer.load(path)
        assert loaded.cluster_stats()["n_clusters"] == fitted.cluster_stats()["n_clusters"]
        assert loaded.sample_merges() == fitted.sample_merges()

    def test_drift_is_recomputable_when_vectors_are_saved(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json", include_vectors=True)
        loaded = SemanticReducer.load(path)
        assert loaded.vectors is not None
        assert "unavailable" not in loaded.drift_report()

    def test_foreign_format_version_is_rejected(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"format_version": 1, "reduction_map": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="format_version"):
            SemanticReducer.load(path)

    def test_saving_before_fitting_is_rejected(self, tmp_path):
        with pytest.raises(RuntimeError, match="fit"):
            SemanticReducer().save(tmp_path / "x.json")

    def test_unknown_config_key_is_rejected(self, fitted, tmp_path):
        path = fitted.save(tmp_path / "sys.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["config"]["invented_option"] = 1
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="unrecognized config keys"):
            SemanticReducer.load(path)


# --------------------------------------------------------------------------- #
class TestRealisticPipeline:
    """The shipping defaults on a vocabulary large enough for correction to be well-posed."""

    @staticmethod
    def planted(n_singletons=270, n_groups=10, group_size=3, dim=32, seed=7):
        """Anisotropic embeddings with planted near-duplicate groups.

        Names are digit-free on purpose: ``protect_numerals`` is on by default,
        so a vocabulary of ``solo001``-style tokens would be entirely protected
        and nothing would merge.
        """
        rng = np.random.default_rng(seed)
        cone = np.ones(dim, dtype=np.float32) * 6.0
        vectors, counts, groups = {}, {}, []

        for i in range(n_singletons):
            w = name(i, prefix="solo")
            vectors[w] = cone + rng.normal(scale=1.0, size=dim).astype(np.float32)
            counts[w] = int(rng.integers(5, 50))

        for g in range(n_groups):
            anchor = cone + rng.normal(scale=1.0, size=dim).astype(np.float32)
            members = []
            for m in range(group_size):
                w = name(g * group_size + m, prefix="grp")
                vectors[w] = anchor + rng.normal(scale=0.02, size=dim).astype(np.float32)
                counts[w] = int(rng.integers(5, 50))
                members.append(w)
            groups.append(members)

        return vectors, counts, groups

    @pytest.fixture(scope="class")
    def fitted(self):
        vectors, counts, groups = self.planted()
        # Calibrate tau into the gap between within-group and cross-group
        # similarity, so the test measures the algorithm rather than a constant.
        probe = build_from_vectors(vectors, counts, threshold=0.99, anisotropy=True)
        S = cosine_matrix(probe)
        idx = {w: i for i, w in enumerate(probe.vocab)}

        within = [S[idx[a], idx[b]]
                  for members in groups
                  for i, a in enumerate(members) for b in members[i + 1:]]
        cross = S.copy()
        np.fill_diagonal(cross, -np.inf)
        for members in groups:
            for a in members:
                for b in members:
                    cross[idx[a], idx[b]] = -np.inf

        min_within, max_cross = float(min(within)), float(cross.max())
        assert min_within > max_cross, "fixture has no separable gap"
        tau = (min_within + max_cross) / 2

        reducer = build_from_vectors(vectors, counts, threshold=tau, anisotropy=True)
        return reducer, groups, tau

    def test_planted_groups_are_recovered_exactly(self, fitted):
        reducer, groups, _ = fitted
        for members in groups:
            reps = {reducer.reduction_map[w] for w in members}
            assert len(reps) == 1, f"group {members} split across {reps}"

    def test_unrelated_types_are_not_swept_in(self, fitted):
        reducer, groups, _ = fitted
        planted = {w for members in groups for w in members}
        for rep, members in reducer.clusters.items():
            if any(m in planted for m in members):
                assert set(members) <= planted, f"cluster {rep} mixes in unrelated types"

    def test_no_runaway_chaining(self, fitted):
        reducer, _, _ = fitted
        assert reducer.cluster_stats()["largest_cluster"] <= 3

    def test_guarantees_hold_on_the_default_path(self, fitted):
        reducer, _, _ = fitted
        assert all(reducer.verify_guarantees().values())

    def test_reduction_is_reported_correctly(self, fitted):
        reducer, groups, _ = fitted
        stats = reducer.cluster_stats()
        assert stats["n_types"] - stats["n_clusters"] == 20   # 10 groups of 3 -> 10 reps
        assert stats["n_clusters_gt1"] == len(groups)


# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_keyword_overrides_build_a_config(self):
        reducer = SemanticReducer(threshold=0.42, linkage=0.5)
        assert reducer.config.threshold == 0.42
        assert reducer.config.linkage == 0.5

    def test_config_and_overrides_together_are_rejected(self):
        with pytest.raises(TypeError, match="not both"):
            SemanticReducer(config=ReducerConfig(), threshold=0.5)

    def test_repr_before_and_after_fitting(self):
        assert "unfitted" in repr(SemanticReducer())
        vectors = {"a": unit(0), "b": unit(2)}
        reducer = build_from_vectors(vectors, {"a": 5, "b": 5}, threshold=0.9)
        assert "classes" in repr(reducer)

    def test_assign_oov_requires_vectors(self, tmp_path):
        vectors = {"a": unit(0), "b": unit(2)}
        reducer = build_from_vectors(vectors, {"a": 5, "b": 5}, threshold=0.9)
        path = reducer.save(tmp_path / "sys.json")            # no vectors written
        loaded = SemanticReducer.load(path)
        with pytest.raises(RuntimeError, match="include_vectors"):
            loaded.assign_oov(["zebra"])
