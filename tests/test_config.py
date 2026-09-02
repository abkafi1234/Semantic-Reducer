"""Tests for configuration validation and serialization.

Invalid settings must be rejected at construction, not thousands of encoder
batches later, and a config must round-trip so an artifact can record exactly
how it was produced.
"""

import pytest

from semantic_reducer import ReducerConfig


class TestValidation:
    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
    def test_linkage_outside_the_unit_interval_is_rejected(self, value):
        with pytest.raises(ValueError, match="linkage"):
            ReducerConfig(linkage=value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_valid_linkage_is_accepted(self, value):
        assert ReducerConfig(linkage=value).linkage == value

    @pytest.mark.parametrize("value", [-1.5, 1.5])
    def test_threshold_must_be_a_cosine(self, value):
        with pytest.raises(ValueError, match="threshold"):
            ReducerConfig(threshold=value)

    def test_negative_threshold_is_allowed(self):
        """Cosines are legitimately negative; only out-of-range values are errors."""
        assert ReducerConfig(threshold=-0.2).threshold == -0.2

    def test_unknown_subword_pooling_is_rejected(self):
        with pytest.raises(ValueError, match="subword_pooling"):
            ReducerConfig(subword_pooling="cls")

    def test_unknown_dtype_is_rejected(self):
        with pytest.raises(ValueError, match="dtype"):
            ReducerConfig(dtype="float8")

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            ReducerConfig(backend="annoy")

    def test_min_count_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="min_count"):
            ReducerConfig(min_count=0)

    def test_negative_abtt_is_rejected(self):
        with pytest.raises(ValueError, match="n_abtt"):
            ReducerConfig(n_abtt=-1)

    def test_empty_layers_is_rejected(self):
        with pytest.raises(ValueError, match="layers"):
            ReducerConfig(layers=())

    def test_min_concentration_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="min_concentration"):
            ReducerConfig(min_concentration=1.5)

    def test_max_cluster_size_must_be_positive(self):
        with pytest.raises(ValueError, match="max_cluster_size"):
            ReducerConfig(max_cluster_size=0)

    def test_none_max_cluster_size_means_unlimited(self):
        assert ReducerConfig(max_cluster_size=None).max_cluster_size is None


class TestNormalization:
    def test_layers_are_stored_as_a_tuple(self):
        assert ReducerConfig(layers=[-1, -2]).layers == (-1, -2)

    def test_protect_is_stored_as_a_frozenset(self):
        config = ReducerConfig(protect={"a", "b"})
        assert isinstance(config.protect, frozenset)


class TestSerialization:
    def test_round_trip_preserves_every_field(self):
        original = ReducerConfig(
            threshold=0.55, linkage=0.75, layers=(-1, -2),
            protect=frozenset({"x", "y"}), min_concentration=0.3,
        )
        restored = ReducerConfig.from_dict(original.to_dict())
        assert restored == original

    def test_to_dict_is_json_safe(self):
        import json

        payload = json.dumps(ReducerConfig().to_dict())
        assert isinstance(json.loads(payload), dict)

    def test_layers_serialize_as_a_list(self):
        assert ReducerConfig(layers=(-1, -2)).to_dict()["layers"] == [-1, -2]

    def test_protect_serializes_sorted(self):
        assert ReducerConfig(protect={"b", "a"}).to_dict()["protect"] == ["a", "b"]

    def test_unknown_keys_are_rejected(self):
        payload = ReducerConfig().to_dict()
        payload["from_the_future"] = True
        with pytest.raises(ValueError, match="unrecognized config keys"):
            ReducerConfig.from_dict(payload)

    def test_defaults_are_the_documented_ones(self):
        """Guards against a default silently changing what published maps mean."""
        config = ReducerConfig()
        assert config.linkage == 1.0          # bounded drift by default
        assert config.anisotropy is True      # threshold must be interpretable
        assert config.n_abtt == 2
        assert config.min_count == 5
        assert config.dtype == "float32"      # reproducible across hardware
        assert config.protect_capitalized is False   # not language-neutral
