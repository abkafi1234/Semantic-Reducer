"""Tests for the protection rules.

These rules exist to keep merges away from tokens where semantic similarity is
the wrong criterion. They are deliberately language-neutral: an NER model would
work better on English but would reintroduce the per-language component the
method exists to avoid.
"""

import numpy as np
import pytest

from semantic_reducer import build_mergeable_mask, protection_reasons


def reasons(vocab, **kwargs):
    return protection_reasons(vocab, **kwargs)


class TestDefaults:
    def test_punctuation_is_protected(self):
        assert reasons([".", ",", "!?"]) == {".": "punctuation", ",": "punctuation",
                                             "!?": "punctuation"}

    def test_numerals_are_protected(self):
        got = reasons(["1990", "3.14", "covid19"])
        assert got == {"1990": "numeral", "3.14": "numeral", "covid19": "numeral"}

    def test_ordinary_words_are_mergeable(self):
        assert reasons(["fast", "quick", "hound"]) == {}

    def test_punctuation_and_numerals_can_be_switched_off(self):
        got = reasons(["1990", "."], protect_punctuation=False, protect_numerals=False)
        assert got == {}


class TestExplicitRules:
    def test_explicit_set_takes_priority(self):
        assert reasons(["paris"], protect=frozenset({"paris"})) == {"paris": "explicit"}

    def test_pattern_must_match_the_whole_token(self):
        got = reasons(["ID_1", "valid"], protect_pattern=r"ID_\w+", protect_numerals=False)
        assert got == {"ID_1": "pattern"}

    def test_pattern_does_not_match_substrings(self):
        assert reasons(["xxIDxx"], protect_pattern=r"ID") == {}


class TestCapitalization:
    def test_off_by_default(self):
        """German capitalizes every noun, so this heuristic is not language-neutral."""
        assert reasons(["Paris", "Haus", "Katze"]) == {}

    def test_can_be_enabled_explicitly(self):
        got = reasons(["Paris", "london"], protect_capitalized=True)
        assert got == {"Paris": "capitalized"}

    def test_acronyms_are_not_treated_as_capitalized(self):
        assert reasons(["NASA"], protect_capitalized=True) == {}

    def test_single_letters_are_not_treated_as_capitalized(self):
        assert reasons(["A"], protect_capitalized=True) == {}


class TestConcentration:
    def test_low_concentration_types_are_protected(self):
        conc = np.array([0.1, 0.9], dtype=np.float32)
        got = reasons(["bank", "shore"], concentration=conc, min_concentration=0.5)
        assert got == {"bank": "low_concentration"}

    def test_no_effect_without_a_threshold(self):
        conc = np.array([0.1, 0.9], dtype=np.float32)
        assert reasons(["bank", "shore"], concentration=conc) == {}

    def test_exactly_at_the_threshold_is_allowed(self):
        conc = np.array([0.5], dtype=np.float32)
        assert reasons(["edge"], concentration=conc, min_concentration=0.5) == {}


class TestMask:
    def test_mask_is_the_complement_of_the_reasons(self):
        vocab = ["fast", "1990", ".", "quick"]
        mask = build_mergeable_mask(vocab)
        assert mask.tolist() == [True, False, False, True]
        assert mask.dtype == bool

    def test_mask_length_matches_vocabulary(self):
        vocab = [f"w{i}" for i in range(17)]
        assert build_mergeable_mask(vocab).shape == (17,)

    def test_empty_vocabulary_is_handled(self):
        assert build_mergeable_mask([]).shape == (0,)


class TestPriority:
    def test_first_matching_rule_is_reported(self):
        """Explicit protection outranks the structural rules."""
        got = reasons(["1990"], protect=frozenset({"1990"}))
        assert got == {"1990": "explicit"}
