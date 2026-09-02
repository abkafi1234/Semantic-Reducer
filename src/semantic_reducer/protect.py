"""Which types are allowed to merge.

Everything here is deliberately language-neutral. Protecting named entities with
an NER model would work, but it would reintroduce exactly what this method
exists to avoid: a per-language component that has to be built, trained, and
maintained for every language you want to support. These rules use only
properties of the string and of the corpus statistics.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["build_mergeable_mask", "protection_reasons"]

_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)
_HAS_DIGIT = re.compile(r"\d", re.UNICODE)


def _is_punctuation(token: str) -> bool:
    """No word characters at all -- pure punctuation or symbols."""
    return not _HAS_WORD_CHAR.search(token)


def _is_numeral(token: str) -> bool:
    """Contains a digit. Merging numbers is almost never wanted."""
    return bool(_HAS_DIGIT.search(token))


def _is_capitalized(token: str) -> bool:
    """Leading uppercase with a lowercase remainder ("Paris", not "NASA" or "the")."""
    return len(token) > 1 and token[0].isupper() and not token.isupper()


def protection_reasons(
    vocab: list[str],
    protect: frozenset[str] = frozenset(),
    protect_punctuation: bool = True,
    protect_numerals: bool = True,
    protect_pattern: str | None = None,
    protect_capitalized: bool = False,
    concentration: np.ndarray | None = None,
    min_concentration: float | None = None,
) -> dict[str, str]:
    """Map each protected type to the single rule that first protected it.

    Useful for auditing: it answers "why did this word never merge?".
    """
    pattern = re.compile(protect_pattern) if protect_pattern else None
    reasons: dict[str, str] = {}

    for i, token in enumerate(vocab):
        if token in protect:
            reasons[token] = "explicit"
        elif protect_punctuation and _is_punctuation(token):
            reasons[token] = "punctuation"
        elif protect_numerals and _is_numeral(token):
            reasons[token] = "numeral"
        elif pattern is not None and pattern.fullmatch(token):
            reasons[token] = "pattern"
        elif protect_capitalized and _is_capitalized(token):
            reasons[token] = "capitalized"
        elif (
            min_concentration is not None
            and concentration is not None
            and concentration[i] < min_concentration
        ):
            reasons[token] = "low_concentration"

    return reasons


def build_mergeable_mask(
    vocab: list[str],
    protect: frozenset[str] = frozenset(),
    protect_punctuation: bool = True,
    protect_numerals: bool = True,
    protect_pattern: str | None = None,
    protect_capitalized: bool = False,
    concentration: np.ndarray | None = None,
    min_concentration: float | None = None,
) -> np.ndarray:
    """Boolean mask over ``vocab``: True where a type may participate in a merge."""
    reasons = protection_reasons(
        vocab,
        protect=protect,
        protect_punctuation=protect_punctuation,
        protect_numerals=protect_numerals,
        protect_pattern=protect_pattern,
        protect_capitalized=protect_capitalized,
        concentration=concentration,
        min_concentration=min_concentration,
    )
    return np.array([tok not in reasons for tok in vocab], dtype=bool)
