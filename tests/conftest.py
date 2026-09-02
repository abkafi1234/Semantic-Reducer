"""Shared test fixtures and helpers.

``src/`` is put on the path so the suite runs against the working tree without
an install step.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_reducer import ReducerConfig, SemanticReducer  # noqa: E402


def name(i: int, prefix: str = "w", width: int = 3) -> str:
    """A digit-free identifier, e.g. ``name(0) == "waaa"``.

    Generated vocabularies must avoid digits: ``protect_numerals`` is on by
    default, so a fixture using ``w001`` would have its entire vocabulary held
    out of merging and would silently test nothing.
    """
    letters = ""
    for _ in range(width):
        i, remainder = divmod(i, 26)
        letters = chr(ord("a") + remainder) + letters
    return prefix + letters


def unit(angle_deg: float, dim: int = 8) -> np.ndarray:
    """A unit vector at a given angle within the first two dimensions.

    Lets a fixture state its geometry directly: two vectors at 40 degrees have
    cosine cos(40) = 0.766, so thresholds in tests are exact, not guessed.
    """
    theta = np.deg2rad(angle_deg)
    v = np.zeros(dim, dtype=np.float32)
    v[0], v[1] = np.cos(theta), np.sin(theta)
    return v


def build_from_vectors(
    vectors: dict[str, np.ndarray],
    counts: dict[str, int],
    threshold: float,
    anisotropy: bool = False,
    **config_overrides,
) -> SemanticReducer:
    """Fit a reducer from explicit type vectors, bypassing the encoder.

    Anisotropy correction defaults to OFF so fixtures have exactly the geometry
    they declare — the correction is estimated from the population of type
    vectors and is degenerate on a handful of them. ``TestRealisticPipeline`` in
    test_reducer.py exercises the real default path.
    """
    config = ReducerConfig(
        threshold=threshold,
        anisotropy=anisotropy,
        min_count=1,
        **config_overrides,
    )
    reducer = SemanticReducer(config=config)
    reducer.vocab = sorted(vectors)
    reducer.counts = Counter({w: counts[w] for w in reducer.vocab})
    reducer.concentration = np.ones(len(reducer.vocab), dtype=np.float32)
    raw = np.stack([np.asarray(vectors[w], dtype=np.float32) for w in reducer.vocab])
    reducer._fit_from_type_vectors(raw)
    return reducer


def cosine_matrix(reducer: SemanticReducer) -> np.ndarray:
    """Pairwise cosine over the fitted (already normalized) type vectors."""
    X = reducer.vectors
    return X @ X.T
