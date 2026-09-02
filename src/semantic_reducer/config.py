"""Configuration for the reduction pipeline.

Every knob that can change the produced map lives here, and the whole config is
serialized alongside each saved artifact so a reduction map always records the
exact settings that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence


@dataclass
class ReducerConfig:
    """Settings for :class:`semantic_reducer.SemanticReducer`."""

    # ----------------------------- encoder ---------------------------------- #
    model_name: str = "bert-base-multilingual-cased"

    layers: tuple[int, ...] = (-4, -3, -2, -1)
    """Hidden-state indices to average into the occurrence vector.

    Averaging the last four layers is the standard choice; ``(-1,)`` uses only
    the final layer. Exposed because layer choice is an ablation axis.
    """

    subword_pooling: str = "mean"
    """How to pool the sub-word pieces of one word: ``mean``, ``first``, or ``max``.

    A word split by WordPiece into several pieces needs its pieces recombined.
    ``first`` takes the head piece, which is the common shortcut; ``mean``
    averages all pieces and is the default.
    """

    max_length: int = 256
    batch_size: int = 32

    device: str | None = None
    """``None`` auto-selects CUDA, then MPS, then CPU. Or force ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``."""

    dtype: str = "float32"
    """Encoder precision: ``float32`` (default, reproducible) or ``float16``/``bfloat16``.

    Reduced precision roughly halves encoding time on CUDA but makes results
    sensitive to hardware, so the reproducible option is the default.
    """

    # ------------------------- optional fine-tuning -------------------------- #
    finetune: bool = False
    """Continue self-supervised (masked-language-model) pretraining of the
    encoder on the fitting corpus before extracting type vectors. OFF by
    default -- the method is training-free by default; this is an opt-in
    stage, not a change to the default pipeline.

    Uses no labels: it continues whatever objective the plugged-in model was
    already pretrained with (masked-language-modeling), on the same raw text
    already being normalized. No new resource is introduced. This is
    materially more expensive than a single frozen forward pass, and is not
    guaranteed to help -- on a small corpus it can as easily overfit or
    degrade representations as improve them. Validate before trusting it;
    do not assume it helps.

    Requires the model to be loadable via ``AutoModelForMaskedLM`` (i.e. a
    masked-language-model architecture, e.g. BERT-family; not a causal/
    decoder-only model).
    """

    finetune_epochs: int = 1
    finetune_lr: float = 5e-5
    finetune_batch_size: int | None = None
    """``None`` reuses ``batch_size``."""
    finetune_mlm_probability: float = 0.15
    """Fraction of tokens masked per batch, standard BERT-style MLM default."""

    # --------------------------- vocabulary --------------------------------- #
    lowercase: bool = False
    min_count: int = 5
    """Types occurring fewer than this many times are dropped before clustering.

    A type seen once has a vector from a single context; it is noise, and it is
    exactly the kind of type that merges spuriously.
    """

    # ---------------------------- geometry ---------------------------------- #
    anisotropy: bool = True
    """Master switch for anisotropy correction. ``False`` is the ablation baseline.

    Contextual embeddings occupy a narrow cone, so raw cosine between unrelated
    words is ~0.9 and a cosine threshold is not interpretable. Leave this on
    unless you are deliberately measuring its effect.
    """

    n_abtt: int = 2
    """Principal directions removed by all-but-the-top (0 = mean-centering only)."""

    # ---------------------------- clustering -------------------------------- #
    threshold: float = 0.6
    """Cosine cutoff tau, applied AFTER anisotropy correction. Inclusive (>= tau).

    Not comparable to a threshold on uncorrected embeddings: correction shifts
    the whole similarity distribution down. Tune per corpus with a sweep.
    """

    linkage: float = 1.0
    """Linkage strictness lambda in [0, 1].

    Two clusters merge only if at least this fraction of the pairs across them
    reach tau.

    * ``1.0`` -- complete linkage. Every pair inside a cluster is guaranteed to
      be at or above tau, so cluster diameter is bounded by ``1 - tau``. This is
      the default because the bound is the method's central guarantee.
    * ``0.0`` -- single linkage (plain connected components). Maximum
      compression, no bound: A~B~C merges even when A and C are unrelated.
    * in between -- interpolates, trading the bound for compression.
    """

    max_cluster_size: int | None = None
    """Optional hard cap on cluster size; merges that would exceed it are refused."""

    cannot_link: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Pairs that must never share a class, e.g. ``{"good": ("bad",)}``.

    Enforced at the COMPONENT level, so a forbidden pair cannot be united even
    transitively through a chain of intermediaries, and symmetrized on load, so
    naming a pair once is enough.

    Nothing is bundled. An antonym lexicon is inherently per-language -- WordNet
    coverage is English-first -- so shipping one would quietly make the method
    language-specific, which is the property it exists to avoid. Supply your own
    when you have one, and treat it as an optional constraint rather than part
    of the method.
    """

    # ------------------------------ safety ---------------------------------- #
    protect: frozenset[str] = frozenset()
    """Literal tokens that must never be merged (case-sensitive)."""

    protect_punctuation: bool = True
    protect_numerals: bool = True

    protect_pattern: str | None = None
    """Optional regex; any type fully matching it is excluded from merging."""

    protect_capitalized: bool = False
    """Treat mid-sentence capitalized types as protected (crude named-entity guard).

    OFF by default on purpose: German capitalizes every noun, so this heuristic
    is not language-neutral and would quietly disable reduction for some
    languages. Enable only when you know it fits your corpus.
    """

    min_concentration: float | None = None
    """If set, types whose occurrence vectors are more scattered than this are
    excluded from merging.

    The score is the mean resultant length of a type's unit occurrence vectors,
    in [0, 1]: near 1 the word is used consistently, near 0 its contexts pull in
    many directions, which is the signature of polysemy. Averaging such a type
    into one vector is the method's main known weakness, so this is the lever
    that avoids merging on an unreliable average.
    """

    # ---------------------------- execution --------------------------------- #
    backend: str = "auto"
    """Neighbour search backend: ``auto`` (torch), ``torch``, ``numpy``, or ``faiss``.

    All backends are exact -- they differ only in speed. ``torch`` runs the
    search on the configured device.
    """

    search_chunk: int = 1024
    """Rows per block during neighbour search; controls peak search memory."""

    seed: int = 0

    # ------------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        self.protect = frozenset(self.protect)
        self.cannot_link = {
            str(word): tuple(sorted(others))
            for word, others in dict(self.cannot_link).items()
        }
        self.validate()

    def validate(self) -> None:
        """Reject invalid settings up front rather than failing deep in a run."""
        if not 0.0 <= self.linkage <= 1.0:
            raise ValueError(f"linkage must be in [0, 1], got {self.linkage}")
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold is a cosine and must be in [-1, 1], got {self.threshold}")
        if self.subword_pooling not in {"mean", "first", "max"}:
            raise ValueError(
                f"subword_pooling must be 'mean', 'first', or 'max', got {self.subword_pooling!r}"
            )
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                f"dtype must be 'float32', 'float16', or 'bfloat16', got {self.dtype!r}"
            )
        if self.backend not in {"auto", "torch", "numpy", "faiss"}:
            raise ValueError(
                f"backend must be 'auto', 'torch', 'numpy', or 'faiss', got {self.backend!r}"
            )
        if self.min_count < 1:
            raise ValueError(f"min_count must be >= 1, got {self.min_count}")
        if self.n_abtt < 0:
            raise ValueError(f"n_abtt must be >= 0, got {self.n_abtt}")
        if self.max_cluster_size is not None and self.max_cluster_size < 1:
            raise ValueError(f"max_cluster_size must be >= 1 or None, got {self.max_cluster_size}")
        if self.min_concentration is not None and not 0.0 <= self.min_concentration <= 1.0:
            raise ValueError(
                f"min_concentration must be in [0, 1] or None, got {self.min_concentration}"
            )
        if self.search_chunk < 1:
            raise ValueError(f"search_chunk must be >= 1, got {self.search_chunk}")
        if self.finetune_epochs < 1:
            raise ValueError(f"finetune_epochs must be >= 1, got {self.finetune_epochs}")
        if self.finetune_lr <= 0:
            raise ValueError(f"finetune_lr must be > 0, got {self.finetune_lr}")
        if not 0.0 < self.finetune_mlm_probability < 1.0:
            raise ValueError(
                f"finetune_mlm_probability must be in (0, 1), got {self.finetune_mlm_probability}"
            )
        if not self.layers:
            raise ValueError("layers must name at least one hidden state")
        for word, others in self.cannot_link.items():
            if isinstance(others, str):
                raise ValueError(
                    f"cannot_link[{word!r}] must be a sequence of words, not a "
                    f"bare string (did you mean ({others!r},)?)"
                )
            if word in others:
                raise ValueError(f"cannot_link[{word!r}] lists {word!r} against itself")

    # ------------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """JSON-safe dict (tuples become lists, frozensets become sorted lists)."""
        d = asdict(self)
        d["layers"] = list(self.layers)
        d["protect"] = sorted(self.protect)
        d["cannot_link"] = {w: list(o) for w, o in sorted(self.cannot_link.items())}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ReducerConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"unrecognized config keys: {sorted(unknown)}. This artifact was "
                f"probably written by a different version of semantic-reducer."
            )
        return cls(**d)
