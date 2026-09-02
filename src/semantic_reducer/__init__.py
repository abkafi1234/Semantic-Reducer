"""Semantic Reducer -- corpus-grounded vocabulary reduction.

Collapses a corpus vocabulary into semantic equivalence classes learned from
contextual embeddings, then reduces text by dictionary lookup: O(1) per token,
no encoder at inference, and idempotent by construction.
"""

from .config import ReducerConfig
from .encoder import ContextualEncoder, describe_device, resolve_device, tokenize
from .finetuning import FinetuneResult, continue_pretraining
from .geometry import correct_anisotropy, l2_normalize, mean_offdiagonal_cosine
from .linkage import UnionFind, agglomerate, min_internal_similarity
from .neighbors import find_edges
from .protect import build_mergeable_mask, protection_reasons
from .reducer import SemanticReducer

__version__ = "0.4.0"

__all__ = [
    "SemanticReducer",
    "ReducerConfig",
    # geometry
    "correct_anisotropy",
    "l2_normalize",
    "mean_offdiagonal_cosine",
    # clustering
    "UnionFind",
    "agglomerate",
    "min_internal_similarity",
    "find_edges",
    # protection
    "build_mergeable_mask",
    "protection_reasons",
    # encoding
    "ContextualEncoder",
    "tokenize",
    "resolve_device",
    "describe_device",
    # optional fine-tuning
    "continue_pretraining",
    "FinetuneResult",
    "__version__",
]
