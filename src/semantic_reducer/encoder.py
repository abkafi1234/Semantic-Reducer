"""Contextual embedding extraction, device-agnostic.

Words are encoded inside their sentences, never in isolation: an occurrence
vector is the encoder's output *for that word in that context*, and a type
vector is the average over all of the word's occurrences in the corpus.

Occurrence vectors are L2-normalized before accumulation. That has two
consequences worth knowing. Each occurrence then contributes equally regardless
of vector magnitude (which in Transformer encoders tracks position and frequency
artefacts more than meaning), and the norm of the accumulated sum divided by the
occurrence count gives the *mean resultant length* -- a concentration statistic
in [0, 1] that measures how consistently a word is used. It costs nothing extra
and is the package's polysemy signal.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict

import numpy as np
import regex
from tqdm import tqdm

__all__ = ["tokenize", "resolve_device", "describe_device", "ContextualEncoder", "CorpusStatistics"]


# Used IDENTICALLY when building the corpus and when reducing new text; any
# divergence between the two would silently break every dictionary lookup.
#
# This uses the third-party `regex` module rather than stdlib `re` because
# stdlib `\w` is defined via str.isalnum(), which does NOT include Unicode
# combining marks (categories Mn/Mc/Me). Bengali, Devanagari, Tamil, and every
# other script that writes vowels as dependent signs attached to a consonant
# rely on exactly those combining marks -- a word like Bengali "বছরের" is a
# base letter followed by a spacing mark, and stdlib \w+ splits the mark off
# as its own token (category Mc: 'ে' is a Mc character, not alnum). The result
# is silent word-shredding for those scripts specifically: this project's
# flagship low-resource case, Bangla, was being tokenized into isolated vowel
# signs and consonant fragments rather than words. \p{L}\p{M}\p{N} explicitly
# keeps Letter + Mark + Number runs together, which \w does not guarantee.
_TOKEN_RE = regex.compile(r"[\p{L}\p{M}\p{N}_]+|[^\p{L}\p{M}\p{N}_\s]", regex.UNICODE)


def tokenize(text: str, lowercase: bool = False) -> list[str]:
    """Split into word and punctuation tokens."""
    toks = _TOKEN_RE.findall(str(text))
    return [t.lower() for t in toks] if lowercase else toks


# --------------------------------------------------------------------------- #
#  Device handling
# --------------------------------------------------------------------------- #
def resolve_device(requested: str | None = None):
    """Pick a torch device: explicit request, else CUDA, then MPS, then CPU.

    Args:
        requested: e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``. ``None``
            auto-detects.
    """
    import torch

    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} was requested but CUDA is not available. "
                f"Install a CUDA build of torch, or pass device='cpu'."
            )
        if device.type == "mps" and not _mps_available():
            raise RuntimeError(
                f"device={requested!r} was requested but MPS is not available."
            )
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    import torch

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def describe_device(device) -> str:
    """Human-readable device description for logs and saved provenance."""
    import torch

    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def _resolve_dtype(name: str, device):
    """Map a dtype name to a torch dtype, refusing combinations that misbehave."""
    import torch

    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = mapping[name]

    if device.type == "cpu" and dtype is torch.float16:
        warnings.warn(
            "float16 is not well supported for inference on CPU and is often "
            "slower than float32; falling back to float32.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.float32
    if device.type == "mps" and dtype is torch.bfloat16:
        warnings.warn(
            "bfloat16 support on MPS is incomplete; falling back to float32.",
            RuntimeWarning,
            stacklevel=3,
        )
        return torch.float32
    return dtype


# --------------------------------------------------------------------------- #
#  Corpus statistics
# --------------------------------------------------------------------------- #
class CorpusStatistics:
    """Per-type accumulations produced by one pass over a corpus."""

    def __init__(self, unit_sums: dict[str, np.ndarray], counts: Counter,
                 truncated: int, dim: int, unencodable: int = 0):
        self.unit_sums = unit_sums   # sum of L2-normalized occurrence vectors
        self.counts = counts
        self.truncated = truncated
        self.unencodable = unencodable
        self.dim = dim

    def type_vector(self, word: str) -> np.ndarray:
        """Mean of the word's unit occurrence vectors."""
        return self.unit_sums[word] / self.counts[word]

    def concentration(self, word: str) -> float:
        """Mean resultant length in [0, 1]: high = used consistently.

        A low value means the word's occurrences point in many directions, which
        is the signature of polysemy -- and a warning that collapsing it to one
        vector discards a real distinction.
        """
        return float(np.linalg.norm(self.unit_sums[word]) / self.counts[word])

    def merge(self, other: "CorpusStatistics") -> "CorpusStatistics":
        """Combine statistics from another pass (for incremental corpus building)."""
        if self.dim != other.dim:
            raise ValueError(f"dimension mismatch: {self.dim} vs {other.dim}")
        for word, vec in other.unit_sums.items():
            if word in self.unit_sums:
                self.unit_sums[word] = self.unit_sums[word] + vec
            else:
                self.unit_sums[word] = vec.copy()
        self.counts.update(other.counts)
        self.truncated += other.truncated
        self.unencodable += other.unencodable
        return self


# --------------------------------------------------------------------------- #
#  Encoder
# --------------------------------------------------------------------------- #
class ContextualEncoder:
    """Wraps a Hugging Face encoder and turns a corpus into per-type statistics."""

    def __init__(self, config):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._mlm_model = None    # set only when config.finetune; see finetune_on_corpus
        self._device = None
        self._torch_dtype = None
        self.hidden_size: int | None = None

    # ------------------------------------------------------------------ #
    @property
    def device(self):
        self._ensure_loaded()
        return self._device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self._device = resolve_device(cfg.device)
        self._torch_dtype = _resolve_dtype(cfg.dtype, self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if not self._tokenizer.is_fast:
            raise RuntimeError(
                f"{cfg.model_name!r} does not provide a fast tokenizer. Sub-word "
                f"pieces are regrouped into words via word_ids(), which only the "
                f"fast tokenizers expose."
            )

        if cfg.finetune:
            # Load with the masked-LM head attached so it can be trained;
            # `.base_model` is the same underlying encoder `encode_corpus`/
            # `encode_words` use, so weights updated during fine-tuning are
            # exactly the weights used for the (still frozen-at-that-point)
            # embedding extraction that follows it.
            from transformers import AutoModelForMaskedLM

            try:
                self._mlm_model = AutoModelForMaskedLM.from_pretrained(
                    cfg.model_name, dtype=self._torch_dtype
                )
            except (ValueError, OSError) as exc:
                raise ValueError(
                    f"finetune=True requires {cfg.model_name!r} to be loadable via "
                    f"AutoModelForMaskedLM (a masked-language-model architecture, "
                    f"e.g. BERT-family) -- got: {exc}"
                ) from exc
            self._model = self._mlm_model.base_model
            # Move/eval the FULL model (base + MLM head): .to()/.eval() on the
            # base_model submodule alone would skip the head, which is a
            # sibling submodule, not a descendant -- and the head's
            # parameters need to be on-device too for fine-tuning to run.
            self._mlm_model.eval()
            self._mlm_model.to(self._device)
        else:
            self._model = AutoModel.from_pretrained(cfg.model_name, dtype=self._torch_dtype)
            self._model.eval()
            self._model.to(self._device)
        self.hidden_size = self._model.config.hidden_size

    # ------------------------------------------------------------------ #
    def finetune_on_corpus(self, texts, progress: bool = True):
        """Continue MLM pretraining on ``texts`` before any embedding is extracted.

        Requires ``config.finetune=True`` (checked by the caller, normally
        ``SemanticReducer.fit``). Uses no labels -- see ``finetuning.py``.
        Leaves the encoder in ``eval()`` mode when it returns, ready for the
        frozen encoding pass that follows.

        Returns:
            A :class:`finetuning.FinetuneResult` with per-epoch loss, for
            transparency and for the fitted reducer's diagnostics.
        """
        if not self.config.finetune:
            raise RuntimeError(
                "finetune_on_corpus() called but config.finetune is False; "
                "set finetune=True to use this stage."
            )
        from .finetuning import continue_pretraining

        self._ensure_loaded()
        cfg = self.config
        result = continue_pretraining(
            self._mlm_model, self._tokenizer, list(texts), self._device,
            epochs=cfg.finetune_epochs,
            lr=cfg.finetune_lr,
            batch_size=cfg.finetune_batch_size or cfg.batch_size,
            max_length=cfg.max_length,
            mlm_probability=cfg.finetune_mlm_probability,
            seed=cfg.seed,
            progress=progress,
        )
        self._mlm_model.eval()
        return result

    # ------------------------------------------------------------------ #
    def encode_corpus(self, sentences, batch_size: int | None = None,
                      progress: bool = True) -> CorpusStatistics:
        """Encode a corpus and accumulate per-type statistics.

        Args:
            sentences: iterable of raw strings.
            batch_size: overrides ``config.batch_size``.
            progress: show a progress bar.
        """
        import torch

        self._ensure_loaded()
        cfg = self.config
        bs = batch_size or cfg.batch_size

        split = [tokenize(s, cfg.lowercase) for s in sentences]
        split = [toks for toks in split if toks]
        if not split:
            raise ValueError("corpus is empty after tokenization")

        unit_sums: dict[str, np.ndarray] = {}
        counts: Counter = Counter()
        truncated = 0
        unencodable = 0

        batches = range(0, len(split), bs)
        if progress:
            batches = tqdm(batches, desc=f"Encoding on {describe_device(self._device)}")

        for start in batches:
            batch = split[start:start + bs]

            inputs = self._tokenizer(
                batch,
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
            )
            model_inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                out = self._model(**model_inputs, output_hidden_states=True)

            # Average the requested hidden layers -> (B, T, H). Cast to float32
            # before leaving the GPU so accumulation never happens in fp16.
            hs = torch.stack(
                [out.hidden_states[i] for i in cfg.layers], dim=0
            ).mean(dim=0).float().cpu().numpy()

            for bi, words in enumerate(batch):
                word_ids = inputs.word_ids(batch_index=bi)

                # A word can receive no sub-word tokens for two quite different
                # reasons, and conflating them hides a real failure. Words past
                # the last encoded position were cut off by max_length. Words
                # BEFORE it produced no tokens at all -- the tokenizer discarded
                # them, which happens with zero-width and formatting characters
                # such as the soft hyphen U+00AD. The second kind is silent data
                # loss that raising max_length will never fix.
                seen = {w for w in word_ids if w is not None}
                if seen:
                    last_encoded = max(seen)
                    missing = [k for k in range(len(words)) if k not in seen]
                    unencodable += sum(1 for k in missing if k < last_encoded)
                    truncated += sum(1 for k in missing if k > last_encoded)
                else:
                    truncated += len(words)

                pieces: dict[int, list[np.ndarray]] = defaultdict(list)
                for seq_idx, wid in enumerate(word_ids):
                    if wid is not None and wid < len(words):
                        pieces[wid].append(hs[bi, seq_idx, :])

                for wid, vecs in pieces.items():
                    word = words[wid]
                    vec = _pool_subwords(vecs, cfg.subword_pooling)

                    # Normalize each occurrence before accumulating: equal weight
                    # per occurrence, and the summed norm becomes the
                    # concentration statistic.
                    norm = np.linalg.norm(vec)
                    if norm < 1e-12:
                        continue
                    vec = (vec / norm).astype(np.float32)

                    if word in unit_sums:
                        unit_sums[word] += vec
                    else:
                        unit_sums[word] = vec.copy()
                    counts[word] += 1

        if truncated:
            warnings.warn(
                f"{truncated} word occurrences were dropped by truncation at "
                f"max_length={cfg.max_length}. Raise max_length or pre-split long "
                f"documents so those occurrences contribute to their type vectors.",
                RuntimeWarning,
                stacklevel=2,
            )
        if unencodable:
            warnings.warn(
                f"{unencodable} word occurrences produced no sub-word tokens and "
                f"were skipped: {cfg.model_name!r} discards them entirely. This is "
                f"usually zero-width or formatting characters (soft hyphens, "
                f"joiners) that the tokenizer normalizes away. Such types never "
                f"enter the vocabulary, so raising max_length will not recover "
                f"them -- strip them from the corpus if you need them counted.",
                RuntimeWarning,
                stacklevel=2,
            )

        return CorpusStatistics(unit_sums, counts, truncated,
                                int(self.hidden_size), unencodable)

    # ------------------------------------------------------------------ #
    def encode_words(self, words: list[str]) -> np.ndarray:
        """Encode bare words with no surrounding context.

        Provided only for out-of-vocabulary assignment at inference time, where
        no context is available. These vectors are NOT contextual and are not
        comparable in kind to corpus-derived type vectors; see
        ``SemanticReducer.assign_oov``.
        """
        import torch

        self._ensure_loaded()
        cfg = self.config

        vectors = np.zeros((len(words), int(self.hidden_size)), dtype=np.float32)
        for start in range(0, len(words), cfg.batch_size):
            chunk = words[start:start + cfg.batch_size]
            inputs = self._tokenizer(
                [[w] for w in chunk],
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
            )
            model_inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**model_inputs, output_hidden_states=True)
            hs = torch.stack(
                [out.hidden_states[i] for i in cfg.layers], dim=0
            ).mean(dim=0).float().cpu().numpy()

            for bi in range(len(chunk)):
                word_ids = inputs.word_ids(batch_index=bi)
                vecs = [hs[bi, k, :] for k, wid in enumerate(word_ids) if wid == 0]
                if vecs:
                    vectors[start + bi] = _pool_subwords(vecs, cfg.subword_pooling)
        return vectors


def _pool_subwords(vecs: list[np.ndarray], how: str) -> np.ndarray:
    """Recombine the WordPiece fragments of a single word into one vector."""
    if how == "first":
        return np.asarray(vecs[0], dtype=np.float32)
    if how == "max":
        return np.max(np.stack(vecs, axis=0), axis=0).astype(np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
