"""Tests for the contextual encoder and device handling.

These run a real Transformer, so they use a tiny randomly-initialized BERT
rather than a pretrained multilingual model: what is under test is the encoding
and aggregation machinery, not what any particular model knows. They are skipped
when the model cannot be fetched.

The central test here is ``test_occurrence_vectors_depend_on_context``. The
method claims to build *contextual* type vectors by averaging a word's
occurrences across the corpus, as opposed to embedding words in isolation. That
claim is only true if the encoder actually produces different vectors for the
same word in different sentences, so it is verified directly rather than assumed.
"""

import warnings

import numpy as np
import pytest

from semantic_reducer import (
    ContextualEncoder,
    ReducerConfig,
    SemanticReducer,
    describe_device,
    resolve_device,
    tokenize,
)

TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"

torch = pytest.importorskip("torch", reason="encoder tests need torch")
pytest.importorskip("transformers", reason="encoder tests need transformers")


@pytest.fixture(scope="module")
def config():
    """Config for the tiny model; layers=(-1,) since it has very few layers."""
    return ReducerConfig(
        model_name=TINY_MODEL,
        layers=(-1,),
        min_count=1,
        anisotropy=False,
        batch_size=4,
        device="cpu",
    )


@pytest.fixture(scope="module")
def encoder(config):
    enc = ContextualEncoder(config)
    try:
        enc._ensure_loaded()
    except Exception as exc:                       # offline, or hub unavailable
        pytest.skip(f"could not load {TINY_MODEL}: {exc}")
    return enc


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a fast dark canine leaps above a tired hound",
    "speedy foxes jump over sleepy dogs",
    "the dog barked at the fox in the garden",
]


# --------------------------------------------------------------------------- #
class TestDeviceResolution:
    def test_explicit_cpu(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto_selects_something_usable(self):
        device = resolve_device(None)
        assert device.type in {"cuda", "mps", "cpu"}

    def test_auto_prefers_cuda_when_present(self):
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")
        assert resolve_device(None).type == "cuda"

    def test_requesting_absent_cuda_fails_loudly(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is present, so this cannot be exercised here")
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")

    def test_describe_device_is_readable(self):
        assert isinstance(describe_device(resolve_device("cpu")), str)

    def test_float16_on_cpu_falls_back_with_a_warning(self):
        from semantic_reducer.encoder import _resolve_dtype

        with pytest.warns(RuntimeWarning, match="float16"):
            dtype = _resolve_dtype("float16", torch.device("cpu"))
        assert dtype is torch.float32


# --------------------------------------------------------------------------- #
class TestTokenize:
    def test_punctuation_is_split_off(self):
        assert tokenize("the fox, quickly.") == ["the", "fox", ",", "quickly", "."]

    def test_lowercase_flag(self):
        assert tokenize("The Fox", lowercase=True) == ["the", "fox"]
        assert tokenize("The Fox") == ["The", "Fox"]

    def test_handles_non_latin_script(self):
        assert tokenize("привет мир") == ["привет", "мир"]

    def test_empty_string_yields_nothing(self):
        assert tokenize("") == []


# --------------------------------------------------------------------------- #
class TestEncodeCorpus:
    def test_counts_match_the_tokenizer(self, encoder):
        stats = encoder.encode_corpus(CORPUS, progress=False)
        expected = sum(len(tokenize(s)) for s in CORPUS)
        assert sum(stats.counts.values()) == expected

    def test_every_type_is_recorded(self, encoder):
        stats = encoder.encode_corpus(CORPUS, progress=False)
        expected = {t for s in CORPUS for t in tokenize(s)}
        assert set(stats.counts) == expected

    def test_type_vectors_have_the_model_dimension(self, encoder):
        stats = encoder.encode_corpus(CORPUS, progress=False)
        assert stats.type_vector("fox").shape == (encoder.hidden_size,)

    def test_occurrence_vectors_depend_on_context(self, config):
        """The claim that these embeddings are contextual, verified directly.

        The same word is encoded in two very different sentences. If the encoder
        were being fed isolated tokens, the two vectors would be identical.
        """
        enc = ContextualEncoder(config)
        try:
            enc._ensure_loaded()
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")

        first = enc.encode_corpus(["the bank approved my loan today"], progress=False)
        second = enc.encode_corpus(["we sat on the river bank at sunset"], progress=False)

        a = first.type_vector("bank")
        b = second.type_vector("bank")
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cosine < 0.999, (
            "identical vectors for 'bank' in different sentences would mean the "
            "encoder is not seeing context"
        )

    def test_concentration_is_a_fraction(self, encoder):
        stats = encoder.encode_corpus(CORPUS, progress=False)
        for word in stats.counts:
            assert 0.0 <= stats.concentration(word) <= 1.0 + 1e-6

    def test_a_single_occurrence_is_maximally_concentrated(self, encoder):
        """One occurrence cannot disagree with itself."""
        stats = encoder.encode_corpus(["garden"], progress=False)
        assert stats.concentration("garden") == pytest.approx(1.0, abs=1e-5)

    def test_empty_corpus_is_rejected(self, encoder):
        with pytest.raises(ValueError, match="empty"):
            encoder.encode_corpus(["", "   "], progress=False)

    def test_tokenizer_dropped_words_are_reported_separately(self, encoder):
        """A soft hyphen yields no sub-word pieces, so it can never be embedded.

        Counting that as truncation would send users to raise max_length, which
        cannot possibly recover it.
        """
        with pytest.warns(RuntimeWarning, match="no sub-word tokens"):
            stats = encoder.encode_corpus(["the \xad quick fox"], progress=False)
        assert stats.unencodable >= 1
        assert stats.truncated == 0
        assert "\xad" not in stats.counts

    def test_clean_text_reports_no_losses(self, encoder):
        stats = encoder.encode_corpus(CORPUS, progress=False)
        assert stats.truncated == 0
        assert stats.unencodable == 0

    def test_truncation_is_reported(self, config):
        enc = ContextualEncoder(ReducerConfig(**{**config.to_dict(), "max_length": 8}))
        try:
            enc._ensure_loaded()
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")
        long_sentence = " ".join(f"word{i}" for i in range(50))
        with pytest.warns(RuntimeWarning, match="truncation"):
            stats = enc.encode_corpus([long_sentence], progress=False)
        assert stats.truncated > 0

    @pytest.mark.parametrize("pooling", ["mean", "first", "max"])
    def test_subword_pooling_strategies_all_work(self, config, pooling):
        enc = ContextualEncoder(ReducerConfig(**{**config.to_dict(),
                                                 "subword_pooling": pooling}))
        try:
            enc._ensure_loaded()
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")
        stats = enc.encode_corpus(CORPUS, progress=False)
        assert np.isfinite(stats.type_vector("fox")).all()

    def test_batch_size_does_not_change_results(self, config):
        """Batching is an efficiency detail and must not affect the vectors."""
        enc = ContextualEncoder(config)
        try:
            enc._ensure_loaded()
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")
        one = enc.encode_corpus(CORPUS, batch_size=1, progress=False)
        many = enc.encode_corpus(CORPUS, batch_size=8, progress=False)
        assert np.allclose(one.type_vector("fox"), many.type_vector("fox"), atol=1e-4)


# --------------------------------------------------------------------------- #
class TestCorpusStatisticsMerge:
    def test_merging_two_passes_sums_counts(self, encoder):
        a = encoder.encode_corpus(CORPUS[:2], progress=False)
        b = encoder.encode_corpus(CORPUS[2:], progress=False)
        whole = encoder.encode_corpus(CORPUS, progress=False)
        merged = a.merge(b)
        assert merged.counts == whole.counts

    def test_merging_matches_a_single_pass(self, encoder):
        a = encoder.encode_corpus(CORPUS[:2], progress=False)
        b = encoder.encode_corpus(CORPUS[2:], progress=False)
        whole = encoder.encode_corpus(CORPUS, progress=False)
        merged = a.merge(b)
        assert np.allclose(merged.type_vector("fox"), whole.type_vector("fox"), atol=1e-5)


# --------------------------------------------------------------------------- #
class TestEndToEndFit:
    def test_fit_produces_a_valid_system(self, config):
        reducer = SemanticReducer(config=config)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # tiny vocabulary
                reducer.fit(CORPUS, progress=False)
        except Exception as exc:
            if "tiny-random" in str(exc) or "Connection" in str(exc):
                pytest.skip(f"could not load {TINY_MODEL}: {exc}")
            raise

        assert reducer.vocab
        assert all(reducer.verify_guarantees().values())
        assert reducer.reduce(CORPUS[0])

    def test_fit_then_save_and_load(self, config, tmp_path):
        reducer = SemanticReducer(config=config)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reducer.fit(CORPUS, progress=False)
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")

        path = reducer.save(tmp_path / "sys.json")
        loaded = SemanticReducer.load(path)
        assert loaded.reduce(CORPUS[0]) == reducer.reduce(CORPUS[0])

    def test_min_count_filters_the_vocabulary(self, config):
        strict = ReducerConfig(**{**config.to_dict(), "min_count": 3})
        reducer = SemanticReducer(config=strict)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reducer.fit(CORPUS, progress=False)
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")
        assert all(reducer.counts[w] >= 3 for w in reducer.vocab)


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
class TestDeviceAgnosticism:
    """The same corpus must give the same answer on CPU and GPU."""

    def test_cpu_and_cuda_agree_on_type_vectors(self, config):
        cpu_cfg = ReducerConfig(**{**config.to_dict(), "device": "cpu"})
        gpu_cfg = ReducerConfig(**{**config.to_dict(), "device": "cuda"})

        try:
            cpu = ContextualEncoder(cpu_cfg).encode_corpus(CORPUS, progress=False)
            gpu = ContextualEncoder(gpu_cfg).encode_corpus(CORPUS, progress=False)
        except Exception as exc:
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")

        for word in sorted(cpu.counts):
            assert np.allclose(
                cpu.type_vector(word), gpu.type_vector(word), atol=1e-4
            ), f"type vector for {word!r} differs between CPU and CUDA"

    def test_cpu_and_cuda_produce_the_same_map(self, config):
        results = {}
        for device in ("cpu", "cuda"):
            cfg = ReducerConfig(**{**config.to_dict(), "device": device})
            reducer = SemanticReducer(config=cfg)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    reducer.fit(CORPUS, progress=False)
            except Exception as exc:
                pytest.skip(f"could not load {TINY_MODEL}: {exc}")
            results[device] = reducer.reduction_map
        assert results["cpu"] == results["cuda"]
