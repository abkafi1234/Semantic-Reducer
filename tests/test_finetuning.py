"""Tests for the opt-in continued-pretraining (fine-tuning) stage.

Uses the same tiny randomly-initialized BERT as test_encoder.py, for the same
reason: what is under test is the training/wiring machinery, not what any
particular model learns. Skipped when the model cannot be fetched.
"""

import pytest

from semantic_reducer import ContextualEncoder, ReducerConfig, SemanticReducer

TINY_MODEL = "hf-internal-testing/tiny-random-BertModel"

torch = pytest.importorskip("torch", reason="finetuning tests need torch")
pytest.importorskip("transformers", reason="finetuning tests need transformers")

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "a fast dark canine leaps above a tired hound",
    "speedy foxes jump over sleepy dogs",
    "the dog barked at the fox in the garden",
    "every dog has its day in the sun",
    "foxes and dogs rarely share the same den",
]


@pytest.fixture(scope="module")
def finetune_config():
    return ReducerConfig(
        model_name=TINY_MODEL,
        layers=(-1,),
        min_count=1,
        anisotropy=False,
        batch_size=4,
        device="cpu",
        finetune=True,
        finetune_epochs=1,
        finetune_batch_size=2,
    )


def _skip_if_unavailable(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                          # offline, or hub unavailable
        if "tiny-random" in str(exc) or "Connection" in str(exc):
            pytest.skip(f"could not load {TINY_MODEL}: {exc}")
        raise


class TestConfigValidation:
    def test_default_finetune_is_off(self):
        assert ReducerConfig().finetune is False

    def test_rejects_zero_epochs(self):
        with pytest.raises(ValueError, match="finetune_epochs"):
            ReducerConfig(finetune=True, finetune_epochs=0)

    def test_rejects_nonpositive_lr(self):
        with pytest.raises(ValueError, match="finetune_lr"):
            ReducerConfig(finetune=True, finetune_lr=0.0)

    def test_rejects_out_of_range_mlm_probability(self):
        with pytest.raises(ValueError, match="finetune_mlm_probability"):
            ReducerConfig(finetune=True, finetune_mlm_probability=1.5)


class TestEncoderLoading:
    def test_finetune_true_loads_mlm_head(self, finetune_config):
        enc = ContextualEncoder(finetune_config)
        _skip_if_unavailable(enc._ensure_loaded)
        assert enc._mlm_model is not None
        assert enc._model is enc._mlm_model.base_model

    def test_finetune_false_has_no_mlm_head(self):
        cfg = ReducerConfig(model_name=TINY_MODEL, layers=(-1,), device="cpu")
        enc = ContextualEncoder(cfg)
        _skip_if_unavailable(enc._ensure_loaded)
        assert enc._mlm_model is None


class TestContinuePretraining:
    def test_weights_actually_change(self, finetune_config):
        enc = ContextualEncoder(finetune_config)
        _skip_if_unavailable(enc._ensure_loaded)

        before = next(enc._mlm_model.parameters()).detach().clone()
        _skip_if_unavailable(enc.finetune_on_corpus, CORPUS, progress=False)
        after = next(enc._mlm_model.parameters()).detach().clone()

        assert not torch.equal(before, after), (
            "fine-tuning ran but the model's first parameter tensor is "
            "bit-identical -- the training step did not actually update anything"
        )

    def test_model_is_eval_mode_after_finetuning(self, finetune_config):
        enc = ContextualEncoder(finetune_config)
        _skip_if_unavailable(enc._ensure_loaded)
        _skip_if_unavailable(enc.finetune_on_corpus, CORPUS, progress=False)
        assert not enc._mlm_model.training

    def test_result_reports_one_loss_per_epoch(self):
        cfg = ReducerConfig(
            model_name=TINY_MODEL, layers=(-1,), device="cpu",
            finetune=True, finetune_epochs=2, finetune_batch_size=2,
        )
        enc = ContextualEncoder(cfg)
        _skip_if_unavailable(enc._ensure_loaded)
        result = _skip_if_unavailable(enc.finetune_on_corpus, CORPUS, progress=False)
        assert result.epochs == 2
        assert len(result.losses) == 2
        assert all(isinstance(v, float) for v in result.losses)

    def test_rejects_empty_corpus(self, finetune_config):
        enc = ContextualEncoder(finetune_config)
        _skip_if_unavailable(enc._ensure_loaded)
        with pytest.raises(ValueError, match="no non-empty texts"):
            enc.finetune_on_corpus(["", "   "], progress=False)

    def test_finetune_on_corpus_without_config_flag_raises(self):
        cfg = ReducerConfig(model_name=TINY_MODEL, layers=(-1,), device="cpu")
        enc = ContextualEncoder(cfg)
        with pytest.raises(RuntimeError, match="finetune_on_corpus"):
            enc.finetune_on_corpus(CORPUS)


class TestFitWithFinetuning:
    def test_fit_produces_a_valid_map(self, finetune_config):
        reducer = SemanticReducer(finetune_config)
        _skip_if_unavailable(reducer.fit, CORPUS, progress=False)

        # Idempotence still holds: every value in the map is itself a fixed
        # point, exactly as it must for the frozen-encoder path.
        for word, rep in reducer.reduction_map.items():
            assert reducer.reduction_map[rep] == rep

    def test_finetune_report_is_populated(self, finetune_config):
        reducer = SemanticReducer(finetune_config)
        _skip_if_unavailable(reducer.fit, CORPUS, progress=False)
        report = reducer.finetune_report()
        assert report["epochs"] == finetune_config.finetune_epochs
        assert len(report["losses"]) == finetune_config.finetune_epochs

    def test_default_fit_has_empty_finetune_report(self):
        cfg = ReducerConfig(model_name=TINY_MODEL, layers=(-1,), min_count=1,
                            anisotropy=False, device="cpu")
        reducer = SemanticReducer(cfg)
        _skip_if_unavailable(reducer.fit, CORPUS, progress=False)
        assert reducer.finetune_report() == {}
