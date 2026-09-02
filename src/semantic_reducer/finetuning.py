"""Optional continued self-supervised pretraining of the encoder.

Every other stage of this package treats the encoder as frozen (see
``encoder.py``): a single forward pass under ``torch.no_grad()``, no weights
touched. This module is the one place that trains anything, and it is opt-in
(``ReducerConfig.finetune``) precisely so the default pipeline's
"unsupervised, training-free" description stays literally true unless a
caller deliberately asks for this stage.

It uses no labels. It continues the SAME masked-language-modeling objective
the plugged-in model was already pretrained with, on the same raw corpus
already being normalized -- no new resource, no annotation. This is what
makes it consistent with the method's "no per-language resource" claim even
though it adds real training cost: a caller with an under-served language
and no suitable existing pretrained model can pretrain their own small
Transformer from scratch (self-supervised, on their own unlabeled text),
point ``model_name`` at it, and additionally continue-pretrain it further on
the exact corpus being reduced, all without any labeled data appearing
anywhere in the pipeline.

Whether this actually improves the resulting reduction is an empirical
question, not assumed here -- see
``benchmark/analysis/finetuning_validation.py`` for the frozen-vs-fine-tuned
comparison this module exists to make possible.
"""

from __future__ import annotations

import warnings


class FinetuneResult:
    """Diagnostics from one continued-pretraining run."""

    def __init__(self, epochs: int, steps: int, losses: list[float]):
        self.epochs = epochs
        self.steps = steps
        self.losses = losses  # one mean loss per epoch

    def to_dict(self) -> dict:
        return {"epochs": self.epochs, "steps": self.steps, "losses": self.losses}


def continue_pretraining(
    model,
    tokenizer,
    texts: list[str],
    device,
    *,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 32,
    max_length: int = 256,
    mlm_probability: float = 0.15,
    seed: int = 0,
    progress: bool = True,
) -> FinetuneResult:
    """Continue MLM pretraining of ``model`` on ``texts``, in place.

    Args:
        model: a ``*ForMaskedLM`` model (has an MLM head; ``forward(...,
            labels=...)`` returns a masked-LM loss). Mutated in place.
        tokenizer: the matching fast tokenizer.
        texts: raw sentences/documents -- plain strings, not pre-tokenized.
        device: torch device the model already lives on.

    Returns:
        Diagnostics (steps run, per-epoch mean loss). The caller is
        responsible for calling ``model.eval()`` afterward if it will be used
        for frozen inference next (this function leaves the model in
        ``train()`` mode when it returns, matching the state it needs while
        running).
    """
    import torch
    from tqdm import tqdm
    from transformers import DataCollatorForLanguageModeling

    texts = [t for t in texts if str(t).strip()]
    if not texts:
        raise ValueError("continue_pretraining got no non-empty texts")

    encodings = tokenizer(
        texts, truncation=True, max_length=max_length, padding=False,
        return_special_tokens_mask=True,
    )
    examples = [
        {"input_ids": encodings["input_ids"][i],
         "special_tokens_mask": encodings["special_tokens_mask"][i]}
        for i in range(len(texts))
        if len(encodings["input_ids"][i]) >= 2   # at least one real token + a special token
    ]
    if not examples:
        raise ValueError(
            "every text was too short to fine-tune on after tokenization "
            "(need at least one real token)"
        )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )

    rng = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    losses: list[float] = []
    total_steps = 0
    n_batches = (len(examples) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        order = torch.randperm(len(examples), generator=rng).tolist()
        batches = range(0, len(order), batch_size)
        if progress:
            batches = tqdm(batches, total=n_batches,
                          desc=f"Continued pretraining, epoch {epoch + 1}/{epochs}")

        epoch_loss = 0.0
        epoch_steps = 0
        for start in batches:
            idx = order[start:start + batch_size]
            batch = collator([examples[i] for i in idx])
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            epoch_steps += 1
            total_steps += 1

        if epoch_steps == 0:
            warnings.warn("continue_pretraining epoch produced zero batches",
                         RuntimeWarning, stacklevel=2)
            losses.append(float("nan"))
        else:
            losses.append(epoch_loss / epoch_steps)

    return FinetuneResult(epochs=epochs, steps=total_steps, losses=losses)
