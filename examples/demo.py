"""End-to-end demonstration on a real multilingual encoder.

Run:  python examples/demo.py

Shows the three things that distinguish this method:

1. Anisotropy correction. Raw Transformer embeddings put unrelated words at
   ~0.9 cosine, so a threshold applied before correction measures the cone
   rather than meaning.
2. The linkage parameter. Sweeping lambda traces the trade-off between
   compression and semantic drift on real embeddings.
3. The diameter bound. Under complete linkage every cluster is a clique in the
   tau-graph, so no pair inside a class falls below tau -- checked, not assumed.
"""

import warnings

from semantic_reducer import (
    ReducerConfig,
    SemanticReducer,
    agglomerate,
    build_mergeable_mask,
    find_edges,
    min_internal_similarity,
)

# A small corpus with deliberate synonym families, repeated across contexts so
# type vectors are averaged over several occurrences rather than one.
CORPUS = [
    "The quick brown fox jumps over the lazy dog near the river.",
    "A fast animal leaps above a sleepy hound beside the water.",
    "Speedy foxes jump over tired dogs every single morning.",
    "The rapid runner sprinted past the slow walker on the path.",
    "A swift messenger delivered the urgent letter before sunset.",
    "The large house stood beside a big barn on the wide field.",
    "A huge building towered over the enormous warehouse downtown.",
    "The small cottage sat near a tiny shed at the narrow lane.",
    "A little cabin rested beside the minor stream in the valley.",
    "The happy children played joyfully in the sunny garden today.",
    "Cheerful students laughed together during the pleasant afternoon.",
    "The sad man walked slowly through the gloomy empty street.",
    "An unhappy woman sighed quietly in the dreary waiting room.",
    "The doctor examined the patient carefully in the busy hospital.",
    "A physician treated the sick person inside the crowded clinic.",
    "The teacher explained the difficult lesson to the young pupils.",
    "An instructor described the hard problem to the new students.",
    "The car drove quickly along the empty highway last night.",
    "An automobile travelled fast down the deserted road yesterday.",
    "The vehicle moved rapidly across the quiet bridge this morning.",
    "She bought fresh bread and warm milk at the local market.",
    "He purchased new food and cold water from the nearby shop.",
    "They acquired several goods at the busy store downtown.",
    "The book described an interesting story about ancient kings.",
    "The novel narrated a fascinating tale concerning old rulers.",
    "The text explained a compelling account regarding former monarchs.",
    "Scientists discovered a new species in the deep dark forest.",
    "Researchers found an unknown creature within the thick woods.",
    "Investigators located a strange animal inside the dense jungle.",
    "The river flowed gently beneath the old stone bridge.",
    "A stream ran softly under the ancient wooden crossing.",
    "The creek moved quietly below the aged metal structure.",
    "He spoke loudly during the important meeting this afternoon.",
    "She talked clearly throughout the significant conference today.",
    "They discussed calmly across the crucial gathering last week.",
    "The weather turned cold and the temperature dropped sharply.",
    "The climate became chilly while the heat decreased quickly.",
    "Winter arrived early bringing freezing wind and heavy snow.",
    "The computer processed the large dataset in a few seconds.",
    "The machine handled the huge collection within several moments.",
]


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    config = ReducerConfig(
        model_name="bert-base-multilingual-cased",
        threshold=0.40,   # low, to make the chaining effect visible on a small corpus
        linkage=1.0,
        min_count=1,      # small demo corpus; the default of 5 would empty it
        device=None,      # auto: CUDA -> MPS -> CPU
    )

    reducer = SemanticReducer(config=config)
    reducer.fit(CORPUS)

    banner("1. GEOMETRY -- why the threshold needs anisotropy correction")
    for key, value in reducer.geometry_report().items():
        print(f"  {key:36} {value}")
    print()
    print("  Before correction, two unrelated words already sit near this cosine.")
    print("  A threshold applied at that point would be measuring the cone.")

    banner("2. CLUSTERING at lambda = 1.0 (complete linkage)")
    for key, value in reducer.cluster_stats().items():
        print(f"  {key:36} {value}")

    print()
    print("  Sample merges (word -> representative):")
    for word, rep in reducer.sample_merges(12):
        print(f"    {word:20} -> {rep}")

    banner("3. DRIFT -- sweeping lambda on the same embeddings")

    # The encoding, correction, and neighbour search are all independent of
    # lambda, so they are done once and only the agglomeration is repeated.
    X = reducer.vectors
    tau = config.threshold
    mergeable = build_mergeable_mask(
        reducer.vocab,
        protect=config.protect,
        protect_punctuation=config.protect_punctuation,
        protect_numerals=config.protect_numerals,
    )
    sims, rows, cols = find_edges(X, threshold=tau, backend=config.backend)

    print(f"  {'lambda':>8}  {'classes':>8}  {'largest':>8}  {'worst pair':>11}  "
          f"{'bound holds':>12}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 11}  {'-' * 12}")

    runaway = None
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = agglomerate(X, sims, rows, cols, threshold=tau,
                             linkage=lam, mergeable=mergeable)
        merged = [c for c in result.components if len(c) > 1]
        worst = min((min_internal_similarity(X, c) for c in merged), default=1.0)
        largest = max((len(c) for c in result.components), default=0)
        holds = worst >= tau - 1e-6
        print(f"  {lam:>8}  {result.n_clusters:>8}  {largest:>8}  {worst:>11.4f}  "
              f"{str(holds):>12}")
        if lam == 0.0:
            runaway = max(result.components, key=len)

    print()
    print(f"  tau = {tau}, over {len(reducer.vocab)} types.")
    print("  Under complete linkage no pair inside a class falls below tau. As")
    print("  lambda drops, compression improves -- and single linkage chains until")
    print("  one class swallows most of the vocabulary, with member pairs at")
    print("  NEGATIVE cosine sitting in the same equivalence class.")

    if runaway is not None and len(runaway) > 10:
        sample = [reducer.vocab[i] for i in runaway[:14]]
        print()
        print(f"  The runaway class at lambda=0 holds {len(runaway)} of "
              f"{len(reducer.vocab)} types, beginning:")
        print(f"    {', '.join(sample)} ...")

    banner("4. POLYSEMY -- least consistently used types")
    for word, score in reducer.polysemy_report(8):
        print(f"    {word:20} {score:.4f}")
    print()
    print("  Low scores mean the word's occurrences point in many directions.")

    banner("5. GUARANTEES")
    for prop, ok in reducer.verify_guarantees().items():
        print(f"  {prop:36} {'PASS' if ok else 'FAIL'}")

    banner("6. REDUCTION IN ACTION")
    print("  NOTE: tau is set low (0.40) so that chaining is visible on a corpus")
    print("  of 40 sentences, so these merges are far more aggressive than you")
    print("  would want in practice -- unrelated words share classes here. On a")
    print("  real corpus, sweep tau and read cluster_stats() and drift_report()")
    print("  rather than adopting this value.")
    print()
    for sentence in CORPUS[:3]:
        print(f"  in : {sentence}")
        print(f"  out: {reducer.reduce(sentence)}")
        print()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
