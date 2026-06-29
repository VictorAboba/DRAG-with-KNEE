"""Subtree-aware metrics for DRAG-Subtree.

Standard retrieval metrics (Recall@K, MRR, nDCG) treat output as a ranked
list of independent passages. DRAG-Subtree's value-prop is that output is
a small set of *coherent context units*, so the natural metrics are
different:

- subtree_recall            fraction of gold paragraphs covered by any
                            returned subtree (==right_sized_recall on
                            other retrievers, kept here for symmetry)

- coherence_score           1.0 if everything was returned in a single
                            subtree, 1/n if returned over n disjoint
                            subtrees. Higher = downstream LLM sees one
                            coherent narrative instead of fragments.

- context_size              total paragraphs returned across all subtrees.
                            Comparable to a fixed-k baseline's k_returned.

- relevance_density         gold-in-returned / context_size. "How much of
                            the context is actually relevant." High =
                            tight; low = noisy.

- avg_subtree_size          mean paragraphs per returned subtree. Lets
                            us see whether descent typically drills down
                            (small subtrees) or stops high (large subtrees).

Reads `benchmarks/results/raw.jsonl`. Only rows that carry
`subtree_groups` (i.e. produced by drag_subtree) get the subtree-specific
metrics. Other rows get them as if their `paragraph_ids` was one big
subtree (which is the right baseline for context-coherence comparison).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean


RESULTS_DIR = Path(__file__).parent / "results"


def per_row_subtree_metrics(row: dict) -> dict:
    """Compute subtree-aware metrics for a single raw.jsonl row.

    Non-subtree retrievers (no `subtree_groups` field) get treated as a
    single subtree containing all their retrieved paragraphs. That is the
    correct comparison for "coherent context": those retrievers return
    isolated paragraphs as one undifferentiated batch.
    """
    retrieved: list[str] = row.get("retrieved", []) or []
    gold: set[str] = set(row.get("gold_paragraph_ids") or [])
    groups: list[list[str]] = row.get("subtree_groups") or [retrieved]

    n_returned = len(retrieved)
    gold_hits = len(set(retrieved) & gold)

    return {
        "subtree_recall": gold_hits / len(gold) if gold else 0.0,
        "coherence_score": (1.0 / len(groups)) if groups else 0.0,
        "context_size": n_returned,
        "relevance_density": (gold_hits / n_returned) if n_returned else 0.0,
        "avg_subtree_size": (mean(len(g) for g in groups) if groups else 0.0),
        "n_subtrees": len(groups),
    }


def main() -> None:
    rows = [
        json.loads(l)
        for l in (RESULTS_DIR / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]
    # Also pull baselines for direct comparison
    baselines: dict[str, list[dict]] = {}
    for name, path in [
        ("hybrid (current)", RESULTS_DIR / "raw.jsonl"),
        ("old prompt", RESULTS_DIR / "raw.old_prompt.jsonl"),
        ("rich prompt", RESULTS_DIR / "raw.rich_prompt.jsonl"),
        ("hybrid prompt", RESULTS_DIR / "raw.hybrid_prompt.jsonl"),
    ]:
        if path.exists():
            baselines[name] = [
                json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l
            ]

    # Per retriever, mean subtree-aware metrics
    print("# Subtree-aware metrics (mean across questions)")
    print()
    print(
        f"{'source':22s}  {'retriever':28s}  "
        f"{'subtree_rec':>11s}  {'coher':>5s}  {'ctx_sz':>6s}  "
        f"{'rel_dens':>8s}  {'avg_st':>6s}  {'n_subt':>6s}"
    )
    print("-" * 110)
    for source, baseline_rows in baselines.items():
        present_retrievers = sorted({r["retriever"] for r in baseline_rows})
        # Filter to retrievers of interest
        interesting = [
            r
            for r in present_retrievers
            if r in {"hybrid_rrf_flat", "raptor_collapsed", "drag_subtree"}
            or r.startswith("drag_beam_sensk")
        ]
        for retr in interesting:
            subset = [r for r in baseline_rows if r["retriever"] == retr]
            metrics_per_q = [per_row_subtree_metrics(r) for r in subset]
            if not metrics_per_q:
                continue
            keys = metrics_per_q[0].keys()
            means = {k: mean(m[k] for m in metrics_per_q) for k in keys}
            print(
                f"{source:22s}  {retr:28s}  "
                f"{means['subtree_recall']:>11.3f}  "
                f"{means['coherence_score']:>5.3f}  "
                f"{means['context_size']:>6.1f}  "
                f"{means['relevance_density']:>8.4f}  "
                f"{means['avg_subtree_size']:>6.1f}  "
                f"{means['n_subtrees']:>6.1f}"
            )
        print()

    # If drag_subtree is in current raw.jsonl, print decision distribution
    drag_sub_rows = [r for r in rows if r["retriever"] == "drag_subtree"]
    if drag_sub_rows:
        dec_counter: Counter = Counter()
        for r in drag_sub_rows:
            for d in r.get("subtree_decisions", []) or []:
                dec_counter[d] += 1
        total = sum(dec_counter.values())
        if total:
            print("# DRAG-Subtree decision distribution (across all returned subtrees)")
            for dec, n in sorted(dec_counter.items(), key=lambda kv: -kv[1]):
                print(f"  {dec:20s}  {n:>4d}  ({n/total*100:>4.1f}%)")


if __name__ == "__main__":
    main()
