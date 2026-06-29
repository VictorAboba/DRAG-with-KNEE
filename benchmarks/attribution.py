"""Root-vs-leaf failure attribution for DRAG variants.

For each (DRAG retriever, query) we ask: when the retriever failed to put
gold in the top-K, was the cause:

  A. WRONG_PAPER         — the retriever never even brought back paragraphs
                           from a paper that contained gold. Root selection
                           failure (find_roots picked the wrong document(s)).

  B. RIGHT_PAPER_LATE    — the retriever brought back some paragraphs from
                           the right paper, but no gold paragraph appeared
                           in the top-K rank window. Beam/ranking quality
                           failure inside the right subtree.

  C. HIT                 — at least one gold paragraph in top-K. Success.

Cross-paper assumption: `paper_id` is parsed from `paragraph_id` via the
prefix before `__`. That format is set by `benchmarks/datasets.py`.

Reads `benchmarks/results/raw.jsonl`, writes a per-retriever table to
stdout. Doesn't touch Qdrant — pure post-hoc analysis.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"
DRAG_PREFIXES = ("drag_",)
# Also include RAPTOR for comparison — it's hierarchical too and might show
# a different failure mix.
ALSO_HIERARCHICAL = ("raptor_collapsed",)
K = 5


def paper_of(paragraph_id: str) -> str:
    return paragraph_id.split("__", 1)[0]


def categorize_row(row: dict) -> str:
    gold = row.get("gold_paragraph_ids") or []
    retrieved = row.get("retrieved") or []
    if not gold:
        return "NO_GOLD"  # shouldn't happen given dataset filtering

    gold_papers = {paper_of(g) for g in gold}
    topk = retrieved[:K]

    # HIT: any gold paragraph in top-K
    if set(topk) & set(gold):
        return "HIT"

    # Did we bring back any paragraph from a paper containing gold?
    retrieved_papers_in_window = {paper_of(p) for p in retrieved}
    if retrieved_papers_in_window & gold_papers:
        return "RIGHT_PAPER_LATE"  # right paper, but gold not in top-K

    return "WRONG_PAPER"


def categorize_row_rightsized(row: dict) -> str:
    """Same categories but using the retriever's full `k_returned` window
    (right-sized). Useful for adaptive DRAG variants where top-K (fixed 5)
    isn't the natural rank window."""
    gold = row.get("gold_paragraph_ids") or []
    retrieved = row.get("retrieved") or []
    if not gold:
        return "NO_GOLD"
    if set(retrieved) & set(gold):
        return "HIT"
    gold_papers = {paper_of(g) for g in gold}
    retrieved_papers = {paper_of(p) for p in retrieved}
    if retrieved_papers & gold_papers:
        return "RIGHT_PAPER_LATE"
    return "WRONG_PAPER"


def main() -> None:
    rows = [
        json.loads(l)
        for l in (RESULTS_DIR / "raw.jsonl").read_text(encoding="utf-8").splitlines()
        if l
    ]

    targets = sorted(
        {r["retriever"] for r in rows if r["retriever"].startswith(DRAG_PREFIXES)}
        | set(ALSO_HIERARCHICAL)
    )

    print(f"# Failure attribution — top-{K} window")
    print()
    print(
        f"{'retriever':28s}  {'HIT':>5s}  {'RIGHT_PAPER_LATE':>17s}  "
        f"{'WRONG_PAPER':>13s}  {'n':>4s}"
    )
    print("-" * 80)
    for retr in targets:
        subset = [r for r in rows if r["retriever"] == retr]
        cats = Counter(categorize_row(r) for r in subset)
        n = len(subset)
        hit = cats.get("HIT", 0)
        late = cats.get("RIGHT_PAPER_LATE", 0)
        wrong = cats.get("WRONG_PAPER", 0)
        print(
            f"{retr:28s}  {hit:>4d}  {late:>11d} ({late/n*100:>3.0f}%) "
            f"  {wrong:>7d} ({wrong/n*100:>3.0f}%)  {n:>4d}"
        )

    print()
    print(f"# Same, but using each retriever's full returned set (right-sized)")
    print()
    print(
        f"{'retriever':28s}  {'HIT':>5s}  {'RIGHT_PAPER_LATE':>17s}  "
        f"{'WRONG_PAPER':>13s}  {'avg_k':>6s}  {'n':>4s}"
    )
    print("-" * 90)
    for retr in targets:
        subset = [r for r in rows if r["retriever"] == retr]
        cats = Counter(categorize_row_rightsized(r) for r in subset)
        n = len(subset)
        hit = cats.get("HIT", 0)
        late = cats.get("RIGHT_PAPER_LATE", 0)
        wrong = cats.get("WRONG_PAPER", 0)
        avg_k = sum(r["k_returned"] for r in subset) / n if n else 0
        print(
            f"{retr:28s}  {hit:>4d}  {late:>11d} ({late/n*100:>3.0f}%) "
            f"  {wrong:>7d} ({wrong/n*100:>3.0f}%)  {avg_k:>6.1f}  {n:>4d}"
        )

    # Cross-DRAG aggregate: of all DRAG misses, what fraction is wrong-paper?
    print()
    print("# Across-DRAG aggregate (only drag_* retrievers, top-K window)")
    all_drag = [r for r in rows if r["retriever"].startswith(DRAG_PREFIXES)]
    cats = Counter(categorize_row(r) for r in all_drag)
    n = len(all_drag)
    print(f"  HIT             : {cats['HIT']:>5d} ({cats['HIT']/n*100:>4.1f}%)")
    print(
        f"  RIGHT_PAPER_LATE: {cats['RIGHT_PAPER_LATE']:>5d} "
        f"({cats['RIGHT_PAPER_LATE']/n*100:>4.1f}%)"
    )
    print(
        f"  WRONG_PAPER     : {cats['WRONG_PAPER']:>5d} "
        f"({cats['WRONG_PAPER']/n*100:>4.1f}%)"
    )
    total_miss = cats["RIGHT_PAPER_LATE"] + cats["WRONG_PAPER"]
    if total_miss:
        print()
        print(f"  Of {total_miss} misses, {cats['WRONG_PAPER']} ({cats['WRONG_PAPER']/total_miss*100:.1f}%) are wrong-paper")
        print(f"                  {cats['RIGHT_PAPER_LATE']} ({cats['RIGHT_PAPER_LATE']/total_miss*100:.1f}%) are right-paper-but-late")


if __name__ == "__main__":
    main()
