"""Standalone metric sanity tests — stdlib only.

Run with: python -m benchmarks.test_metrics
Asserts on every check; raises SystemExit(1) on first failure.
"""

from __future__ import annotations

import math
import sys

from .metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    per_query_metrics,
    bootstrap_ci,
)


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def main() -> int:
    failures: list[str] = []

    # Recall: 2 of 3 gold in top-5
    gold = {"A", "B", "C"}
    retrieved = ["X", "A", "Y", "B", "Z"]
    if not _close(recall_at_k(retrieved, gold, 5), 2 / 3):
        failures.append("recall@5 wrong")
    if not _close(recall_at_k(retrieved, gold, 1), 0.0):
        failures.append("recall@1 (no hit) wrong")
    if not _close(recall_at_k(retrieved, gold, 2), 1 / 3):
        failures.append("recall@2 wrong")

    # Hit rate
    if hit_rate_at_k(retrieved, gold, 1) != 0.0:
        failures.append("hit@1 should be 0")
    if hit_rate_at_k(retrieved, gold, 2) != 1.0:
        failures.append("hit@2 should be 1")

    # Precision
    if not _close(precision_at_k(retrieved, gold, 5), 2 / 5):
        failures.append("precision@5 wrong")
    if precision_at_k([], gold, 5) != 0.0:
        failures.append("precision on empty wrong")

    # MRR: first gold at position 2 → 1/2
    if not _close(reciprocal_rank(retrieved, gold), 0.5):
        failures.append("reciprocal rank wrong")
    if reciprocal_rank(retrieved, gold, k=1) != 0.0:
        failures.append("reciprocal rank @ k=1 should be 0")

    # nDCG: hits at positions 2 and 4
    expected_dcg = 1 / math.log2(2 + 1) + 1 / math.log2(4 + 1)
    expected_idcg = 1 / math.log2(1 + 1) + 1 / math.log2(2 + 1) + 1 / math.log2(3 + 1)
    expected_ndcg5 = expected_dcg / expected_idcg
    if not _close(ndcg_at_k(retrieved, gold, 5), expected_ndcg5):
        failures.append(f"ndcg@5 wrong: got {ndcg_at_k(retrieved, gold, 5)}, expected {expected_ndcg5}")

    # Empty cases
    if recall_at_k([], gold, 5) != 0.0:
        failures.append("recall on empty retrieved wrong")
    if recall_at_k(retrieved, set(), 5) != 0.0:
        failures.append("recall on empty gold wrong")

    # per_query_metrics smoke
    m = per_query_metrics(retrieved, gold)
    expected_keys = {
        "k_returned",
        "right_sized_recall",
        "right_sized_hit",
        "mrr_full",
    }
    for k in (1, 3, 5, 10):
        expected_keys |= {f"recall@{k}", f"hit@{k}", f"mrr@{k}", f"ndcg@{k}"}
    missing = expected_keys - set(m.keys())
    if missing:
        failures.append(f"per_query_metrics missing keys: {missing}")
    if m["k_returned"] != 5.0:
        failures.append("k_returned wrong")
    if not _close(m["right_sized_recall"], 2 / 3):
        failures.append("right_sized_recall wrong")

    # Bootstrap CI: trivially constant inputs produce zero-width CI
    mean, lo, hi = bootstrap_ci([0.5] * 50, n_resamples=200, seed=1)
    if not (_close(mean, 0.5) and _close(lo, 0.5) and _close(hi, 0.5)):
        failures.append(f"bootstrap on constants wrong: mean={mean} lo={lo} hi={hi}")

    # Bootstrap CI on real-ish data should bracket the mean
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    mean, lo, hi = bootstrap_ci(values, n_resamples=1000, seed=0)
    truth = sum(values) / len(values)
    if not _close(mean, truth, tol=0.05):
        failures.append(f"bootstrap mean drift: {mean} vs {truth}")
    if not (lo <= truth <= hi):
        failures.append(f"CI does not bracket sample mean: [{lo}, {hi}] vs {truth}")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("metrics tests OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
