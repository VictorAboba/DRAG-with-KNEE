"""Retrieval-quality metrics for QASPER-style ground truth.

Inputs are always two lists per (query, retriever):

    retrieved: list[str]    ranked paragraph_ids, possibly truncated at k
    gold:      set[str]     paragraph_ids cited as evidence for this question

For fixed-k methods we truncate at k. For adaptive methods (DRAG knee variants)
we evaluate over what they returned, and additionally report a "right-sized"
metric (recall at k_returned).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable


# -------------------------- per-query metrics --------------------------


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for pid in top if pid in gold)
    return hits / len(gold)


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for pid in top if pid in gold)
    return hits / len(top)


def hit_rate_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    return 1.0 if any(pid in gold for pid in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: list[str], gold: set[str], k: int | None = None) -> float:
    seq = retrieved if k is None else retrieved[:k]
    for i, pid in enumerate(seq, start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top = retrieved[:k]
    dcg = 0.0
    for i, pid in enumerate(top, start=1):
        if pid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# -------------------------- aggregation --------------------------


@dataclass
class AggregatedMetric:
    name: str
    mean: float
    lo95: float
    hi95: float
    n: int


def bootstrap_ci(values: list[float], n_resamples: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return sum(values) / n, lo, hi


def aggregate(values: list[float], name: str, n_resamples: int = 1000) -> AggregatedMetric:
    mean, lo, hi = bootstrap_ci(values, n_resamples)
    return AggregatedMetric(name=name, mean=mean, lo95=lo, hi95=hi, n=len(values))


# -------------------------- full per-retriever report --------------------------


FIXED_KS = (1, 3, 5, 10)


def per_query_metrics(
    retrieved: list[str], gold: Iterable[str], ks: tuple[int, ...] = FIXED_KS
) -> dict[str, float]:
    gold_set = set(gold)
    k_returned = len(retrieved)
    out: dict[str, float] = {
        "k_returned": float(k_returned),
        "right_sized_recall": recall_at_k(retrieved, gold_set, k_returned)
        if k_returned > 0
        else 0.0,
        "right_sized_hit": hit_rate_at_k(retrieved, gold_set, k_returned)
        if k_returned > 0
        else 0.0,
        "mrr_full": reciprocal_rank(retrieved, gold_set),
    }
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, gold_set, k)
        out[f"hit@{k}"] = hit_rate_at_k(retrieved, gold_set, k)
        out[f"mrr@{k}"] = reciprocal_rank(retrieved, gold_set, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold_set, k)
    return out
