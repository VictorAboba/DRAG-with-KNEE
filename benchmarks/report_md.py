"""Post-process benchmark results into a richer analytical markdown.

Reads `benchmarks/results/aggregate.json` and `benchmarks/results/raw.jsonl`
and produces `benchmarks/results/report_full.md` — the same headline numbers
plus per-question coverage, head-to-head comparisons (hierarchy vs flat,
adaptive vs fixed-k, DRAG vs RAPTOR), and efficiency trade-off summaries.

Run:
    python -m benchmarks.report_md
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _fmt_ci(m: dict) -> str:
    return f"{m['mean']:.3f} [{m['lo95']:.3f}, {m['hi95']:.3f}]"


def _rank_by(agg: dict, metric: str, descending: bool = True) -> list[tuple[str, float]]:
    pairs = [(r, agg[r][metric]["mean"]) for r in agg if metric in agg[r]]
    pairs.sort(key=lambda x: x[1], reverse=descending)
    return pairs


def _winner(agg: dict, metric: str, descending: bool = True) -> str:
    pairs = _rank_by(agg, metric, descending)
    if not pairs:
        return "—"
    return pairs[0][0]


def _section_headline(agg: dict) -> list[str]:
    lines = ["## TL;DR", ""]
    # Find best at recall@5, mrr@10, ndcg@10
    for metric, label in [
        ("recall@5", "Recall@5"),
        ("mrr@10", "MRR@10"),
        ("ndcg@10", "nDCG@10"),
        ("right_sized_recall", "Right-sized Recall"),
    ]:
        if not any(metric in agg[r] for r in agg):
            continue
        pairs = _rank_by(agg, metric, descending=True)
        leader = pairs[0]
        runner_up = pairs[1] if len(pairs) > 1 else None
        gap = (leader[1] - runner_up[1]) if runner_up else 0.0
        runner_str = f"vs `{runner_up[0]}` at {_fmt(runner_up[1])} (Δ={_fmt(gap)})" if runner_up else ""
        lines.append(f"- **{label}**: `{leader[0]}` at {_fmt(leader[1])} {runner_str}")
    fastest = _rank_by(agg, "latency_s", descending=False)
    if fastest:
        lines.append(f"- **Fastest**: `{fastest[0][0]}` at {_fmt(fastest[0][1])} s/query")
    smallest_k = _rank_by(agg, "avg_k_returned", descending=False)
    if smallest_k:
        lines.append(f"- **Tightest selection**: `{smallest_k[0][0]}` returns {smallest_k[0][1]:.1f} chunks on average")
    lines.append("")
    return lines


def _section_full_table(agg: dict) -> list[str]:
    metrics = [
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "hit@5",
        "mrr@10",
        "ndcg@5",
        "ndcg@10",
        "right_sized_recall",
        "avg_k_returned",
        "latency_s",
    ]
    metrics = [m for m in metrics if any(m in agg[r] for r in agg)]
    lines = ["## Full retrieval table (mean [95% CI])", ""]
    lines.append("| Retriever | " + " | ".join(metrics) + " |")
    lines.append("|" + "---|" * (len(metrics) + 1))
    # Sort rows: DRAG variants first, then RAPTOR, then flat baselines
    order_prefix = {"drag_": 0, "raptor": 1, "hybrid": 2, "vanilla": 3, "bm25": 4}
    sorted_retrievers = sorted(
        agg.keys(),
        key=lambda r: (next((v for k, v in order_prefix.items() if r.startswith(k)), 9), r),
    )
    for retriever in sorted_retrievers:
        row = [f"`{retriever}`"]
        for m in metrics:
            row.append(_fmt_ci(agg[retriever][m]) if m in agg[retriever] else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _head_to_head(agg: dict, a: str, b: str, metrics: list[str], title: str) -> list[str]:
    if a not in agg or b not in agg:
        return []
    lines = [f"### {title}", "", "| Metric | `" + a + "` | `" + b + "` | Δ (a − b) |", "|---|---|---|---|"]
    for m in metrics:
        if m not in agg[a] or m not in agg[b]:
            continue
        va, vb = agg[a][m]["mean"], agg[b][m]["mean"]
        delta = va - vb
        sign = "+" if delta >= 0 else ""
        lines.append(f"| `{m}` | {_fmt(va)} | {_fmt(vb)} | {sign}{_fmt(delta)} |")
    lines.append("")
    return lines


def _section_comparisons(agg: dict) -> list[str]:
    metrics = ["recall@5", "ndcg@5", "right_sized_recall", "avg_k_returned", "latency_s"]
    lines = ["## Head-to-head comparisons", ""]
    lines += _head_to_head(
        agg,
        "drag_beam_fixed",
        "hybrid_rrf_flat",
        metrics,
        "Hierarchy lift (DRAG fixed vs flat hybrid)",
    )
    lines += _head_to_head(
        agg,
        "drag_beam_knee",
        "drag_beam_fixed",
        metrics,
        "Knee adaptive vs fixed-k (k=5)",
    )
    lines += _head_to_head(
        agg,
        "drag_beam_sensitive_knee",
        "drag_beam_knee",
        metrics,
        "Sensitive-knee (0.85) vs plain knee",
    )
    lines += _head_to_head(
        agg,
        "drag_beam_knee",
        "raptor_collapsed",
        metrics,
        "DRAG knee vs RAPTOR (both hierarchical, different construction)",
    )
    lines += _head_to_head(
        agg,
        "hybrid_rrf_flat",
        "vanilla_dense",
        metrics,
        "Dense + sparse fusion vs pure dense",
    )
    return lines


def _section_per_question(raw_rows: list[dict]) -> list[str]:
    by_q: dict[str, dict[str, dict]] = {}
    for r in raw_rows:
        q = (r["paper_id"], r["question_id"], r["question"])
        by_q.setdefault(q, {})[r["retriever"]] = r
    lines = ["## Per-question coverage", ""]
    lines.append(
        "For each question, **✓** means at least one gold paragraph appeared anywhere in the "
        "returned set (right-sized hit). Lower-case **k=N** is the number of chunks the retriever returned."
    )
    lines.append("")
    retrievers = sorted({r["retriever"] for r in raw_rows})
    header = ["Q (paper, id)"] + retrievers
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * (len(header)))
    for (paper_id, qid, qtext), row_for_retr in sorted(by_q.items()):
        q_short = qtext if len(qtext) < 60 else qtext[:57] + "…"
        cells = [f"{q_short} _({paper_id[-5:]}/{qid[:6]})_"]
        for retr in retrievers:
            row = row_for_retr.get(retr)
            if row is None:
                cells.append("—")
                continue
            hit = "✓" if row["metrics"].get("right_sized_hit", 0.0) > 0 else " "
            k = row.get("k_returned", 0)
            cells.append(f"{hit} k={k}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _section_efficiency(agg: dict) -> list[str]:
    lines = ["## Efficiency trade-offs", ""]
    pairs = []
    for r, metrics in agg.items():
        if "ndcg@10" in metrics and "latency_s" in metrics:
            pairs.append(
                {
                    "retriever": r,
                    "ndcg10": metrics["ndcg@10"]["mean"],
                    "latency_s": metrics["latency_s"]["mean"],
                    "avg_k": metrics.get("avg_k_returned", {}).get("mean", 0.0),
                }
            )
    pairs.sort(key=lambda d: d["ndcg10"], reverse=True)
    lines.append("| Retriever | nDCG@10 | latency (s) | avg k returned | nDCG per second |")
    lines.append("|---|---:|---:|---:|---:|")
    for p in pairs:
        nps = p["ndcg10"] / p["latency_s"] if p["latency_s"] > 0 else 0.0
        lines.append(
            f"| `{p['retriever']}` | {_fmt(p['ndcg10'])} | {_fmt(p['latency_s'])} | {p['avg_k']:.1f} | {_fmt(nps)} |"
        )
    lines.append("")
    return lines


def _section_indexing(stats_path: Path) -> list[str]:
    if not stats_path.exists():
        return []
    try:
        data = json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    lines = ["## Indexing one-time cost", ""]
    lines.append("| Collection | Leaves | Parents | LLM calls | Seconds |")
    lines.append("|---|---:|---:|---:|---:|")
    for coll, s in data.items():
        lines.append(
            f"| `{coll}` | {s.get('leaves',0)} | {s.get('parent_nodes',0)} | "
            f"{s.get('llm_calls',0)} | {s.get('seconds',0):.1f} |"
        )
    lines.append("")
    return lines


def build_report(
    aggregate_path: Path,
    raw_jsonl_path: Path,
    indexing_stats_path: Path | None,
    out_path: Path,
    run_meta: dict | None = None,
) -> None:
    agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
    raw_rows = [json.loads(l) for l in raw_jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    lines: list[str] = []
    lines.append("# DRAG-with-KNEE benchmark — full analytical report")
    lines.append("")
    if run_meta:
        meta_bits = [f"{k}: `{v}`" for k, v in run_meta.items()]
        lines.append("- " + "  ·  ".join(meta_bits))
        lines.append("")
    lines += _section_headline(agg)
    if indexing_stats_path is not None:
        lines += _section_indexing(indexing_stats_path)
    lines += _section_full_table(agg)
    lines += _section_comparisons(agg)
    lines += _section_efficiency(agg)
    lines += _section_per_question(raw_rows)
    lines.append("---")
    lines.append("Generated by `benchmarks/report_md.py`. Raw rows in `raw.jsonl`, aggregated stats in `aggregate.json`.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _try_load_run_meta() -> dict:
    """Pull the most-recent runner header from raw.jsonl (paper count, question count, etc)."""
    raw_path = RESULTS_DIR / "raw.jsonl"
    if not raw_path.exists():
        return {}
    rows = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return {}
    papers = {r["paper_id"] for r in rows}
    questions = {(r["paper_id"], r["question_id"]) for r in rows}
    retrievers = {r["retriever"] for r in rows}
    latencies = [r["latency_s"] for r in rows]
    return {
        "papers": len(papers),
        "questions": len(questions),
        "retrievers": len(retrievers),
        "median latency (all rows)": f"{statistics.median(latencies):.2f}s" if latencies else "—",
    }


def main() -> int:
    agg = RESULTS_DIR / "aggregate.json"
    raw = RESULTS_DIR / "raw.jsonl"
    indexing = RESULTS_DIR / "indexing.json"  # optional
    if not agg.exists() or not raw.exists():
        print(f"Missing inputs: need {agg} and {raw}")
        return 1
    out = RESULTS_DIR / "report_full.md"
    build_report(
        aggregate_path=agg,
        raw_jsonl_path=raw,
        indexing_stats_path=indexing if indexing.exists() else None,
        out_path=out,
        run_meta=_try_load_run_meta(),
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
