"""End-to-end benchmark orchestrator.

Stages:

    1. Load (or download) a QASPER slice and cache it locally.
    2. Build three Qdrant collections (flat / DRAG tree / RAPTOR tree)
       unless --skip-build is passed.
    3. For each (retriever, question) compute retrieved paragraph_ids.
    4. Score retrieval against gold-evidence paragraph_ids and emit:
         - results/raw.jsonl  one row per (retriever, question)
         - results/aggregate.json   per-retriever bootstrapped CIs
         - results/report.md  human-readable summary table
         - results/plots/*.png

Run:

    python -m benchmarks.runner --num-papers 5 --max-questions 40
    python -m benchmarks.runner --smoke         # 1 paper, 2 questions, --no-llm
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from rag_lib.utils import llm_call

from .datasets import load_qasper_slice, load_slice, save_slice
from .indexing import (
    drop_collection,
    existing_paper_ids,
    index_drag_tree,
    index_flat_leaves,
    index_raptor_tree,
    max_existing_node_id,
)
from .metrics import aggregate, per_query_metrics
from .retrievers import (
    RETRIEVERS,
    collection_for,
)


DEFAULT_K_FOR_FIXED = 5
DEFAULT_BEAM_FIXED_WIDTH = 5
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / "cache"


def _ensure_slice(args) -> Path:
    """Make sure a QASPER slice is on disk; return its directory."""
    slice_dir = CACHE_DIR / f"qasper_p{args.num_papers}_q{args.max_questions}_s{args.seed}"
    papers_path = slice_dir / "papers.json"
    questions_path = slice_dir / "questions.json"
    cached_nonempty = (
        papers_path.exists()
        and questions_path.exists()
        and papers_path.stat().st_size > 2  # bigger than "[]"
        and questions_path.stat().st_size > 2
    )
    if cached_nonempty and not args.refresh_dataset:
        return slice_dir
    slice_dir.mkdir(parents=True, exist_ok=True)
    print(f"[runner] Downloading QASPER slice -> {slice_dir}")
    papers, questions = load_qasper_slice(
        num_papers=args.num_papers,
        max_questions=args.max_questions,
        seed=args.seed,
    )
    save_slice(papers, questions, slice_dir)
    print(f"[runner]   papers={len(papers)} questions={len(questions)}")
    return slice_dir


def _incremental_state(collection_name: str) -> tuple[set[str], int]:
    """Return (paper_ids already present, next safe node-id start) for a collection.

    On a missing collection: empty set, 0. Used only when --incremental is set.
    """
    existing = existing_paper_ids(collection_name)
    next_id = max_existing_node_id(collection_name) + 1
    if existing:
        print(
            f"[runner]   {collection_name}: {len(existing)} paper(s) already "
            f"indexed (next node_id={next_id})"
        )
    return existing, next_id


def _build_indices(args, papers) -> dict:
    summarizer = None if args.no_llm else llm_call
    incremental = getattr(args, "incremental", False)

    stats: dict[str, dict] = {}
    if args.rebuild:
        for c in ("bench_flat", "bench_drag", "bench_raptor"):
            print(f"[runner] Dropping collection {c}")
            drop_collection(c)

    if "flat" in args.indices:
        skip, start_nid = (
            _incremental_state("bench_flat") if incremental else (set(), 0)
        )
        print("[runner] Indexing flat leaves (bench_flat)")
        s = index_flat_leaves(
            papers,
            collection_name="bench_flat",
            skip_paper_ids=skip,
            start_node_id=start_nid,
        )
        stats["bench_flat"] = asdict(s)
        print(f"[runner]   leaves={s.leaves} took={s.seconds:.1f}s")

    if "drag" in args.indices:
        skip, start_nid = (
            _incremental_state("bench_drag") if incremental else (set(), 0)
        )
        print("[runner] Indexing DRAG tree (bench_drag)")
        s = index_drag_tree(
            papers,
            collection_name="bench_drag",
            summarizer=summarizer,
            skip_paper_ids=skip,
            start_node_id=start_nid,
        )
        stats["bench_drag"] = asdict(s)
        print(
            f"[runner]   leaves={s.leaves} parents={s.parent_nodes} "
            f"llm_calls={s.llm_calls} took={s.seconds:.1f}s"
        )

    if "raptor" in args.indices:
        skip, start_nid = (
            _incremental_state("bench_raptor") if incremental else (set(), 0)
        )
        print("[runner] Indexing RAPTOR tree (bench_raptor)")
        s = index_raptor_tree(
            papers,
            collection_name="bench_raptor",
            summarizer=summarizer,
            skip_paper_ids=skip,
            start_node_id=start_nid,
        )
        stats["bench_raptor"] = asdict(s)
        print(
            f"[runner]   leaves={s.leaves} parents={s.parent_nodes} "
            f"llm_calls={s.llm_calls} took={s.seconds:.1f}s"
        )

    return stats


def _run_retrieval(
    args,
    questions,
    retriever_names: list[str],
    raw_path: Path | None = None,
) -> list[dict]:
    raw_rows: list[dict] = []
    total = len(questions) * len(retriever_names)
    done = 0
    cross_paper = getattr(args, "cross_paper", False)
    weight_schedule = getattr(args, "weight_schedule", False)
    raw_fh = None
    if raw_path is not None:
        raw_path.write_text("", encoding="utf-8")  # truncate any stale content
        raw_fh = open(raw_path, "a", encoding="utf-8")
    for retriever in retriever_names:
        fn = RETRIEVERS[retriever]
        coll = collection_for(retriever)  # type: ignore[arg-type]
        for q in questions:
            t0 = time.perf_counter()
            try:
                res = fn(
                    query=q.text,
                    paper_id=None if cross_paper else q.paper_id,
                    k=args.k,
                    collection_name=coll,
                    weight_schedule=weight_schedule,
                )
                err = None
            except Exception as exc:
                res = None
                err = repr(exc)
            wall = time.perf_counter() - t0

            if res is None:
                row = {
                    "retriever": retriever,
                    "paper_id": q.paper_id,
                    "question_id": q.question_id,
                    "question": q.text,
                    "gold_paragraph_ids": q.gold_paragraph_ids,
                    "retrieved": [],
                    "k_returned": 0,
                    "latency_s": wall,
                    "error": err,
                    "metrics": {},
                }
            else:
                metrics = per_query_metrics(res.paragraph_ids, q.gold_paragraph_ids)
                row = {
                    "retriever": retriever,
                    "paper_id": q.paper_id,
                    "question_id": q.question_id,
                    "question": q.text,
                    "gold_paragraph_ids": q.gold_paragraph_ids,
                    "retrieved": res.paragraph_ids,
                    "k_returned": res.k_returned,
                    "latency_s": res.latency_s,
                    "error": None,
                    "metrics": metrics,
                }
            raw_rows.append(row)
            if raw_fh is not None:
                raw_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_fh.flush()
            done += 1
            if done % 10 == 0 or done == total:
                print(f"[runner]   retrieval progress {done}/{total}", flush=True)
    if raw_fh is not None:
        raw_fh.close()
    return raw_rows


def _aggregate(raw_rows: list[dict]) -> dict:
    metric_keys = sorted({k for row in raw_rows for k in row.get("metrics", {})})
    by_retriever: dict[str, dict] = {}
    for retriever in sorted({r["retriever"] for r in raw_rows}):
        subset = [r for r in raw_rows if r["retriever"] == retriever]
        agg: dict[str, dict] = {}
        for mk in metric_keys:
            values = [r["metrics"][mk] for r in subset if "metrics" in r and mk in r["metrics"]]
            a = aggregate(values, name=mk)
            agg[mk] = {"mean": a.mean, "lo95": a.lo95, "hi95": a.hi95, "n": a.n}
        latencies = [r["latency_s"] for r in subset]
        lat = aggregate(latencies, name="latency_s")
        agg["latency_s"] = {
            "mean": lat.mean,
            "lo95": lat.lo95,
            "hi95": lat.hi95,
            "n": lat.n,
        }
        ks = aggregate([float(r["k_returned"]) for r in subset], name="k_returned")
        agg["avg_k_returned"] = {
            "mean": ks.mean,
            "lo95": ks.lo95,
            "hi95": ks.hi95,
            "n": ks.n,
        }
        by_retriever[retriever] = agg
    return by_retriever


def _markdown_report(args, agg: dict, indexing_stats: dict, out_path: Path) -> None:
    key_metrics = [
        "recall@1",
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@5",
        "ndcg@10",
        "right_sized_recall",
        "avg_k_returned",
        "latency_s",
    ]
    lines: list[str] = []
    lines.append("# DRAG-with-KNEE benchmark report")
    lines.append("")
    lines.append(f"- Dataset: QASPER, papers={args.num_papers}, questions={args.max_questions}, seed={args.seed}")
    lines.append(f"- Fixed-k for non-adaptive methods: {args.k}")
    lines.append(f"- LLM-free indexing mode: {args.no_llm}")
    lines.append("")
    lines.append("## Indexing stats")
    lines.append("")
    lines.append("| Collection | Leaves | Parents | LLM calls | Seconds |")
    lines.append("|---|---:|---:|---:|---:|")
    for coll, s in indexing_stats.items():
        lines.append(
            f"| `{coll}` | {s.get('leaves',0)} | {s.get('parent_nodes',0)} | "
            f"{s.get('llm_calls',0)} | {s.get('seconds',0):.1f} |"
        )
    lines.append("")
    lines.append("## Retrieval quality (mean [95% CI])")
    lines.append("")
    header = "| Retriever | " + " | ".join(key_metrics) + " |"
    sep = "|" + "---|" * (len(key_metrics) + 1)
    lines.append(header)
    lines.append(sep)
    for retriever, metrics in agg.items():
        row = [f"`{retriever}`"]
        for mk in key_metrics:
            m = metrics.get(mk)
            if m is None:
                row.append("—")
            else:
                row.append(f"{m['mean']:.3f} [{m['lo95']:.3f}, {m['hi95']:.3f}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "Adaptive-k methods (`drag_branch`, `drag_beam_knee`, `drag_beam_sensitive_knee`) "
        "set their own number of returned chunks; `avg_k_returned` shows that choice and "
        "`right_sized_recall` measures recall over what they returned (not capped at a fixed k)."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plots(agg: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[runner] plotting skipped: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    retrievers = list(agg.keys())

    def bar(metric: str, fname: str, ylabel: str) -> None:
        means = [agg[r][metric]["mean"] for r in retrievers]
        los = [agg[r][metric]["lo95"] for r in retrievers]
        his = [agg[r][metric]["hi95"] for r in retrievers]
        err_lo = [m - lo for m, lo in zip(means, los)]
        err_hi = [hi - m for m, hi in zip(means, his)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(retrievers, means, yerr=[err_lo, err_hi], capsize=4)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    bar("recall@5", "recall_at_5.png", "Recall@5")
    bar("ndcg@10", "ndcg_at_10.png", "nDCG@10")
    bar("latency_s", "latency.png", "Latency (s)")
    bar("avg_k_returned", "avg_k_returned.png", "Avg chunks returned")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRAG-with-KNEE retrieval benchmark.")
    p.add_argument("--num-papers", type=int, default=5)
    p.add_argument("--max-questions", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refresh-dataset", action="store_true")
    p.add_argument("--rebuild", action="store_true", help="Drop collections before building")
    p.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Skip papers already present in each Qdrant collection "
            "(detected via paper_id payload). Node-id counter starts past the "
            "max existing id to avoid UUID collision. Mutually exclusive with "
            "--rebuild. Cannot recover from a paper that was only partially "
            "indexed (interrupted mid-paper) — use --rebuild for that case."
        ),
    )
    p.add_argument(
        "--indices",
        nargs="+",
        default=["flat", "drag", "raptor"],
        choices=["flat", "drag", "raptor"],
    )
    p.add_argument(
        "--skip-build", action="store_true", help="Reuse existing Qdrant collections"
    )
    p.add_argument("--no-llm", action="store_true", help="Use text-truncation instead of LLM for descriptions")
    p.add_argument("--k", type=int, default=DEFAULT_K_FOR_FIXED)
    p.add_argument(
        "--retrievers",
        nargs="+",
        default=list(RETRIEVERS.keys()),
        choices=list(RETRIEVERS.keys()),
    )
    p.add_argument(
        "--cross-paper",
        action="store_true",
        help=(
            "Search across ALL indexed papers per query (no per-paper filter). "
            "Lets DRAG's root-level knee actually pick documents. Without this "
            "flag each query is scoped to its source paper."
        ),
    )
    p.add_argument(
        "--weight-schedule",
        action="store_true",
        help=(
            "Use the BM25→semantic per-step fusion schedule for every DRAG "
            "method (find_roots / parent_vs_children / parents_vs_children). "
            "When unset, all DRAG methods use Qdrant's unweighted RRF as before. "
            "Flat retrievers (vanilla, BM25, hybrid, HyDE) and RAPTOR ignore this "
            "flag — they don't have a multi-step tree to schedule over."
        ),
    )
    p.add_argument("--smoke", action="store_true", help="Tiny config for end-to-end pipeline verification")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.num_papers = 1
        args.max_questions = 2
        args.no_llm = True
        args.rebuild = True

    if args.rebuild and args.incremental:
        print("[runner] --rebuild and --incremental are mutually exclusive.")
        return 2
    if args.skip_build and args.incremental:
        print("[runner] --incremental has no effect with --skip-build; ignoring.")
        args.incremental = False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    slice_dir = _ensure_slice(args)
    papers, questions = load_slice(slice_dir)
    print(f"[runner] Loaded {len(papers)} papers / {len(questions)} questions")
    if not questions:
        print("[runner] No questions with matchable gold evidence. Try a different seed or split.")
        return 1

    indexing_stats: dict = {}
    if not args.skip_build:
        indexing_stats = _build_indices(args, papers)
    if indexing_stats:
        with open(RESULTS_DIR / "indexing.json", "w", encoding="utf-8") as f:
            json.dump(indexing_stats, f, ensure_ascii=False, indent=2)

    raw_path = RESULTS_DIR / "raw.jsonl"
    raw_rows = _run_retrieval(args, questions, args.retrievers, raw_path=raw_path)
    print(f"[runner] wrote {raw_path}")

    agg = _aggregate(raw_rows)
    with open(RESULTS_DIR / "aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    _markdown_report(args, agg, indexing_stats, RESULTS_DIR / "report.md")
    _plots(agg, RESULTS_DIR / "plots")
    print(f"[runner] report -> {RESULTS_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
