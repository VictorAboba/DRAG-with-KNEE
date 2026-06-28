# DRAG-with-KNEE retrieval benchmark

Compares the DRAG-with-KNEE retrievers against four baselines on QASPER
(scientific-paper QA with annotated evidence paragraphs).

## What's compared

| Retriever | Index | Hierarchy | Adaptive k | Notes |
|---|---|---|---|---|
| `vanilla_dense` | `bench_flat` | none | no | `jina-v3` cosine, top-k over leaves |
| `bm25_only` | `bench_flat` | none | no | Sparse BM25 only |
| `hybrid_rrf_flat` | `bench_flat` | none | no | Dense + BM25 fused via RRF |
| `hyde_hybrid` | `bench_flat` | none | no | HyDE (Gao et al. 2022): LLM writes a hypothetical answer, dense side embeds that |
| `raptor_collapsed` | `bench_raptor` | GMM clustering | no | RAPTOR-style cluster summaries, collapsed-tree retrieval |
| `drag_branch` | `bench_drag` | structural | yes | Existing `branch_search` |
| `drag_beam_fixed` | `bench_drag` | structural | no | `beam_search(method="fixed")` |
| `drag_beam_knee` | `bench_drag` | structural | yes | `beam_search(method="adaptive_with_knee")` |
| `drag_beam_sensitive_knee` | `bench_drag` | structural | yes | `beam_search(method="adaptive_with_sensitive_knee", sensitivity=0.85)` |
| `drag_beam_sensk_0.25` | `bench_drag` | structural | yes | sensitive_knee at 0.25 |
| `drag_beam_sensk_0.5` | `bench_drag` | structural | yes | sensitive_knee at 0.5 |
| `drag_beam_sensk_0.75` | `bench_drag` | structural | yes | sensitive_knee at 0.75 |
| `drag_beam_scheduled` | `bench_drag` | structural | no | Per-step weighted RRF: BM25 at root, dense at leaves |

Three Qdrant collections share the same QASPER paragraphs as leaves so retrievers
see identical data; only the index *shape* differs.

## Two orthogonal flags

| Flag | What it changes | Why |
|---|---|---|
| `--cross-paper` | Drops the per-paper filter so every query searches across all indexed papers | Lets DRAG's root-level knee actually pick which document(s) matter. Without this each QASPER question is scoped to its source paper and the root-level knee operates on a 1-point distribution. |
| `--weight-schedule` | Every DRAG method uses a per-step BM25→semantic schedule instead of unweighted RRF | Hypothesis: BM25 keyword matching identifies the right topic/document at root level; dense embeddings identify the right fact at leaf level. Schedule is `(0.2, 0.8) → (0.5, 0.5) → (0.8, 0.2) → (1.0, 0.0)` (dense_w, sparse_w). Flat retrievers and RAPTOR silently ignore this flag. |

The two flags compose — `--cross-paper --weight-schedule` is the most aggressive setting and the most informative single experiment for DRAG.

## Metrics

Retrieval-only — no answer generation, no LLM-as-judge.

- `recall@{1,3,5,10}` — fraction of gold evidence paragraphs found in top-k
- `hit@{1,3,5,10}` — at least one gold paragraph found
- `mrr@{1,3,5,10}` — reciprocal rank of first gold paragraph
- `ndcg@{1,3,5,10}` — rank-discounted gain
- `right_sized_recall` — recall over what the retriever actually returned (adaptive methods)
- `avg_k_returned` — characterizes the adaptive choice
- `latency_s` — per-query wall time

All means come with bootstrap 95% CIs.

Subtree-returning methods (DRAG, RAPTOR) expand each returned node to its leaf
paragraph_ids before metrics — so "the answer lies in this 3-leaf subtree"
correctly credits hits across all three leaves.

## Setup

```powershell
uv sync --extra benchmarks
cp .env.example .env  # set API_KEY, URL_BASE, MODEL
```

The LLM is only needed for the `bench_drag` and `bench_raptor` indexing step
(node descriptions / cluster summaries). Once indices exist you can rerun the
retrieval phase any number of times without touching the LLM.

## Smoke test (~30 s, no API calls)

Verifies the full pipeline — QASPER load, three indices, all 13 retrievers,
metrics, report — on a tiny slice with text-truncation in place of LLM
descriptions. Run this before spending API budget on the real run.

```powershell
python -m benchmarks.runner --smoke
```

This builds 1 paper, runs 2 questions, dumps `benchmarks/results/` with
`raw.jsonl`, `aggregate.json`, `report.md`, and `plots/*.png`.

## Full lean run (~5 papers, 30–50 questions)

```powershell
# First time: builds all three indices (LLM cost during DRAG + RAPTOR index)
python -m benchmarks.runner --num-papers 5 --max-questions 40 --rebuild

# Subsequent runs: reuse indices, change retriever set / k
python -m benchmarks.runner --skip-build --k 10
python -m benchmarks.runner --skip-build --retrievers drag_beam_knee drag_beam_sensitive_knee
```

## Indexing modes — `--rebuild` / `--incremental` / `--skip-build`

| Mode | What it does | When to use |
|---|---|---|
| (default) | Index every paper in the slice; error if collection exists | First run from empty |
| `--rebuild` | Drop all three collections, then index every paper | Schema change, want to start clean, or recover from a partial index |
| `--incremental` | Detect already-indexed papers via `paper_id` payload and skip them. Node-id counter starts past the max existing id to avoid UUID collision | Scaled up the slice (e.g. 5 → 10 papers) and want to add only the new ones without re-running the LLM on the existing 5 |
| `--skip-build` | Don't touch the indices at all, jump straight to retrieval | Iterate on retrievers / metrics with no LLM cost |

`--rebuild` and `--incremental` are mutually exclusive. `--incremental` + `--skip-build` is harmless but `--incremental` becomes a no-op (warned). `--incremental` cannot recover a paper that was *partially* indexed (interrupted mid-paper) — use `--rebuild` for that case.

```powershell
# Index the first 5 papers, then later add 5 more without re-paying the LLM:
python -m benchmarks.runner --num-papers 5  --max-questions 40 --rebuild
python -m benchmarks.runner --num-papers 10 --max-questions 60 --incremental
```

## Outputs

- `benchmarks/cache/qasper_p{N}_q{M}_s{seed}/` — frozen QASPER slice
- `benchmarks/results/raw.jsonl` — one row per (retriever, question)
- `benchmarks/results/aggregate.json` — per-retriever bootstrapped CIs
- `benchmarks/results/report.md` — markdown summary table
- `benchmarks/results/plots/*.png` — bar charts with error bars

## Reading the report

- **Hierarchy lift**: compare `hybrid_rrf_flat` (no hierarchy) vs `drag_beam_fixed` at the same k. Same retrieval scoring, different index shape.
- **Adaptive-k value**: compare `drag_beam_fixed` (manual k) vs `drag_beam_knee` (adaptive). If knee's `right_sized_recall` is close to `drag_beam_fixed@5` while `avg_k_returned` is lower, the adaptive choice is paying off.
- **Sensitivity sweep**: rerun `drag_beam_sensitive_knee` with several `--sensitivity` values (CLI arg can be added) to characterize the trade-off.
- **DRAG vs RAPTOR**: both are hierarchical. Differences come from construction (structural tree vs GMM clustering) and from beam-with-knee vs collapsed-tree retrieval.

## Node descriptions for `bench_drag`

As of the abstract+bullets refactor, leaf and parent nodes in the DRAG tree are annotated by `benchmarks/rich_descriptor.py` instead of `rag_lib.build_tree.DESCRIPTOR_SYSTEM_PROMPT`. The schema is two-form, but **leaves and parents are produced differently**:

| Node | `abstract` (→ payload `description` → dense side) | `bullets` (→ payload `keywords` → BM25 side) |
|---|---|---|
| Leaf | LLM call with `LEAF_PROMPT` (1–2 sentences, paraphrase-friendly) | LLM call: 5–8 short phrases preserving acronyms / numbers / entity names verbatim |
| Parent | LLM call with `PARENT_PROMPT` (1–2 sentence synthesis of children) | **Dedup-union of child bullets**, capped at 10. No LLM call. |

The split mirrors `rag_lib.build_tree.create_parent_node_and_update_children`, which also computes parent keywords as a union of child keywords. Asking the LLM to redo that dedup at every parent level both costs more and loses high-entropy terms; the union path is cheaper and more faithful.

The leaf prompt is tuned for academic NLP / ML papers and explicitly forbids stage-setting like "this paragraph" / "the authors" / "we". RAPTOR's parent nodes still use the old `rag_lib` summarizer prompt — that's a controlled reference: the two hierarchical methods now differ in description quality AND construction (structural width-3 vs GMM clustering), so a remaining DRAG-vs-RAPTOR gap after this change is attributable to construction, not prompt.

## Caveats / known confounds

- `bench_drag` uses LLM-enhanced two-form descriptions (abstract + bullets);
  `bench_raptor`'s leaves use raw paragraph text and parents use the old
  rag_lib summarizer. To control for the description-quality lever you could
  retrofit `index_raptor_tree` to call `describe_parent_rich` too.
- `bench_flat` uses raw paragraph text (no LLM). That's the standard baseline
  and the cleanest ablation for hierarchy.
- QASPER evidence matching uses normalize-and-substring; very short or
  pathological evidence strings may match multiple paragraphs. `datasets.py`
  reports `gold_paragraph_ids` as a set so this only over-counts gold, never
  under-counts.
- The DRAG tree's `adaptive_with_knee` mode can take a long time on papers
  with many sections because the knee analysis runs at every level.

## Reproducibility

- Dataset slicing is seeded (`--seed 0` default).
- RAPTOR's GMM uses `random_state=0`.
- All bootstrap CIs use `seed=0`.

To reproduce numbers from a saved run: keep `benchmarks/cache/` and rerun with
`--skip-build`.
