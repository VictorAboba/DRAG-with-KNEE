# DRAG-with-KNEE benchmark — full analytical report

- papers: `5`  ·  questions: `25`  ·  retrievers: `8`  ·  median latency (all rows): `15.01s`

## TL;DR

- **Recall@5**: `hybrid_rrf_flat` at 0.487 vs `vanilla_dense` at 0.480 (Δ=0.007)
- **MRR@10**: `hybrid_rrf_flat` at 0.363 vs `vanilla_dense` at 0.341 (Δ=0.021)
- **nDCG@10**: `drag_beam_fixed` at 0.376 vs `hybrid_rrf_flat` at 0.353 (Δ=0.023)
- **Right-sized Recall**: `drag_beam_fixed` at 0.833 vs `raptor_collapsed` at 0.800 (Δ=0.033)
- **Fastest**: `bm25_only` at 4.057 s/query
- **Tightest selection**: `bm25_only` returns 4.8 chunks on average

## Indexing one-time cost

| Collection | Leaves | Parents | LLM calls | Seconds |
|---|---:|---:|---:|---:|
| `bench_drag` | 216 | 108 | 324 | 2005.3 |
| `bench_raptor` | 216 | 15 | 15 | 1002.9 |

## Full retrieval table (mean [95% CI])

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | hit@5 | mrr@10 | ndcg@5 | ndcg@10 | right_sized_recall | avg_k_returned | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `drag_beam_fixed` | 0.113 [0.033, 0.220] | 0.273 [0.113, 0.447] | 0.433 [0.260, 0.613] | 0.593 [0.420, 0.760] | 0.480 [0.280, 0.680] | 0.319 [0.180, 0.465] | 0.323 [0.188, 0.474] | 0.376 [0.247, 0.511] | 0.833 [0.700, 0.947] | 19.360 [13.520, 26.040] | 33.159 [28.355, 37.518] |
| `drag_beam_knee` | 0.013 [0.000, 0.040] | 0.167 [0.047, 0.313] | 0.220 [0.080, 0.387] | 0.273 [0.127, 0.453] | 0.280 [0.120, 0.480] | 0.146 [0.054, 0.249] | 0.144 [0.053, 0.245] | 0.163 [0.074, 0.263] | 0.500 [0.327, 0.673] | 16.040 [9.960, 23.560] | 37.321 [32.195, 42.158] |
| `drag_beam_sensitive_knee` | 0.013 [0.000, 0.040] | 0.167 [0.040, 0.307] | 0.233 [0.080, 0.393] | 0.287 [0.120, 0.453] | 0.280 [0.120, 0.480] | 0.153 [0.053, 0.255] | 0.158 [0.053, 0.264] | 0.177 [0.071, 0.280] | 0.553 [0.380, 0.720] | 16.520 [10.360, 24.200] | 45.318 [39.285, 51.352] |
| `drag_branch` | 0.053 [0.000, 0.147] | 0.153 [0.033, 0.300] | 0.193 [0.060, 0.360] | 0.260 [0.120, 0.433] | 0.240 [0.080, 0.440] | 0.157 [0.053, 0.277] | 0.142 [0.040, 0.256] | 0.166 [0.070, 0.279] | 0.540 [0.353, 0.713] | 18.080 [11.840, 24.960] | 36.282 [28.866, 44.618] |
| `raptor_collapsed` | 0.140 [0.040, 0.260] | 0.293 [0.153, 0.453] | 0.400 [0.240, 0.580] | 0.473 [0.300, 0.647] | 0.520 [0.320, 0.720] | 0.319 [0.190, 0.468] | 0.307 [0.182, 0.457] | 0.334 [0.206, 0.473] | 0.800 [0.660, 0.920] | 25.120 [17.520, 32.880] | 5.856 [4.915, 7.045] |
| `hybrid_rrf_flat` | 0.153 [0.053, 0.273] | 0.287 [0.140, 0.447] | 0.487 [0.333, 0.653] | 0.487 [0.333, 0.653] | 0.640 [0.480, 0.800] | 0.363 [0.229, 0.513] | 0.353 [0.230, 0.497] | 0.353 [0.230, 0.497] | 0.487 [0.333, 0.653] | 5.000 [5.000, 5.000] | 11.734 [10.441, 13.048] |
| `vanilla_dense` | 0.140 [0.040, 0.280] | 0.400 [0.227, 0.580] | 0.480 [0.313, 0.660] | 0.480 [0.313, 0.660] | 0.600 [0.400, 0.800] | 0.341 [0.197, 0.480] | 0.352 [0.214, 0.496] | 0.352 [0.214, 0.496] | 0.480 [0.313, 0.660] | 5.000 [5.000, 5.000] | 9.787 [8.997, 10.662] |
| `bm25_only` | 0.080 [0.000, 0.180] | 0.233 [0.093, 0.393] | 0.433 [0.267, 0.600] | 0.433 [0.267, 0.600] | 0.520 [0.320, 0.720] | 0.253 [0.140, 0.380] | 0.283 [0.161, 0.415] | 0.283 [0.161, 0.415] | 0.433 [0.267, 0.600] | 4.840 [4.520, 5.000] | 4.057 [3.906, 4.255] |

## Head-to-head comparisons

### Hierarchy lift (DRAG fixed vs flat hybrid)

| Metric | `drag_beam_fixed` | `hybrid_rrf_flat` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.433 | 0.487 | -0.053 |
| `ndcg@5` | 0.323 | 0.353 | -0.030 |
| `right_sized_recall` | 0.833 | 0.487 | +0.347 |
| `avg_k_returned` | 19.360 | 5.000 | +14.360 |
| `latency_s` | 33.159 | 11.734 | +21.424 |

### Knee adaptive vs fixed-k (k=5)

| Metric | `drag_beam_knee` | `drag_beam_fixed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.220 | 0.433 | -0.213 |
| `ndcg@5` | 0.144 | 0.323 | -0.179 |
| `right_sized_recall` | 0.500 | 0.833 | -0.333 |
| `avg_k_returned` | 16.040 | 19.360 | -3.320 |
| `latency_s` | 37.321 | 33.159 | +4.163 |

### Sensitive-knee (0.85) vs plain knee

| Metric | `drag_beam_sensitive_knee` | `drag_beam_knee` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.233 | 0.220 | +0.013 |
| `ndcg@5` | 0.158 | 0.144 | +0.014 |
| `right_sized_recall` | 0.553 | 0.500 | +0.053 |
| `avg_k_returned` | 16.520 | 16.040 | +0.480 |
| `latency_s` | 45.318 | 37.321 | +7.997 |

### DRAG knee vs RAPTOR (both hierarchical, different construction)

| Metric | `drag_beam_knee` | `raptor_collapsed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.220 | 0.400 | -0.180 |
| `ndcg@5` | 0.144 | 0.307 | -0.163 |
| `right_sized_recall` | 0.500 | 0.800 | -0.300 |
| `avg_k_returned` | 16.040 | 25.120 | -9.080 |
| `latency_s` | 37.321 | 5.856 | +31.465 |

### Dense + sparse fusion vs pure dense

| Metric | `hybrid_rrf_flat` | `vanilla_dense` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.487 | 0.480 | +0.007 |
| `ndcg@5` | 0.353 | 0.352 | +0.001 |
| `right_sized_recall` | 0.487 | 0.480 | +0.007 |
| `avg_k_returned` | 5.000 | 5.000 | +0.000 |
| `latency_s` | 11.734 | 9.787 | +1.947 |

## Efficiency trade-offs

| Retriever | nDCG@10 | latency (s) | avg k returned | nDCG per second |
|---|---:|---:|---:|---:|
| `drag_beam_fixed` | 0.376 | 33.159 | 19.4 | 0.011 |
| `hybrid_rrf_flat` | 0.353 | 11.734 | 5.0 | 0.030 |
| `vanilla_dense` | 0.352 | 9.787 | 5.0 | 0.036 |
| `raptor_collapsed` | 0.334 | 5.856 | 25.1 | 0.057 |
| `bm25_only` | 0.283 | 4.057 | 4.8 | 0.070 |
| `drag_beam_sensitive_knee` | 0.177 | 45.318 | 16.5 | 0.004 |
| `drag_branch` | 0.166 | 36.282 | 18.1 | 0.005 |
| `drag_beam_knee` | 0.163 | 37.321 | 16.0 | 0.004 |

## Per-question coverage

For each question, **✓** means at least one gold paragraph appeared anywhere in the returned set (right-sized hit). Lower-case **k=N** is the number of chunks the retriever returned.

| Q (paper, id) | bm25_only | drag_beam_fixed | drag_beam_knee | drag_beam_sensitive_knee | drag_branch | hybrid_rrf_flat | raptor_collapsed | vanilla_dense |
|---|---|---|---|---|---|---|---|---|
| How does their decoder generate text? _(07814/0a75a5)_ |   k=5 | ✓ k=5 | ✓ k=6 | ✓ k=8 | ✓ k=15 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| Which architecture do they use for the encoder and decoder? _(07814/1b23c4)_ | ✓ k=5 | ✓ k=5 | ✓ k=3 | ✓ k=4 | ✓ k=3 | ✓ k=5 | ✓ k=26 | ✓ k=5 |
| Which dataset do they use? _(07814/fd0a3e)_ | ✓ k=5 | ✓ k=9 | ✓ k=3 | ✓ k=3 | ✓ k=3 | ✓ k=5 | ✓ k=26 |   k=5 |
| what crowdsourcing platform was used? _(05223/154a72)_ | ✓ k=5 | ✓ k=15 |   k=3 |   k=3 |   k=9 | ✓ k=5 | ✓ k=60 | ✓ k=5 |
| what is the size of their dataset? _(05223/2eb928)_ |   k=5 | ✓ k=21 |   k=6 | ✓ k=6 |   k=6 |   k=5 | ✓ k=29 |   k=5 |
| how was the data collected? _(05223/84bad9)_ | ✓ k=5 |   k=5 |   k=2 |   k=3 |   k=3 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| what dataset statistics are provided? _(05223/ad1be6)_ |   k=5 | ✓ k=19 | ✓ k=12 | ✓ k=13 | ✓ k=12 |   k=5 | ✓ k=36 |   k=5 |
| what language does this paper focus on? _(04428/2c6b50)_ |   k=5 | ✓ k=15 | ✓ k=9 | ✓ k=9 | ✓ k=9 |   k=5 | ✓ k=26 |   k=5 |
| what state of the art methods did they compare with? _(04428/326588)_ | ✓ k=5 | ✓ k=22 | ✓ k=8 | ✓ k=8 | ✓ k=17 | ✓ k=5 | ✓ k=26 | ✓ k=5 |
| by how much did their model improve? _(04428/4625cf)_ | ✓ k=5 | ✓ k=26 | ✓ k=26 | ✓ k=26 | ✓ k=26 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| what are the sizes of both datasets? _(04428/ebf0d9)_ | ✓ k=5 | ✓ k=12 |   k=1 |   k=1 | ✓ k=9 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| what evaluation metrics did they use? _(04428/f651cd)_ | ✓ k=5 | ✓ k=7 | ✓ k=3 | ✓ k=3 | ✓ k=3 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| How many different types of entities exist in the dataset? _(05828/1462eb)_ | ✓ k=5 | ✓ k=7 |   k=4 |   k=5 |   k=3 | ✓ k=5 | ✓ k=52 | ✓ k=5 |
| What is the best model? _(05828/567dc9)_ |   k=5 | ✓ k=52 | ✓ k=52 | ✓ k=52 | ✓ k=52 |   k=5 | ✓ k=37 |   k=5 |
| Which models are used to solve NER for Nepali? _(05828/6d1217)_ |   k=5 | ✓ k=31 | ✓ k=18 | ✓ k=23 | ✓ k=16 |   k=5 | ✓ k=52 | ✓ k=5 |
| Which machine learning models do they explore? _(05828/8a7615)_ |   k=5 | ✓ k=52 | ✓ k=52 | ✓ k=52 | ✓ k=52 |   k=5 |   k=5 |   k=5 |
| What is the performance improvement of the grapheme-level… _(05828/9bd080)_ | ✓ k=5 | ✓ k=52 | ✓ k=52 | ✓ k=52 | ✓ k=52 | ✓ k=5 | ✓ k=33 | ✓ k=5 |
| What is the size of the dataset? _(05828/a1b3e2)_ | ✓ k=5 | ✓ k=5 |   k=2 |   k=3 |   k=2 | ✓ k=5 | ✓ k=19 | ✓ k=5 |
| What is the source of their dataset? _(05828/bb2de2)_ | ✓ k=5 |   k=5 |   k=25 |   k=25 |   k=25 | ✓ k=5 | ✓ k=52 |   k=5 |
| What is the baseline? _(05828/cb77d6)_ |   k=1 | ✓ k=52 | ✓ k=52 | ✓ k=52 | ✓ k=52 |   k=5 |   k=5 |   k=5 |
| How many sentences does the dataset contain? _(05828/d51dc3)_ |   k=5 | ✓ k=5 |   k=25 |   k=25 |   k=25 | ✓ k=5 | ✓ k=5 |   k=5 |
| How big is the new Nepali NER dataset? _(05828/f59f1f)_ |   k=5 |   k=17 |   k=6 |   k=6 |   k=4 |   k=5 | ✓ k=52 |   k=5 |
| Do they train their model starting from a checkpoint? _(03405/7b4fb6)_ |   k=5 | ✓ k=19 |   k=18 |   k=18 |   k=16 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| What BERT model do they test? _(03405/bc31a3)_ |   k=5 | ✓ k=9 | ✓ k=7 | ✓ k=7 | ✓ k=36 |   k=5 |   k=5 | ✓ k=5 |
| How much is performance improved on NLI? _(03405/bdc91d)_ | ✓ k=5 | ✓ k=17 |   k=6 |   k=6 |   k=2 | ✓ k=5 | ✓ k=52 | ✓ k=5 |

---
Generated by `benchmarks/report_md.py`. Raw rows in `raw.jsonl`, aggregated stats in `aggregate.json`.