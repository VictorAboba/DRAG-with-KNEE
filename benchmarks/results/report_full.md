# DRAG-with-KNEE benchmark — full analytical report

- papers: `10`  ·  questions: `35`  ·  retrievers: `13`  ·  median latency (all rows): `41.68s`

## TL;DR

- **Recall@5**: `hybrid_rrf_flat` at 0.387 vs `vanilla_dense` at 0.374 (Δ=0.012)
- **MRR@10**: `hybrid_rrf_flat` at 0.301 vs `raptor_collapsed` at 0.282 (Δ=0.019)
- **nDCG@10**: `hybrid_rrf_flat` at 0.295 vs `vanilla_dense` at 0.273 (Δ=0.022)
- **Right-sized Recall**: `drag_beam_sensk_0.25` at 0.666 vs `drag_beam_sensk_0.5` at 0.500 (Δ=0.165)
- **Fastest**: `bm25_only` at 3.136 s/query
- **Tightest selection**: `bm25_only` returns 5.0 chunks on average

## Indexing one-time cost

| Collection | Leaves | Parents | LLM calls | Seconds |
|---|---:|---:|---:|---:|
| `bench_flat` | 530 | 0 | 0 | 719.7 |
| `bench_drag` | 530 | 268 | 798 | 6691.8 |
| `bench_raptor` | 530 | 32 | 32 | 2314.6 |

## Full retrieval table (mean [95% CI])

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | hit@5 | mrr@10 | ndcg@5 | ndcg@10 | right_sized_recall | avg_k_returned | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `drag_beam_fixed` | 0.030 [0.000, 0.073] | 0.070 [0.010, 0.150] | 0.070 [0.010, 0.150] | 0.089 [0.019, 0.175] | 0.114 [0.029, 0.229] | 0.098 [0.010, 0.213] | 0.066 [0.013, 0.139] | 0.075 [0.014, 0.152] | 0.279 [0.138, 0.441] | 48.000 [33.514, 62.143] | 46.972 [43.876, 50.470] |
| `drag_beam_knee` | 0.030 [0.000, 0.073] | 0.070 [0.010, 0.150] | 0.070 [0.010, 0.150] | 0.089 [0.019, 0.175] | 0.114 [0.029, 0.229] | 0.103 [0.014, 0.221] | 0.070 [0.013, 0.146] | 0.078 [0.018, 0.157] | 0.358 [0.212, 0.514] | 38.343 [26.000, 52.000] | 50.171 [44.141, 56.369] |
| `drag_beam_scheduled` | 0.010 [0.000, 0.029] | 0.019 [0.000, 0.057] | 0.019 [0.000, 0.057] | 0.086 [0.010, 0.181] | 0.029 [0.000, 0.086] | 0.041 [0.003, 0.111] | 0.022 [0.000, 0.066] | 0.046 [0.004, 0.103] | 0.390 [0.238, 0.571] | 50.057 [34.629, 66.514] | 43.897 [40.345, 48.123] |
| `drag_beam_sensitive_knee` | 0.020 [0.000, 0.060] | 0.070 [0.010, 0.150] | 0.070 [0.010, 0.150] | 0.089 [0.019, 0.175] | 0.114 [0.029, 0.229] | 0.084 [0.010, 0.186] | 0.061 [0.008, 0.131] | 0.070 [0.014, 0.143] | 0.300 [0.160, 0.459] | 43.371 [29.314, 57.886] | 77.543 [70.872, 84.057] |
| `drag_beam_sensk_0.25` | 0.020 [0.000, 0.060] | 0.026 [0.000, 0.069] | 0.119 [0.036, 0.230] | 0.190 [0.090, 0.311] | 0.171 [0.057, 0.314] | 0.101 [0.032, 0.195] | 0.074 [0.022, 0.138] | 0.103 [0.045, 0.170] | 0.666 [0.514, 0.811] | 203.600 [174.343, 230.771] | 167.288 [148.333, 188.007] |
| `drag_beam_sensk_0.5` | 0.020 [0.000, 0.060] | 0.060 [0.000, 0.137] | 0.117 [0.029, 0.231] | 0.155 [0.055, 0.270] | 0.143 [0.029, 0.286] | 0.089 [0.021, 0.182] | 0.077 [0.018, 0.149] | 0.091 [0.032, 0.164] | 0.500 [0.355, 0.664] | 95.657 [72.171, 119.514] | 91.885 [81.558, 103.788] |
| `drag_beam_sensk_0.75` | 0.020 [0.000, 0.060] | 0.060 [0.000, 0.137] | 0.060 [0.000, 0.137] | 0.089 [0.019, 0.175] | 0.086 [0.000, 0.200] | 0.073 [0.007, 0.175] | 0.052 [0.000, 0.123] | 0.065 [0.012, 0.135] | 0.300 [0.160, 0.459] | 41.457 [28.886, 55.229] | 62.043 [57.319, 66.872] |
| `drag_branch` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.048 [0.000, 0.114] | 0.000 [0.000, 0.000] | 0.011 [0.000, 0.023] | 0.000 [0.000, 0.000] | 0.018 [0.000, 0.042] | 0.458 [0.306, 0.620] | 97.857 [80.229, 115.257] | 92.031 [79.116, 105.675] |
| `raptor_collapsed` | 0.163 [0.063, 0.277] | 0.263 [0.134, 0.406] | 0.334 [0.197, 0.486] | 0.349 [0.214, 0.497] | 0.400 [0.257, 0.571] | 0.282 [0.151, 0.423] | 0.263 [0.146, 0.388] | 0.268 [0.155, 0.393] | 0.487 [0.339, 0.640] | 14.657 [10.371, 19.771] | 6.511 [6.413, 6.618] |
| `hybrid_rrf_flat` | 0.163 [0.063, 0.277] | 0.315 [0.171, 0.477] | 0.387 [0.243, 0.543] | 0.387 [0.243, 0.543] | 0.457 [0.286, 0.629] | 0.301 [0.170, 0.443] | 0.295 [0.175, 0.427] | 0.295 [0.175, 0.427] | 0.387 [0.243, 0.543] | 5.000 [5.000, 5.000] | 7.478 [7.412, 7.550] |
| `vanilla_dense` | 0.120 [0.034, 0.220] | 0.297 [0.157, 0.443] | 0.374 [0.231, 0.520] | 0.374 [0.231, 0.520] | 0.429 [0.286, 0.600] | 0.273 [0.149, 0.405] | 0.273 [0.158, 0.384] | 0.273 [0.158, 0.384] | 0.374 [0.231, 0.520] | 5.000 [5.000, 5.000] | 7.385 [7.235, 7.531] |
| `bm25_only` | 0.086 [0.014, 0.186] | 0.187 [0.072, 0.320] | 0.244 [0.114, 0.389] | 0.244 [0.114, 0.389] | 0.286 [0.143, 0.457] | 0.175 [0.076, 0.291] | 0.179 [0.082, 0.296] | 0.179 [0.082, 0.296] | 0.244 [0.114, 0.389] | 5.000 [5.000, 5.000] | 3.136 [3.077, 3.203] |
| `hyde_hybrid` | 0.072 [0.010, 0.158] | 0.210 [0.096, 0.339] | 0.253 [0.131, 0.396] | 0.253 [0.131, 0.396] | 0.314 [0.171, 0.486] | 0.188 [0.090, 0.302] | 0.182 [0.092, 0.288] | 0.182 [0.092, 0.288] | 0.253 [0.131, 0.396] | 5.000 [5.000, 5.000] | 11.947 [11.607, 12.350] |

## Head-to-head comparisons

### Hierarchy lift (DRAG fixed vs flat hybrid)

| Metric | `drag_beam_fixed` | `hybrid_rrf_flat` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.070 | 0.387 | -0.317 |
| `ndcg@5` | 0.066 | 0.295 | -0.229 |
| `right_sized_recall` | 0.279 | 0.387 | -0.108 |
| `avg_k_returned` | 48.000 | 5.000 | +43.000 |
| `latency_s` | 46.972 | 7.478 | +39.493 |

### Knee adaptive vs fixed-k (k=5)

| Metric | `drag_beam_knee` | `drag_beam_fixed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.070 | 0.070 | +0.000 |
| `ndcg@5` | 0.070 | 0.066 | +0.004 |
| `right_sized_recall` | 0.358 | 0.279 | +0.079 |
| `avg_k_returned` | 38.343 | 48.000 | -9.657 |
| `latency_s` | 50.171 | 46.972 | +3.199 |

### Sensitive-knee (0.85) vs plain knee

| Metric | `drag_beam_sensitive_knee` | `drag_beam_knee` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.070 | 0.070 | +0.000 |
| `ndcg@5` | 0.061 | 0.070 | -0.009 |
| `right_sized_recall` | 0.300 | 0.358 | -0.057 |
| `avg_k_returned` | 43.371 | 38.343 | +5.029 |
| `latency_s` | 77.543 | 50.171 | +27.371 |

### DRAG knee vs RAPTOR (both hierarchical, different construction)

| Metric | `drag_beam_knee` | `raptor_collapsed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.070 | 0.334 | -0.265 |
| `ndcg@5` | 0.070 | 0.263 | -0.193 |
| `right_sized_recall` | 0.358 | 0.487 | -0.129 |
| `avg_k_returned` | 38.343 | 14.657 | +23.686 |
| `latency_s` | 50.171 | 6.511 | +43.660 |

### Dense + sparse fusion vs pure dense

| Metric | `hybrid_rrf_flat` | `vanilla_dense` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.387 | 0.374 | +0.012 |
| `ndcg@5` | 0.295 | 0.273 | +0.022 |
| `right_sized_recall` | 0.387 | 0.374 | +0.012 |
| `avg_k_returned` | 5.000 | 5.000 | +0.000 |
| `latency_s` | 7.478 | 7.385 | +0.093 |

## Efficiency trade-offs

| Retriever | nDCG@10 | latency (s) | avg k returned | nDCG per second |
|---|---:|---:|---:|---:|
| `hybrid_rrf_flat` | 0.295 | 7.478 | 5.0 | 0.039 |
| `vanilla_dense` | 0.273 | 7.385 | 5.0 | 0.037 |
| `raptor_collapsed` | 0.268 | 6.511 | 14.7 | 0.041 |
| `hyde_hybrid` | 0.182 | 11.947 | 5.0 | 0.015 |
| `bm25_only` | 0.179 | 3.136 | 5.0 | 0.057 |
| `drag_beam_sensk_0.25` | 0.103 | 167.288 | 203.6 | 0.001 |
| `drag_beam_sensk_0.5` | 0.091 | 91.885 | 95.7 | 0.001 |
| `drag_beam_knee` | 0.078 | 50.171 | 38.3 | 0.002 |
| `drag_beam_fixed` | 0.075 | 46.972 | 48.0 | 0.002 |
| `drag_beam_sensitive_knee` | 0.070 | 77.543 | 43.4 | 0.001 |
| `drag_beam_sensk_0.75` | 0.065 | 62.043 | 41.5 | 0.001 |
| `drag_beam_scheduled` | 0.046 | 43.897 | 50.1 | 0.001 |
| `drag_branch` | 0.018 | 92.031 | 97.9 | 0.000 |

## Per-question coverage

For each question, **✓** means at least one gold paragraph appeared anywhere in the returned set (right-sized hit). Lower-case **k=N** is the number of chunks the retriever returned.

| Q (paper, id) | bm25_only | drag_beam_fixed | drag_beam_knee | drag_beam_scheduled | drag_beam_sensitive_knee | drag_beam_sensk_0.25 | drag_beam_sensk_0.5 | drag_beam_sensk_0.75 | drag_branch | hybrid_rrf_flat | hyde_hybrid | raptor_collapsed | vanilla_dense |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| How do they detect spammers? _(08504/3cd185)_ |   k=5 | ✓ k=31 | ✓ k=27 | ✓ k=59 | ✓ k=27 | ✓ k=215 | ✓ k=34 | ✓ k=27 | ✓ k=106 |   k=5 |   k=5 | ✓ k=32 |   k=5 |
| What is the benchmark dataset and is its quality high? _(08504/a1645d)_ |   k=5 |   k=191 |   k=60 |   k=198 |   k=60 |   k=342 |   k=69 |   k=63 |   k=195 |   k=5 |   k=5 |   k=5 |   k=5 |
| LDA is an unsupervised method; is this paper introducing … _(08504/dac087)_ |   k=5 |   k=61 |   k=27 |   k=33 |   k=31 | ✓ k=185 | ✓ k=71 |   k=31 |   k=35 |   k=5 |   k=5 | ✓ k=32 |   k=5 |
| What parallel corpus did they use? _(08198/602403)_ | ✓ k=5 |   k=49 |   k=9 |   k=75 |   k=76 |   k=294 |   k=158 |   k=97 |   k=96 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| Is pre-training effective in their evaluation? _(08198/8ce115)_ | ✓ k=5 |   k=15 |   k=15 |   k=16 |   k=15 |   k=193 |   k=35 |   k=20 |   k=56 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| How does their decoder generate text? _(07814/0a75a5)_ |   k=5 | ✓ k=7 | ✓ k=6 | ✓ k=7 | ✓ k=18 | ✓ k=264 | ✓ k=56 | ✓ k=15 | ✓ k=12 |   k=5 |   k=5 |   k=11 |   k=5 |
| Which architecture do they use for the encoder and decoder? _(07814/1b23c4)_ | ✓ k=5 | ✓ k=92 | ✓ k=26 | ✓ k=52 | ✓ k=65 | ✓ k=280 | ✓ k=234 | ✓ k=65 | ✓ k=74 | ✓ k=5 | ✓ k=5 | ✓ k=22 |   k=5 |
| Which dataset do they use? _(07814/fd0a3e)_ |   k=5 |   k=15 |   k=60 |   k=73 |   k=141 | ✓ k=122 |   k=39 |   k=16 |   k=96 |   k=5 |   k=5 |   k=5 |   k=5 |
| what crowdsourcing platform was used? _(05223/154a72)_ | ✓ k=5 | ✓ k=98 | ✓ k=98 | ✓ k=72 | ✓ k=101 | ✓ k=134 | ✓ k=224 | ✓ k=111 | ✓ k=129 | ✓ k=5 | ✓ k=5 | ✓ k=29 | ✓ k=5 |
| what is the size of their dataset? _(05223/2eb928)_ |   k=5 |   k=5 |   k=6 |   k=5 |   k=3 |   k=34 |   k=23 |   k=10 | ✓ k=127 |   k=5 |   k=5 |   k=21 |   k=5 |
| how was the data collected? _(05223/84bad9)_ | ✓ k=5 |   k=131 |   k=75 |   k=150 |   k=75 | ✓ k=332 |   k=144 |   k=145 | ✓ k=178 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| what dataset statistics are provided? _(05223/ad1be6)_ |   k=5 |   k=12 | ✓ k=127 | ✓ k=94 |   k=3 | ✓ k=157 |   k=6 |   k=4 | ✓ k=139 |   k=5 |   k=5 |   k=5 |   k=5 |
| what language does this paper focus on? _(04428/2c6b50)_ |   k=5 |   k=93 |   k=83 |   k=127 |   k=84 |   k=257 |   k=144 |   k=84 |   k=162 |   k=5 |   k=5 |   k=5 |   k=5 |
| what state of the art methods did they compare with? _(04428/326588)_ |   k=5 |   k=13 |   k=12 |   k=15 |   k=18 | ✓ k=252 |   k=43 |   k=21 |   k=68 |   k=5 |   k=5 |   k=5 |   k=5 |
| by how much did their model improve? _(04428/4625cf)_ |   k=5 |   k=41 |   k=12 | ✓ k=20 |   k=52 | ✓ k=239 | ✓ k=155 |   k=52 |   k=27 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| what are the sizes of both datasets? _(04428/ebf0d9)_ |   k=5 |   k=5 |   k=4 |   k=13 |   k=6 | ✓ k=24 | ✓ k=22 |   k=5 |   k=68 | ✓ k=5 |   k=5 | ✓ k=21 | ✓ k=5 |
| what evaluation metrics did they use? _(04428/f651cd)_ | ✓ k=5 |   k=35 |   k=38 |   k=7 |   k=9 |   k=313 |   k=70 |   k=50 |   k=168 | ✓ k=5 | ✓ k=5 | ✓ k=21 | ✓ k=5 |
| what was the baseline? _(03060/761de1)_ |   k=5 |   k=20 |   k=18 |   k=9 |   k=19 | ✓ k=195 |   k=33 |   k=23 |   k=67 |   k=5 |   k=5 |   k=5 |   k=5 |
| How many different types of entities exist in the dataset? _(05828/1462eb)_ |   k=5 | ✓ k=41 | ✓ k=5 | ✓ k=67 | ✓ k=41 | ✓ k=134 | ✓ k=108 | ✓ k=59 | ✓ k=42 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| What is the best model? _(05828/567dc9)_ |   k=5 |   k=31 |   k=12 |   k=25 |   k=16 | ✓ k=109 |   k=53 |   k=18 |   k=98 |   k=5 |   k=5 |   k=5 |   k=5 |
| Which models are used to solve NER for Nepali? _(05828/6d1217)_ |   k=5 | ✓ k=31 | ✓ k=27 | ✓ k=31 | ✓ k=29 | ✓ k=229 | ✓ k=47 | ✓ k=29 | ✓ k=92 |   k=5 |   k=5 |   k=5 | ✓ k=5 |
| Which machine learning models do they explore? _(05828/8a7615)_ |   k=5 |   k=135 |   k=154 |   k=49 |   k=172 |   k=272 |   k=193 |   k=156 |   k=40 |   k=5 |   k=5 |   k=11 |   k=5 |
| What is the performance improvement of the grapheme-level… _(05828/9bd080)_ | ✓ k=5 | ✓ k=123 | ✓ k=104 | ✓ k=90 | ✓ k=104 | ✓ k=208 | ✓ k=110 | ✓ k=131 | ✓ k=131 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| What is the size of the dataset? _(05828/a1b3e2)_ |   k=5 |   k=5 |   k=2 |   k=5 |   k=11 |   k=37 |   k=19 |   k=11 |   k=95 |   k=5 |   k=5 | ✓ k=21 |   k=5 |
| What is the source of their dataset? _(05828/bb2de2)_ |   k=5 |   k=13 |   k=60 |   k=64 |   k=141 |   k=206 |   k=171 |   k=14 |   k=144 |   k=5 |   k=5 |   k=5 |   k=5 |
| What is the baseline? _(05828/cb77d6)_ |   k=5 |   k=113 |   k=51 | ✓ k=164 |   k=10 | ✓ k=210 | ✓ k=310 |   k=10 |   k=111 |   k=5 |   k=5 |   k=5 |   k=5 |
| How many sentences does the dataset contain? _(05828/d51dc3)_ |   k=5 |   k=13 |   k=1 |   k=13 |   k=5 |   k=144 |   k=101 |   k=3 |   k=16 |   k=5 |   k=5 |   k=5 |   k=5 |
| How big is the new Nepali NER dataset? _(05828/f59f1f)_ |   k=5 |   k=15 | ✓ k=27 |   k=5 |   k=13 | ✓ k=121 | ✓ k=43 |   k=15 | ✓ k=171 |   k=5 |   k=5 | ✓ k=52 |   k=5 |
| Do they train their model starting from a checkpoint? _(03405/7b4fb6)_ |   k=5 |   k=31 |   k=3 |   k=15 |   k=5 | ✓ k=244 | ✓ k=46 |   k=4 | ✓ k=81 |   k=5 |   k=5 |   k=5 |   k=5 |
| What BERT model do they test? _(03405/bc31a3)_ |   k=5 | ✓ k=55 | ✓ k=52 | ✓ k=63 | ✓ k=55 | ✓ k=120 | ✓ k=55 | ✓ k=54 | ✓ k=73 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| How much is performance improved on NLI? _(03405/bdc91d)_ | ✓ k=5 | ✓ k=69 | ✓ k=52 | ✓ k=56 | ✓ k=67 | ✓ k=256 | ✓ k=129 | ✓ k=52 | ✓ k=88 | ✓ k=5 | ✓ k=5 | ✓ k=23 |   k=5 |
| How were the datasets annotated? _(04866/71b1af)_ | ✓ k=5 | ✓ k=15 | ✓ k=3 |   k=5 | ✓ k=3 | ✓ k=154 | ✓ k=3 | ✓ k=3 | ✓ k=69 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| What are the 12 languages covered? _(04866/a616a3)_ |   k=5 |   k=11 | ✓ k=79 | ✓ k=43 | ✓ k=30 | ✓ k=178 | ✓ k=168 | ✓ k=34 | ✓ k=191 | ✓ k=5 |   k=5 | ✓ k=64 | ✓ k=5 |
| What is masked document generation? _(01853/193ee4)_ | ✓ k=5 | ✓ k=5 | ✓ k=2 | ✓ k=7 | ✓ k=5 | ✓ k=371 | ✓ k=154 | ✓ k=7 | ✓ k=77 | ✓ k=5 | ✓ k=5 | ✓ k=42 | ✓ k=5 |
| Which of the three pretraining tasks is the most helpful? _(01853/ed2eb4)_ |   k=5 |   k=60 |   k=5 |   k=25 |   k=8 | ✓ k=301 | ✓ k=78 |   k=12 |   k=103 | ✓ k=5 |   k=5 | ✓ k=11 | ✓ k=5 |

---
Generated by `benchmarks/report_md.py`. Raw rows in `raw.jsonl`, aggregated stats in `aggregate.json`.