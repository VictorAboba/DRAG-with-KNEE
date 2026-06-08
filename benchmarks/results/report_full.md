# DRAG-with-KNEE benchmark — full analytical report

- papers: `10`  ·  questions: `35`  ·  retrievers: `13`  ·  median latency (all rows): `35.44s`

## TL;DR

- **Recall@5**: `hybrid_rrf_flat` at 0.387 vs `vanilla_dense` at 0.374 (Δ=0.012)
- **MRR@10**: `hybrid_rrf_flat` at 0.301 vs `raptor_collapsed` at 0.284 (Δ=0.017)
- **nDCG@10**: `hybrid_rrf_flat` at 0.295 vs `raptor_collapsed` at 0.286 (Δ=0.009)
- **Right-sized Recall**: `drag_beam_sensk_0.25` at 0.663 vs `raptor_collapsed` at 0.524 (Δ=0.140)
- **Fastest**: `bm25_only` at 2.556 s/query
- **Tightest selection**: `bm25_only` returns 5.0 chunks on average

## Indexing one-time cost

| Collection | Leaves | Parents | LLM calls | Seconds |
|---|---:|---:|---:|---:|
| `bench_flat` | 530 | 0 | 0 | 1566.6 |
| `bench_drag` | 530 | 268 | 798 | 4618.0 |
| `bench_raptor` | 530 | 32 | 32 | 2687.0 |

## Full retrieval table (mean [95% CI])

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | hit@5 | mrr@10 | ndcg@5 | ndcg@10 | right_sized_recall | avg_k_returned | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `drag_beam_fixed` | 0.086 [0.014, 0.186] | 0.124 [0.038, 0.238] | 0.124 [0.038, 0.238] | 0.152 [0.057, 0.262] | 0.171 [0.057, 0.314] | 0.142 [0.046, 0.265] | 0.119 [0.037, 0.224] | 0.128 [0.044, 0.238] | 0.267 [0.143, 0.419] | 32.800 [20.000, 46.629] | 33.141 [30.999, 35.542] |
| `drag_beam_knee` | 0.029 [0.000, 0.071] | 0.067 [0.010, 0.143] | 0.067 [0.010, 0.143] | 0.124 [0.043, 0.229] | 0.114 [0.029, 0.229] | 0.089 [0.018, 0.181] | 0.062 [0.007, 0.130] | 0.081 [0.027, 0.152] | 0.171 [0.071, 0.300] | 17.943 [11.714, 24.571] | 39.888 [36.480, 43.633] |
| `drag_beam_scheduled` | 0.038 [0.000, 0.114] | 0.076 [0.010, 0.162] | 0.095 [0.010, 0.200] | 0.140 [0.045, 0.257] | 0.114 [0.029, 0.229] | 0.092 [0.024, 0.182] | 0.079 [0.007, 0.168] | 0.097 [0.027, 0.188] | 0.169 [0.062, 0.295] | 21.400 [14.971, 29.429] | 73.204 [65.199, 81.961] |
| `drag_beam_sensitive_knee` | 0.029 [0.000, 0.071] | 0.067 [0.010, 0.143] | 0.067 [0.010, 0.143] | 0.124 [0.043, 0.229] | 0.114 [0.029, 0.229] | 0.089 [0.018, 0.181] | 0.062 [0.007, 0.130] | 0.081 [0.027, 0.152] | 0.171 [0.071, 0.300] | 21.971 [14.029, 32.514] | 40.154 [37.331, 42.711] |
| `drag_beam_sensk_0.25` | 0.063 [0.006, 0.140] | 0.198 [0.081, 0.331] | 0.236 [0.112, 0.368] | 0.331 [0.179, 0.489] | 0.314 [0.171, 0.486] | 0.202 [0.096, 0.322] | 0.179 [0.084, 0.286] | 0.215 [0.115, 0.322] | 0.663 [0.514, 0.805] | 151.429 [122.229, 180.257] | 167.873 [149.838, 188.571] |
| `drag_beam_sensk_0.5` | 0.063 [0.006, 0.140] | 0.198 [0.084, 0.325] | 0.208 [0.095, 0.337] | 0.231 [0.110, 0.368] | 0.286 [0.143, 0.429] | 0.177 [0.076, 0.295] | 0.163 [0.076, 0.261] | 0.174 [0.081, 0.281] | 0.486 [0.340, 0.655] | 67.000 [46.143, 89.657] | 95.618 [86.088, 105.779] |
| `drag_beam_sensk_0.75` | 0.095 [0.024, 0.190] | 0.152 [0.052, 0.267] | 0.186 [0.076, 0.310] | 0.186 [0.076, 0.310] | 0.229 [0.114, 0.371] | 0.179 [0.071, 0.307] | 0.161 [0.066, 0.278] | 0.161 [0.066, 0.278] | 0.279 [0.152, 0.438] | 30.743 [19.229, 44.657] | 89.934 [80.628, 98.799] |
| `drag_branch` | 0.000 [0.000, 0.000] | 0.007 [0.000, 0.021] | 0.007 [0.000, 0.021] | 0.026 [0.000, 0.071] | 0.029 [0.000, 0.086] | 0.018 [0.000, 0.051] | 0.007 [0.000, 0.021] | 0.016 [0.000, 0.040] | 0.321 [0.171, 0.486] | 75.943 [65.657, 87.143] | 110.732 [94.728, 128.962] |
| `raptor_collapsed` | 0.163 [0.063, 0.277] | 0.287 [0.152, 0.438] | 0.330 [0.183, 0.478] | 0.381 [0.229, 0.538] | 0.400 [0.229, 0.571] | 0.284 [0.150, 0.429] | 0.266 [0.148, 0.400] | 0.286 [0.163, 0.421] | 0.524 [0.362, 0.681] | 18.629 [11.114, 27.543] | 10.928 [9.711, 12.090] |
| `hybrid_rrf_flat` | 0.163 [0.063, 0.277] | 0.315 [0.171, 0.477] | 0.387 [0.243, 0.543] | 0.387 [0.243, 0.543] | 0.457 [0.286, 0.629] | 0.301 [0.170, 0.443] | 0.295 [0.175, 0.427] | 0.295 [0.175, 0.427] | 0.387 [0.243, 0.543] | 5.000 [5.000, 5.000] | 9.391 [8.725, 10.064] |
| `vanilla_dense` | 0.120 [0.034, 0.220] | 0.297 [0.157, 0.443] | 0.374 [0.231, 0.520] | 0.374 [0.231, 0.520] | 0.429 [0.286, 0.600] | 0.273 [0.149, 0.405] | 0.273 [0.158, 0.384] | 0.273 [0.158, 0.384] | 0.374 [0.231, 0.520] | 5.000 [5.000, 5.000] | 7.814 [7.598, 8.032] |
| `bm25_only` | 0.086 [0.014, 0.186] | 0.187 [0.072, 0.320] | 0.244 [0.114, 0.389] | 0.244 [0.114, 0.389] | 0.286 [0.143, 0.457] | 0.175 [0.076, 0.291] | 0.179 [0.082, 0.296] | 0.179 [0.082, 0.296] | 0.244 [0.114, 0.389] | 5.000 [5.000, 5.000] | 2.556 [2.508, 2.606] |
| `hyde_hybrid` | 0.034 [0.000, 0.091] | 0.177 [0.071, 0.306] | 0.187 [0.079, 0.314] | 0.187 [0.079, 0.314] | 0.257 [0.114, 0.400] | 0.140 [0.060, 0.233] | 0.134 [0.062, 0.224] | 0.134 [0.062, 0.224] | 0.187 [0.079, 0.314] | 5.000 [5.000, 5.000] | 17.713 [16.375, 19.572] |

## Head-to-head comparisons

### Hierarchy lift (DRAG fixed vs flat hybrid)

| Metric | `drag_beam_fixed` | `hybrid_rrf_flat` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.124 | 0.387 | -0.263 |
| `ndcg@5` | 0.119 | 0.295 | -0.176 |
| `right_sized_recall` | 0.267 | 0.387 | -0.120 |
| `avg_k_returned` | 32.800 | 5.000 | +27.800 |
| `latency_s` | 33.141 | 9.391 | +23.750 |

### Knee adaptive vs fixed-k (k=5)

| Metric | `drag_beam_knee` | `drag_beam_fixed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.067 | 0.124 | -0.057 |
| `ndcg@5` | 0.062 | 0.119 | -0.057 |
| `right_sized_recall` | 0.171 | 0.267 | -0.095 |
| `avg_k_returned` | 17.943 | 32.800 | -14.857 |
| `latency_s` | 39.888 | 33.141 | +6.746 |

### Sensitive-knee (0.85) vs plain knee

| Metric | `drag_beam_sensitive_knee` | `drag_beam_knee` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.067 | 0.067 | +0.000 |
| `ndcg@5` | 0.062 | 0.062 | +0.000 |
| `right_sized_recall` | 0.171 | 0.171 | +0.000 |
| `avg_k_returned` | 21.971 | 17.943 | +4.029 |
| `latency_s` | 40.154 | 39.888 | +0.266 |

### DRAG knee vs RAPTOR (both hierarchical, different construction)

| Metric | `drag_beam_knee` | `raptor_collapsed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.067 | 0.330 | -0.263 |
| `ndcg@5` | 0.062 | 0.266 | -0.204 |
| `right_sized_recall` | 0.171 | 0.524 | -0.352 |
| `avg_k_returned` | 17.943 | 18.629 | -0.686 |
| `latency_s` | 39.888 | 10.928 | +28.960 |

### Dense + sparse fusion vs pure dense

| Metric | `hybrid_rrf_flat` | `vanilla_dense` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.387 | 0.374 | +0.012 |
| `ndcg@5` | 0.295 | 0.273 | +0.022 |
| `right_sized_recall` | 0.387 | 0.374 | +0.012 |
| `avg_k_returned` | 5.000 | 5.000 | +0.000 |
| `latency_s` | 9.391 | 7.814 | +1.577 |

## Efficiency trade-offs

| Retriever | nDCG@10 | latency (s) | avg k returned | nDCG per second |
|---|---:|---:|---:|---:|
| `hybrid_rrf_flat` | 0.295 | 9.391 | 5.0 | 0.031 |
| `raptor_collapsed` | 0.286 | 10.928 | 18.6 | 0.026 |
| `vanilla_dense` | 0.273 | 7.814 | 5.0 | 0.035 |
| `drag_beam_sensk_0.25` | 0.215 | 167.873 | 151.4 | 0.001 |
| `bm25_only` | 0.179 | 2.556 | 5.0 | 0.070 |
| `drag_beam_sensk_0.5` | 0.174 | 95.618 | 67.0 | 0.002 |
| `drag_beam_sensk_0.75` | 0.161 | 89.934 | 30.7 | 0.002 |
| `hyde_hybrid` | 0.134 | 17.713 | 5.0 | 0.008 |
| `drag_beam_fixed` | 0.128 | 33.141 | 32.8 | 0.004 |
| `drag_beam_scheduled` | 0.097 | 73.204 | 21.4 | 0.001 |
| `drag_beam_knee` | 0.081 | 39.888 | 17.9 | 0.002 |
| `drag_beam_sensitive_knee` | 0.081 | 40.154 | 22.0 | 0.002 |
| `drag_branch` | 0.016 | 110.732 | 75.9 | 0.000 |

## Per-question coverage

For each question, **✓** means at least one gold paragraph appeared anywhere in the returned set (right-sized hit). Lower-case **k=N** is the number of chunks the retriever returned.

| Q (paper, id) | bm25_only | drag_beam_fixed | drag_beam_knee | drag_beam_scheduled | drag_beam_sensitive_knee | drag_beam_sensk_0.25 | drag_beam_sensk_0.5 | drag_beam_sensk_0.75 | drag_branch | hybrid_rrf_flat | hyde_hybrid | raptor_collapsed | vanilla_dense |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| How do they detect spammers? _(08504/3cd185)_ |   k=5 |   k=9 |   k=9 | ✓ k=11 |   k=9 | ✓ k=51 | ✓ k=17 | ✓ k=13 | ✓ k=20 |   k=5 |   k=5 | ✓ k=32 |   k=5 |
| What is the benchmark dataset and is its quality high? _(08504/a1645d)_ |   k=5 |   k=82 |   k=7 |   k=32 |   k=7 |   k=264 |   k=204 |   k=175 |   k=46 |   k=5 |   k=5 |   k=5 |   k=5 |
| LDA is an unsupervised method; is this paper introducing … _(08504/dac087)_ |   k=5 |   k=11 |   k=5 |   k=14 |   k=14 |   k=161 |   k=24 |   k=17 |   k=70 |   k=5 |   k=5 | ✓ k=32 |   k=5 |
| What parallel corpus did they use? _(08198/602403)_ | ✓ k=5 |   k=72 |   k=69 |   k=31 |   k=38 |   k=152 |   k=6 |   k=4 |   k=48 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| Is pre-training effective in their evaluation? _(08198/8ce115)_ | ✓ k=5 | ✓ k=58 | ✓ k=43 |   k=25 | ✓ k=55 | ✓ k=197 | ✓ k=141 | ✓ k=58 |   k=67 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| How does their decoder generate text? _(07814/0a75a5)_ |   k=5 | ✓ k=13 |   k=1 | ✓ k=5 |   k=2 | ✓ k=101 | ✓ k=8 | ✓ k=4 |   k=126 |   k=5 |   k=5 |   k=44 |   k=5 |
| Which architecture do they use for the encoder and decoder? _(07814/1b23c4)_ | ✓ k=5 | ✓ k=15 | ✓ k=14 | ✓ k=17 | ✓ k=17 | ✓ k=153 | ✓ k=55 | ✓ k=9 | ✓ k=59 | ✓ k=5 | ✓ k=5 | ✓ k=22 |   k=5 |
| Which dataset do they use? _(07814/fd0a3e)_ |   k=5 |   k=37 |   k=33 |   k=9 |   k=33 |   k=127 |   k=27 |   k=38 |   k=19 |   k=5 |   k=5 |   k=5 |   k=5 |
| what crowdsourcing platform was used? _(05223/154a72)_ | ✓ k=5 | ✓ k=15 | ✓ k=27 | ✓ k=31 | ✓ k=27 | ✓ k=118 | ✓ k=60 | ✓ k=21 | ✓ k=61 | ✓ k=5 |   k=5 | ✓ k=62 | ✓ k=5 |
| what is the size of their dataset? _(05223/2eb928)_ |   k=5 | ✓ k=13 | ✓ k=10 |   k=7 | ✓ k=12 |   k=10 |   k=5 |   k=3 | ✓ k=70 |   k=5 |   k=5 |   k=5 |   k=5 |
| how was the data collected? _(05223/84bad9)_ | ✓ k=5 |   k=5 |   k=5 |   k=5 |   k=7 | ✓ k=196 |   k=12 |   k=9 |   k=86 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| what dataset statistics are provided? _(05223/ad1be6)_ |   k=5 |   k=159 |   k=77 | ✓ k=9 |   k=163 | ✓ k=297 |   k=204 |   k=161 | ✓ k=89 |   k=5 |   k=5 | ✓ k=63 |   k=5 |
| what language does this paper focus on? _(04428/2c6b50)_ |   k=5 |   k=7 |   k=4 |   k=31 |   k=5 |   k=323 |   k=166 |   k=72 |   k=109 |   k=5 |   k=5 |   k=5 |   k=5 |
| what state of the art methods did they compare with? _(04428/326588)_ |   k=5 |   k=142 |   k=30 |   k=114 |   k=35 | ✓ k=293 | ✓ k=216 |   k=44 |   k=79 |   k=5 |   k=5 |   k=5 |   k=5 |
| by how much did their model improve? _(04428/4625cf)_ |   k=5 |   k=5 |   k=31 |   k=31 |   k=66 | ✓ k=169 |   k=93 |   k=66 |   k=81 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| what are the sizes of both datasets? _(04428/ebf0d9)_ |   k=5 |   k=13 |   k=10 |   k=41 |   k=3 | ✓ k=49 | ✓ k=10 |   k=5 |   k=65 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| what evaluation metrics did they use? _(04428/f651cd)_ | ✓ k=5 | ✓ k=143 | ✓ k=5 | ✓ k=9 | ✓ k=15 | ✓ k=228 | ✓ k=198 | ✓ k=17 | ✓ k=171 | ✓ k=5 | ✓ k=5 | ✓ k=21 | ✓ k=5 |
| what was the baseline? _(03060/761de1)_ |   k=5 |   k=14 |   k=7 |   k=14 |   k=9 | ✓ k=66 |   k=10 |   k=3 |   k=97 |   k=5 |   k=5 |   k=5 |   k=5 |
| How many different types of entities exist in the dataset? _(05828/1462eb)_ |   k=5 | ✓ k=5 | ✓ k=1 |   k=7 | ✓ k=5 | ✓ k=148 | ✓ k=8 | ✓ k=9 | ✓ k=79 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| What is the best model? _(05828/567dc9)_ |   k=5 |   k=13 |   k=14 |   k=5 |   k=20 |   k=106 |   k=24 |   k=17 |   k=65 |   k=5 |   k=5 |   k=5 |   k=5 |
| Which models are used to solve NER for Nepali? _(05828/6d1217)_ |   k=5 | ✓ k=49 | ✓ k=18 | ✓ k=53 | ✓ k=18 | ✓ k=236 | ✓ k=52 | ✓ k=21 |   k=100 |   k=5 |   k=5 |   k=23 | ✓ k=5 |
| Which machine learning models do they explore? _(05828/8a7615)_ |   k=5 |   k=98 |   k=68 |   k=7 |   k=68 | ✓ k=211 |   k=159 |   k=110 |   k=62 |   k=5 |   k=5 |   k=5 |   k=5 |
| What is the performance improvement of the grapheme-level… _(05828/9bd080)_ | ✓ k=5 |   k=13 |   k=3 |   k=7 |   k=4 | ✓ k=170 | ✓ k=57 |   k=16 | ✓ k=97 | ✓ k=5 |   k=5 | ✓ k=34 | ✓ k=5 |
| What is the size of the dataset? _(05828/a1b3e2)_ |   k=5 |   k=13 |   k=10 |   k=7 |   k=11 |   k=19 |   k=6 |   k=10 | ✓ k=71 |   k=5 | ✓ k=5 |   k=5 |   k=5 |
| What is the source of their dataset? _(05828/bb2de2)_ |   k=5 |   k=17 |   k=7 |   k=7 |   k=8 | ✓ k=119 | ✓ k=34 |   k=8 |   k=36 |   k=5 |   k=5 |   k=5 |   k=5 |
| What is the baseline? _(05828/cb77d6)_ |   k=5 |   k=12 |   k=7 |   k=12 |   k=8 |   k=51 |   k=8 |   k=3 | ✓ k=86 |   k=5 |   k=5 |   k=5 |   k=5 |
| How many sentences does the dataset contain? _(05828/d51dc3)_ |   k=5 |   k=7 |   k=19 |   k=9 |   k=19 |   k=122 |   k=47 |   k=22 |   k=59 |   k=5 |   k=5 |   k=5 |   k=5 |
| How big is the new Nepali NER dataset? _(05828/f59f1f)_ |   k=5 |   k=9 |   k=3 |   k=35 |   k=34 | ✓ k=110 | ✓ k=78 |   k=43 |   k=20 |   k=5 |   k=5 | ✓ k=52 |   k=5 |
| Do they train their model starting from a checkpoint? _(03405/7b4fb6)_ |   k=5 |   k=13 |   k=6 |   k=9 |   k=10 | ✓ k=58 | ✓ k=20 |   k=16 | ✓ k=76 |   k=5 |   k=5 |   k=5 |   k=5 |
| What BERT model do they test? _(03405/bc31a3)_ |   k=5 | ✓ k=41 |   k=12 | ✓ k=15 |   k=15 | ✓ k=61 | ✓ k=53 | ✓ k=42 | ✓ k=105 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| How much is performance improved on NLI? _(03405/bdc91d)_ | ✓ k=5 | ✓ k=9 |   k=5 |   k=55 |   k=5 | ✓ k=224 | ✓ k=63 | ✓ k=5 |   k=56 | ✓ k=5 | ✓ k=5 | ✓ k=5 |   k=5 |
| How were the datasets annotated? _(04866/71b1af)_ | ✓ k=5 |   k=17 |   k=10 |   k=9 |   k=11 | ✓ k=43 | ✓ k=23 |   k=13 |   k=49 | ✓ k=5 | ✓ k=5 | ✓ k=20 | ✓ k=5 |
| What are the 12 languages covered? _(04866/a616a3)_ |   k=5 |   k=5 |   k=52 |   k=9 |   k=8 | ✓ k=327 |   k=154 |   k=8 |   k=142 | ✓ k=5 | ✓ k=5 | ✓ k=126 | ✓ k=5 |
| What is masked document generation? _(01853/193ee4)_ | ✓ k=5 |   k=5 |   k=2 |   k=5 |   k=4 | ✓ k=55 | ✓ k=20 | ✓ k=5 |   k=110 | ✓ k=5 | ✓ k=5 | ✓ k=11 | ✓ k=5 |
| Which of the three pretraining tasks is the most helpful? _(01853/ed2eb4)_ |   k=5 | ✓ k=9 |   k=4 |   k=62 |   k=4 | ✓ k=285 | ✓ k=83 | ✓ k=9 |   k=82 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |

---
Generated by `benchmarks/report_md.py`. Raw rows in `raw.jsonl`, aggregated stats in `aggregate.json`.