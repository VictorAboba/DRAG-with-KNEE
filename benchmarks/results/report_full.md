# DRAG-with-KNEE benchmark — full analytical report

- papers: `10`  ·  questions: `35`  ·  retrievers: `17`  ·  median latency (all rows): `26.65s`

## TL;DR

- **Recall@5**: `hybrid_rrf_flat` at 0.387 vs `raptor_collapsed` at 0.377 (Δ=0.010)
- **MRR@10**: `hybrid_rrf_flat` at 0.301 vs `raptor_collapsed` at 0.299 (Δ=0.002)
- **nDCG@10**: `hybrid_rrf_flat` at 0.295 vs `raptor_collapsed` at 0.294 (Δ=0.001)
- **Right-sized Recall**: `drag_beam_sensk_0.25` at 0.729 vs `drag_beam_sensk_0.5` at 0.523 (Δ=0.206)
- **Fastest**: `bm25_only` at 2.455 s/query
- **Tightest selection**: `bm25_only` returns 5.0 chunks on average

## Indexing one-time cost

| Collection | Leaves | Parents | LLM calls | Seconds |
|---|---:|---:|---:|---:|
| `bench_flat` | 530 | 0 | 0 | 278.9 |
| `bench_drag` | 530 | 268 | 798 | 5643.3 |
| `bench_raptor` | 530 | 32 | 562 | 7680.4 |

## Full retrieval table (mean [95% CI])

| Retriever | recall@1 | recall@3 | recall@5 | recall@10 | hit@5 | mrr@10 | ndcg@5 | ndcg@10 | right_sized_recall | avg_k_returned | latency_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `drag_beam_fixed` | 0.034 [0.000, 0.097] | 0.055 [0.000, 0.130] | 0.084 [0.010, 0.181] | 0.084 [0.010, 0.181] | 0.114 [0.029, 0.229] | 0.074 [0.007, 0.169] | 0.068 [0.007, 0.153] | 0.068 [0.007, 0.153] | 0.360 [0.217, 0.520] | 72.000 [53.714, 92.086] | 30.807 [28.680, 32.799] |
| `drag_beam_knee` | 0.006 [0.000, 0.017] | 0.017 [0.000, 0.051] | 0.027 [0.000, 0.070] | 0.027 [0.000, 0.070] | 0.057 [0.000, 0.143] | 0.036 [0.000, 0.100] | 0.026 [0.000, 0.073] | 0.026 [0.000, 0.073] | 0.255 [0.124, 0.410] | 46.943 [34.143, 59.857] | 26.954 [24.598, 29.477] |
| `drag_beam_scheduled` | 0.034 [0.000, 0.097] | 0.084 [0.010, 0.181] | 0.141 [0.048, 0.261] | 0.141 [0.048, 0.261] | 0.171 [0.057, 0.314] | 0.094 [0.022, 0.195] | 0.097 [0.030, 0.190] | 0.097 [0.030, 0.190] | 0.341 [0.198, 0.499] | 65.571 [48.257, 83.743] | 32.677 [30.947, 34.439] |
| `drag_beam_sensitive_knee` | 0.020 [0.000, 0.057] | 0.031 [0.000, 0.083] | 0.041 [0.000, 0.092] | 0.041 [0.000, 0.092] | 0.086 [0.000, 0.200] | 0.064 [0.000, 0.150] | 0.044 [0.000, 0.105] | 0.044 [0.000, 0.105] | 0.270 [0.133, 0.420] | 53.657 [38.543, 68.714] | 36.361 [32.752, 40.207] |
| `drag_beam_sensk_0.25` | 0.077 [0.006, 0.177] | 0.155 [0.055, 0.270] | 0.212 [0.098, 0.347] | 0.212 [0.098, 0.347] | 0.257 [0.143, 0.400] | 0.159 [0.060, 0.277] | 0.156 [0.069, 0.264] | 0.156 [0.069, 0.264] | 0.729 [0.571, 0.871] | 242.257 [212.514, 269.343] | 73.824 [64.734, 84.059] |
| `drag_beam_sensk_0.5` | 0.063 [0.000, 0.160] | 0.141 [0.046, 0.255] | 0.170 [0.063, 0.291] | 0.170 [0.063, 0.291] | 0.200 [0.086, 0.343] | 0.125 [0.044, 0.243] | 0.128 [0.046, 0.240] | 0.128 [0.046, 0.240] | 0.523 [0.366, 0.689] | 118.857 [90.171, 147.486] | 57.361 [51.515, 63.623] |
| `drag_beam_sensk_0.75` | 0.006 [0.000, 0.017] | 0.084 [0.017, 0.179] | 0.112 [0.029, 0.223] | 0.112 [0.029, 0.223] | 0.143 [0.057, 0.257] | 0.068 [0.015, 0.148] | 0.071 [0.018, 0.140] | 0.071 [0.018, 0.140] | 0.379 [0.223, 0.543] | 65.943 [45.314, 89.286] | 41.648 [37.114, 46.353] |
| `drag_branch` | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.324 [0.181, 0.486] | 102.229 [87.914, 116.943] | 57.328 [49.353, 67.031] |
| `drag_subtree` | 0.071 [0.000, 0.157] | 0.100 [0.014, 0.200] | 0.100 [0.014, 0.200] | 0.100 [0.014, 0.200] | 0.114 [0.029, 0.229] | 0.095 [0.010, 0.200] | 0.089 [0.014, 0.178] | 0.089 [0.014, 0.178] | 0.410 [0.257, 0.567] | 88.257 [71.457, 105.743] | 30.975 [30.840, 31.120] |
| `drag_subtree_1p` | 0.100 [0.014, 0.214] | 0.152 [0.057, 0.267] | 0.152 [0.057, 0.267] | 0.152 [0.057, 0.267] | 0.200 [0.086, 0.343] | 0.143 [0.048, 0.257] | 0.133 [0.044, 0.241] | 0.133 [0.044, 0.241] | 0.324 [0.181, 0.471] | 16.000 [10.914, 21.629] | 15.785 [14.980, 16.678] |
| `drag_subtree_drill` | 0.071 [0.000, 0.157] | 0.100 [0.014, 0.200] | 0.107 [0.029, 0.207] | 0.117 [0.036, 0.212] | 0.143 [0.029, 0.257] | 0.109 [0.029, 0.206] | 0.097 [0.022, 0.189] | 0.101 [0.026, 0.193] | 0.245 [0.124, 0.374] | 50.229 [34.657, 68.514] | 31.704 [31.468, 31.931] |
| `drag_subtree_tight` | 0.110 [0.029, 0.205] | 0.124 [0.038, 0.229] | 0.124 [0.038, 0.229] | 0.124 [0.038, 0.229] | 0.171 [0.057, 0.286] | 0.152 [0.057, 0.267] | 0.125 [0.042, 0.226] | 0.125 [0.042, 0.226] | 0.238 [0.114, 0.371] | 8.714 [4.714, 13.543] | 15.161 [14.837, 15.584] |
| `raptor_collapsed` | 0.163 [0.063, 0.277] | 0.334 [0.186, 0.491] | 0.377 [0.229, 0.534] | 0.377 [0.229, 0.534] | 0.429 [0.257, 0.600] | 0.299 [0.167, 0.438] | 0.294 [0.168, 0.428] | 0.294 [0.168, 0.428] | 0.422 [0.277, 0.576] | 15.400 [10.200, 21.400] | 4.780 [4.720, 4.845] |
| `hybrid_rrf_flat` | 0.163 [0.063, 0.277] | 0.315 [0.171, 0.477] | 0.387 [0.243, 0.543] | 0.387 [0.243, 0.543] | 0.457 [0.286, 0.629] | 0.301 [0.170, 0.443] | 0.295 [0.175, 0.427] | 0.295 [0.175, 0.427] | 0.387 [0.243, 0.543] | 5.000 [5.000, 5.000] | 5.726 [5.676, 5.780] |
| `vanilla_dense` | 0.120 [0.034, 0.220] | 0.297 [0.157, 0.443] | 0.374 [0.231, 0.520] | 0.374 [0.231, 0.520] | 0.429 [0.286, 0.600] | 0.273 [0.149, 0.405] | 0.273 [0.158, 0.384] | 0.273 [0.158, 0.384] | 0.374 [0.231, 0.520] | 5.000 [5.000, 5.000] | 5.569 [5.520, 5.622] |
| `bm25_only` | 0.086 [0.014, 0.186] | 0.187 [0.072, 0.320] | 0.244 [0.114, 0.389] | 0.244 [0.114, 0.389] | 0.286 [0.143, 0.457] | 0.175 [0.076, 0.291] | 0.179 [0.082, 0.296] | 0.179 [0.082, 0.296] | 0.244 [0.114, 0.389] | 5.000 [5.000, 5.000] | 2.455 [2.415, 2.501] |
| `hyde_hybrid` | 0.086 [0.014, 0.186] | 0.187 [0.086, 0.306] | 0.250 [0.129, 0.386] | 0.250 [0.129, 0.386] | 0.343 [0.200, 0.514] | 0.192 [0.093, 0.314] | 0.184 [0.093, 0.288] | 0.184 [0.093, 0.288] | 0.250 [0.129, 0.386] | 5.000 [5.000, 5.000] | 9.644 [9.225, 10.200] |

## Head-to-head comparisons

### Hierarchy lift (DRAG fixed vs flat hybrid)

| Metric | `drag_beam_fixed` | `hybrid_rrf_flat` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.084 | 0.387 | -0.303 |
| `ndcg@5` | 0.068 | 0.295 | -0.227 |
| `right_sized_recall` | 0.360 | 0.387 | -0.027 |
| `avg_k_returned` | 72.000 | 5.000 | +67.000 |
| `latency_s` | 30.807 | 5.726 | +25.081 |

### Knee adaptive vs fixed-k (k=5)

| Metric | `drag_beam_knee` | `drag_beam_fixed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.027 | 0.084 | -0.057 |
| `ndcg@5` | 0.026 | 0.068 | -0.042 |
| `right_sized_recall` | 0.255 | 0.360 | -0.105 |
| `avg_k_returned` | 46.943 | 72.000 | -25.057 |
| `latency_s` | 26.954 | 30.807 | -3.853 |

### Sensitive-knee (0.85) vs plain knee

| Metric | `drag_beam_sensitive_knee` | `drag_beam_knee` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.041 | 0.027 | +0.014 |
| `ndcg@5` | 0.044 | 0.026 | +0.018 |
| `right_sized_recall` | 0.270 | 0.255 | +0.014 |
| `avg_k_returned` | 53.657 | 46.943 | +6.714 |
| `latency_s` | 36.361 | 26.954 | +9.407 |

### DRAG knee vs RAPTOR (both hierarchical, different construction)

| Metric | `drag_beam_knee` | `raptor_collapsed` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.027 | 0.377 | -0.350 |
| `ndcg@5` | 0.026 | 0.294 | -0.267 |
| `right_sized_recall` | 0.255 | 0.422 | -0.167 |
| `avg_k_returned` | 46.943 | 15.400 | +31.543 |
| `latency_s` | 26.954 | 4.780 | +22.174 |

### Dense + sparse fusion vs pure dense

| Metric | `hybrid_rrf_flat` | `vanilla_dense` | Δ (a − b) |
|---|---|---|---|
| `recall@5` | 0.387 | 0.374 | +0.012 |
| `ndcg@5` | 0.295 | 0.273 | +0.022 |
| `right_sized_recall` | 0.387 | 0.374 | +0.012 |
| `avg_k_returned` | 5.000 | 5.000 | +0.000 |
| `latency_s` | 5.726 | 5.569 | +0.157 |

## Efficiency trade-offs

| Retriever | nDCG@10 | latency (s) | avg k returned | nDCG per second |
|---|---:|---:|---:|---:|
| `hybrid_rrf_flat` | 0.295 | 5.726 | 5.0 | 0.051 |
| `raptor_collapsed` | 0.294 | 4.780 | 15.4 | 0.061 |
| `vanilla_dense` | 0.273 | 5.569 | 5.0 | 0.049 |
| `hyde_hybrid` | 0.184 | 9.644 | 5.0 | 0.019 |
| `bm25_only` | 0.179 | 2.455 | 5.0 | 0.073 |
| `drag_beam_sensk_0.25` | 0.156 | 73.824 | 242.3 | 0.002 |
| `drag_subtree_1p` | 0.133 | 15.785 | 16.0 | 0.008 |
| `drag_beam_sensk_0.5` | 0.128 | 57.361 | 118.9 | 0.002 |
| `drag_subtree_tight` | 0.125 | 15.161 | 8.7 | 0.008 |
| `drag_subtree_drill` | 0.101 | 31.704 | 50.2 | 0.003 |
| `drag_beam_scheduled` | 0.097 | 32.677 | 65.6 | 0.003 |
| `drag_subtree` | 0.089 | 30.975 | 88.3 | 0.003 |
| `drag_beam_sensk_0.75` | 0.071 | 41.648 | 65.9 | 0.002 |
| `drag_beam_fixed` | 0.068 | 30.807 | 72.0 | 0.002 |
| `drag_beam_sensitive_knee` | 0.044 | 36.361 | 53.7 | 0.001 |
| `drag_beam_knee` | 0.026 | 26.954 | 46.9 | 0.001 |
| `drag_branch` | 0.000 | 57.328 | 102.2 | 0.000 |

## Per-question coverage

For each question, **✓** means at least one gold paragraph appeared anywhere in the returned set (right-sized hit). Lower-case **k=N** is the number of chunks the retriever returned.

| Q (paper, id) | bm25_only | drag_beam_fixed | drag_beam_knee | drag_beam_scheduled | drag_beam_sensitive_knee | drag_beam_sensk_0.25 | drag_beam_sensk_0.5 | drag_beam_sensk_0.75 | drag_branch | drag_subtree | drag_subtree_1p | drag_subtree_drill | drag_subtree_tight | hybrid_rrf_flat | hyde_hybrid | raptor_collapsed | vanilla_dense |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| How do they detect spammers? _(08504/3cd185)_ |   k=5 | ✓ k=68 | ✓ k=35 | ✓ k=68 | ✓ k=59 | ✓ k=116 | ✓ k=79 | ✓ k=38 | ✓ k=96 | ✓ k=210 | ✓ k=32 | ✓ k=210 | ✓ k=32 |   k=5 |   k=5 | ✓ k=23 |   k=5 |
| What is the benchmark dataset and is its quality high? _(08504/a1645d)_ |   k=5 |   k=50 |   k=33 |   k=175 |   k=34 | ✓ k=357 |   k=7 |   k=34 |   k=211 |   k=147 |   k=15 |   k=18 |   k=3 |   k=5 |   k=5 |   k=5 |   k=5 |
| LDA is an unsupervised method; is this paper introducing … _(08504/dac087)_ |   k=5 | ✓ k=42 | ✓ k=58 | ✓ k=65 | ✓ k=110 | ✓ k=301 | ✓ k=124 | ✓ k=137 | ✓ k=110 | ✓ k=87 | ✓ k=32 | ✓ k=87 | ✓ k=32 |   k=5 |   k=5 |   k=22 |   k=5 |
| What parallel corpus did they use? _(08198/602403)_ | ✓ k=5 |   k=93 |   k=92 |   k=13 |   k=14 | ✓ k=265 |   k=110 |   k=14 | ✓ k=116 | ✓ k=103 | ✓ k=21 | ✓ k=31 | ✓ k=21 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| Is pre-training effective in their evaluation? _(08198/8ce115)_ | ✓ k=5 |   k=114 |   k=51 |   k=57 |   k=83 |   k=249 |   k=191 |   k=83 |   k=77 | ✓ k=79 | ✓ k=12 | ✓ k=40 | ✓ k=12 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| How does their decoder generate text? _(07814/0a75a5)_ |   k=5 |   k=43 |   k=12 |   k=13 |   k=44 | ✓ k=239 | ✓ k=51 | ✓ k=48 | ✓ k=32 |   k=84 |   k=9 |   k=29 |   k=1 |   k=5 |   k=5 |   k=29 |   k=5 |
| Which architecture do they use for the encoder and decoder? _(07814/1b23c4)_ | ✓ k=5 | ✓ k=17 | ✓ k=35 | ✓ k=17 | ✓ k=14 | ✓ k=272 | ✓ k=33 | ✓ k=31 | ✓ k=59 | ✓ k=24 | ✓ k=6 | ✓ k=19 | ✓ k=1 | ✓ k=5 | ✓ k=5 | ✓ k=51 |   k=5 |
| Which dataset do they use? _(07814/fd0a3e)_ |   k=5 |   k=123 |   k=42 |   k=66 |   k=57 | ✓ k=330 |   k=260 |   k=66 |   k=135 |   k=62 |   k=52 |   k=62 |   k=52 |   k=5 |   k=5 |   k=5 |   k=5 |
| what crowdsourcing platform was used? _(05223/154a72)_ | ✓ k=5 |   k=93 |   k=32 | ✓ k=25 |   k=50 | ✓ k=180 |   k=18 |   k=4 |   k=57 |   k=176 |   k=18 |   k=36 |   k=3 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| what is the size of their dataset? _(05223/2eb928)_ |   k=5 | ✓ k=53 |   k=21 | ✓ k=63 | ✓ k=43 | ✓ k=153 | ✓ k=122 | ✓ k=90 | ✓ k=57 |   k=34 |   k=18 |   k=17 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| how was the data collected? _(05223/84bad9)_ | ✓ k=5 |   k=47 |   k=9 |   k=13 |   k=59 | ✓ k=201 | ✓ k=36 |   k=6 |   k=123 |   k=86 |   k=2 |   k=13 |   k=1 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| what dataset statistics are provided? _(05223/ad1be6)_ |   k=5 | ✓ k=144 | ✓ k=99 | ✓ k=123 | ✓ k=114 | ✓ k=293 | ✓ k=211 | ✓ k=114 | ✓ k=114 |   k=75 |   k=3 |   k=57 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| what language does this paper focus on? _(04428/2c6b50)_ |   k=5 |   k=63 |   k=27 |   k=31 |   k=4 |   k=250 |   k=177 |   k=34 |   k=106 |   k=60 |   k=6 |   k=60 |   k=6 |   k=5 |   k=5 |   k=5 |   k=5 |
| what state of the art methods did they compare with? _(04428/326588)_ |   k=5 |   k=105 |   k=90 |   k=95 |   k=99 | ✓ k=309 | ✓ k=130 |   k=114 |   k=114 |   k=56 |   k=1 |   k=56 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| by how much did their model improve? _(04428/4625cf)_ |   k=5 |   k=188 |   k=108 |   k=125 | ✓ k=109 | ✓ k=348 | ✓ k=321 | ✓ k=321 |   k=55 |   k=217 |   k=25 |   k=193 |   k=1 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| what are the sizes of both datasets? _(04428/ebf0d9)_ |   k=5 |   k=66 |   k=39 |   k=35 |   k=50 |   k=159 |   k=131 |   k=87 |   k=66 | ✓ k=50 | ✓ k=17 |   k=17 |   k=1 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| what evaluation metrics did they use? _(04428/f651cd)_ | ✓ k=5 |   k=220 |   k=159 |   k=143 |   k=220 |   k=183 |   k=180 |   k=221 |   k=205 | ✓ k=148 | ✓ k=1 | ✓ k=23 | ✓ k=1 | ✓ k=5 | ✓ k=5 | ✓ k=21 | ✓ k=5 |
| what was the baseline? _(03060/761de1)_ |   k=5 |   k=56 |   k=2 |   k=156 |   k=3 | ✓ k=317 |   k=114 |   k=3 |   k=120 | ✓ k=119 |   k=1 | ✓ k=5 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| How many different types of entities exist in the dataset? _(05828/1462eb)_ |   k=5 | ✓ k=132 | ✓ k=27 |   k=15 |   k=13 | ✓ k=99 | ✓ k=61 |   k=6 | ✓ k=102 | ✓ k=33 |   k=3 |   k=7 |   k=1 | ✓ k=5 | ✓ k=5 | ✓ k=22 | ✓ k=5 |
| What is the best model? _(05828/567dc9)_ |   k=5 |   k=5 |   k=69 |   k=209 |   k=69 |   k=297 |   k=12 |   k=6 |   k=95 |   k=69 |   k=66 |   k=4 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| Which models are used to solve NER for Nepali? _(05828/6d1217)_ |   k=5 | ✓ k=11 |   k=4 | ✓ k=7 |   k=10 | ✓ k=184 | ✓ k=21 | ✓ k=13 |   k=77 |   k=24 |   k=3 |   k=5 |   k=1 |   k=5 |   k=5 | ✓ k=52 | ✓ k=5 |
| Which machine learning models do they explore? _(05828/8a7615)_ |   k=5 |   k=105 |   k=118 |   k=80 |   k=118 |   k=274 |   k=224 |   k=118 |   k=127 |   k=101 |   k=9 |   k=76 |   k=9 |   k=5 |   k=5 |   k=36 |   k=5 |
| What is the performance improvement of the grapheme-level… _(05828/9bd080)_ | ✓ k=5 | ✓ k=174 | ✓ k=52 | ✓ k=151 | ✓ k=52 | ✓ k=285 | ✓ k=179 | ✓ k=121 | ✓ k=125 | ✓ k=114 | ✓ k=3 | ✓ k=49 | ✓ k=3 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| What is the size of the dataset? _(05828/a1b3e2)_ |   k=5 |   k=53 |   k=21 |   k=63 |   k=25 |   k=236 |   k=122 |   k=107 |   k=57 | ✓ k=34 |   k=15 |   k=17 |   k=15 |   k=5 | ✓ k=5 |   k=5 |   k=5 |
| What is the source of their dataset? _(05828/bb2de2)_ |   k=5 |   k=80 |   k=75 |   k=41 |   k=84 |   k=336 |   k=56 |   k=43 |   k=144 |   k=72 |   k=6 |   k=52 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| What is the baseline? _(05828/cb77d6)_ |   k=5 |   k=5 |   k=2 |   k=156 |   k=5 | ✓ k=317 |   k=119 |   k=101 |   k=120 |   k=5 |   k=1 |   k=5 |   k=1 |   k=5 |   k=5 |   k=5 |   k=5 |
| How many sentences does the dataset contain? _(05828/d51dc3)_ |   k=5 |   k=45 |   k=42 |   k=104 |   k=44 |   k=237 |   k=145 |   k=53 |   k=106 |   k=68 |   k=15 |   k=19 |   k=15 |   k=5 |   k=5 |   k=5 |   k=5 |
| How big is the new Nepali NER dataset? _(05828/f59f1f)_ |   k=5 | ✓ k=33 | ✓ k=31 | ✓ k=15 | ✓ k=28 | ✓ k=154 | ✓ k=37 | ✓ k=34 | ✓ k=100 | ✓ k=69 | ✓ k=27 |   k=8 |   k=1 |   k=5 |   k=5 |   k=33 |   k=5 |
| Do they train their model starting from a checkpoint? _(03405/7b4fb6)_ |   k=5 | ✓ k=194 | ✓ k=146 |   k=23 | ✓ k=146 | ✓ k=321 | ✓ k=320 | ✓ k=146 |   k=33 | ✓ k=170 | ✓ k=52 | ✓ k=170 | ✓ k=52 |   k=5 | ✓ k=5 |   k=5 |   k=5 |
| What BERT model do they test? _(03405/bc31a3)_ |   k=5 | ✓ k=23 | ✓ k=52 | ✓ k=21 | ✓ k=52 | ✓ k=111 | ✓ k=36 |   k=21 | ✓ k=118 | ✓ k=115 | ✓ k=52 |   k=48 |   k=2 | ✓ k=5 |   k=5 | ✓ k=5 | ✓ k=5 |
| How much is performance improved on NLI? _(03405/bdc91d)_ | ✓ k=5 |   k=5 |   k=3 | ✓ k=9 |   k=7 | ✓ k=419 | ✓ k=269 | ✓ k=7 |   k=56 | ✓ k=157 | ✓ k=6 | ✓ k=128 | ✓ k=1 | ✓ k=5 | ✓ k=5 | ✓ k=56 |   k=5 |
| How were the datasets annotated? _(04866/71b1af)_ | ✓ k=5 | ✓ k=21 | ✓ k=12 | ✓ k=39 | ✓ k=3 | ✓ k=107 | ✓ k=69 | ✓ k=31 | ✓ k=215 |   k=79 |   k=9 |   k=63 |   k=9 | ✓ k=5 | ✓ k=5 | ✓ k=5 | ✓ k=5 |
| What are the 12 languages covered? _(04866/a616a3)_ |   k=5 |   k=9 |   k=13 |   k=23 |   k=5 | ✓ k=189 |   k=16 |   k=8 |   k=74 |   k=56 |   k=3 |   k=56 |   k=3 | ✓ k=5 |   k=5 | ✓ k=63 | ✓ k=5 |
| What is masked document generation? _(01853/193ee4)_ | ✓ k=5 | ✓ k=15 |   k=4 |   k=9 |   k=4 | ✓ k=291 | ✓ k=121 |   k=4 |   k=90 |   k=93 |   k=18 |   k=67 |   k=18 | ✓ k=5 | ✓ k=5 | ✓ k=10 | ✓ k=5 |
| Which of the three pretraining tasks is the most helpful? _(01853/ed2eb4)_ |   k=5 |   k=25 |   k=28 | ✓ k=47 |   k=47 | ✓ k=100 | ✓ k=58 | ✓ k=44 |   k=86 | ✓ k=13 | ✓ k=1 | ✓ k=11 | ✓ k=1 | ✓ k=5 |   k=5 | ✓ k=11 | ✓ k=5 |

---
Generated by `benchmarks/report_md.py`. Raw rows in `raw.jsonl`, aggregated stats in `aggregate.json`.