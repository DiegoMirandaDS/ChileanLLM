# ChileanLLM

Continual pre-training experiments for adapting `Qwen3-0.6B-Base` to Chilean Spanish using a multi-domain corpus.

The project covers corpus analysis, data cleaning, tokenizer analysis, training-data design, full-parameter continual pre-training and evaluation under consumer-GPU constraints.

**Base model:** `Qwen/Qwen3-0.6B-Base`  
**Dataset:** `jorgeortizfuentes/chilean-spanish-corpus`  
**Framework:** PyTorch + Hugging Face  
**Training hardware:** NVIDIA RTX 3070 8 GB

---

## Current Status

```text
Phase 1 — Data preparation
[x] Corpus inspection and stratified EDA
[x] Data-quality audit
[x] PII-aware cleaning
[x] Exact deduplication
[x] Chilean lexical analysis
[x] Tokenizer analysis
[x] Token-budget estimation
[x] CPT mixture design

Phase 2 — Continual pre-training
[x] Packed CPT dataset
[x] Fixed held-out evaluation set
[x] Base-model baseline
[x] CUDA/BF16 smoke test
[x] Full-parameter CPT (12M tokens)
[x] Base vs CPT evaluation

Phase 3 — Post-training
[ ] Supervised / instruction fine-tuning
[ ] Broader Chilean-language evaluation
```

The 12M-token CPT pilot is complete. On the fixed balanced evaluation set, perplexity decreased from **26.35 to 20.30** overall. The largest change was observed on Twitter, where perplexity decreased from **65.23 to 32.30**.

---

## Results

The base model and CPT checkpoint were evaluated on the same held-out set of **262,144 tokens**, balanced equally across mC4, Twitter, News and Complaints.

| Domain | Base loss | CPT loss | Base PPL | CPT PPL | PPL change |
|---|---:|---:|---:|---:|---:|
| Overall | 3.2714 | **3.0105** | 26.35 | **20.30** | **-23.0%** |
| mC4 | 2.6320 | **2.5554** | 13.90 | **12.88** | **-7.4%** |
| News | 2.8116 | **2.6690** | 16.64 | **14.43** | **-13.3%** |
| Complaints | 3.4642 | **3.3426** | 31.95 | **28.29** | **-11.4%** |
| Twitter | 4.1780 | **3.4751** | 65.23 | **32.30** | **-50.5%** |

No degradation was observed across the four held-out subsets in this evaluation.

Because the evaluation set contains the same number of blocks from each source, the overall metric is a **balanced domain average**, not an estimate weighted by the corpus' natural distribution.

Detailed Phase 2 results are in [`reports/Phase_2_Results.md`](reports/Phase_2_Results.md).

---

## Training Run

The CPT pilot used full-parameter training.

```text
Model:                    Qwen/Qwen3-0.6B-Base
Training tokens:          11,993,088
Sequence length:          1024
Micro-batch size:         1
Gradient accumulation:    8
Effective tokens / step:  8,192
Optimizer steps:          1,464
Precision:                BF16
Optimizer:                AdamW 8-bit
Scheduler:                cosine
Gradient checkpointing:   enabled
Runtime:                  61.56 min
Throughput:               3,246.9 tokens/s
Peak GPU memory:          5.46 GB
GPU:                      NVIDIA RTX 3070 8 GB
```

A 262K-token smoke test was run before the full pilot to validate backward passes, optimizer updates, checkpoint saving, throughput and GPU memory use.

---

## Dataset

The project uses the gated `jorgeortizfuentes/chilean-spanish-corpus`, containing approximately 37 million documents.

| Source | Documents |
|---|---:|
| Twitter | 27.31M |
| mC4 `.cl` | 8.71M |
| News | 1.08M |
| Complaints | 31K |

The cleaned corpus is estimated at approximately **8.49B Qwen3 tokens**.

Its natural token distribution is strongly imbalanced:

| Source | Estimated tokens | Natural share |
|---|---:|---:|
| mC4 | 7.27B | 85.55% |
| Twitter | 0.72B | 8.44% |
| News | 0.51B | 5.96% |
| Complaints | 0.004B | 0.05% |

For CPT, the training mixture was controlled by token count rather than document count:

| Source | Training share |
|---|---:|
| mC4 | 45% |
| Twitter | 35% |
| News | 19% |
| Complaints | 1% |

The resulting 12M-token training dataset contained:

| Source | Packed blocks | Tokens |
|---|---:|---:|
| mC4 | 5,273 | 5,399,552 |
| Twitter | 4,101 | 4,199,424 |
| News | 2,227 | 2,280,448 |
| Complaints | 117 | 119,808 |
| **Total** | **11,718** | **11,999,232** |

Training used 11,712 blocks so that every optimizer step contained a complete gradient-accumulation cycle.

---

## Phase 1 — Data Preparation

Phase 1 focused on understanding the corpus before training.

The pipeline includes:

- source-level corpus profiling;
- stratified exploratory analysis;
- quality auditing;
- conservative text cleaning;
- PII detection and replacement;
- exact deduplication;
- encoding/mojibake inspection;
- Chilean lexical-marker analysis;
- tokenizer analysis;
- token-budget estimation;
- token-level CPT mixture design.

The cleaning policy preserves useful linguistic variation such as Chilean slang, voseo, informal spelling, accents, capitalization, emojis, hashtags and short social-media text.

Technical noise and selected PII patterns are replaced or repaired:

```text
URLs       -> <URL>
mentions   -> <USER>
emails     -> <EMAIL>
RUT        -> <RUT>
phones     -> <PHONE>
```

On the controlled 55K-document sample:

```text
Original documents:   55,000
Invalid removed:          11
Duplicates removed:      702
Final documents:      54,287
Retention:             98.70%
```

More detail is available in [`reports/Phase_1_Observations.md`](reports/Phase_1_Observations.md).

---

## Tokenizer Analysis

`Qwen3-0.6B-Base` was evaluated across the four corpus domains.

| Source | Mean tokens / word |
|---|---:|
| Complaints | 1.525 |
| News | 1.653 |
| mC4 | 1.922 |
| Twitter | 2.598 |

Twitter is substantially more fragmented than the other domains. A controlled comparison between tweets with and without Chilean-associated lexical markers did not indicate that those lexical items were responsible for the difference.

The original Qwen3 tokenizer was therefore retained for CPT.

---

## Evaluation

A fixed evaluation set was created before training.

```text
64 blocks per source
4 sources
1024 tokens per block
262,144 evaluation tokens total
```

The same set is used for both the base model and the CPT checkpoint.

Metrics:

- causal language-model cross-entropy loss;
- perplexity;
- aggregate balanced-domain performance;
- source-level performance.

The current evaluation establishes a measurable adaptation signal, but it does not by itself measure instruction-following ability, cultural knowledge, downstream-task performance or general capabilities outside these four domains.

---

## Technologies

| Area | Tools |
|---|---|
| Language / ML | Python, PyTorch, Transformers |
| Data | Hugging Face Datasets, pandas, PyArrow, Parquet |
| Text processing | regex, ftfy, xxHash |
| Training | CUDA, BF16, gradient checkpointing, bitsandbytes |
| Evaluation | cross-entropy, perplexity |
| Experiment config | YAML |
| Environment | Ubuntu / WSL2, NVIDIA RTX 3070 |

---

## Repository Structure

```text
ChileanLLM/
├── configs/
│   └── cpt_data.yaml
├── reports/
│   ├── Phase_1_Observations.md
│   ├── Phase_2_Results.md
│   ├── evaluation/
│   │   ├── baseline_qwen3.json
│   │   └── cpt_12m_qwen3.json
│   └── *.csv
├── src/
│   ├── data/
│   │   └── build_cpt_dataset.py
│   ├── training/
│   │   └── train_cpt.py
│   └── evaluation/
│       └── evaluate_lm.py
├── requirements.txt
└── README.md
```

Datasets and model checkpoints are intentionally excluded from the repository.

---

## Reproduction

Create the environment and authenticate with Hugging Face:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
hf auth login
```

The dataset is gated and requires Hugging Face access.

Build the CPT and evaluation datasets:

```bash
python -m src.data.build_cpt_dataset
```

Evaluate the untouched base model:

```bash
python -m src.evaluation.evaluate_lm \
  --output reports/evaluation/baseline_qwen3.json
```

Run the GPU smoke test:

```bash
python -m src.training.train_cpt --smoke-test
```

Run the 12M-token CPT pilot:

```bash
python -m src.training.train_cpt
```

Evaluate the CPT checkpoint:

```bash
python -m src.evaluation.evaluate_lm \
  --model checkpoints/qwen3_chilean_cpt \
  --output reports/evaluation/cpt_12m_qwen3.json
```

---

## Next Steps

Phase 3 will focus on post-training and broader evaluation. Possible extensions include:

- supervised fine-tuning / instruction tuning;
- Chilean-specific downstream evaluation;
- qualitative generation analysis;
- comparison against the untouched base model on external benchmarks;
- longer CPT runs such as 50M tokens when additional training time is available.

A larger run would be treated as a scaling experiment rather than replacing the 12M-token pilot.

---

## Limitations

- Training was performed on a single consumer GPU.
- The original corpus is highly imbalanced across domains.
- The evaluation set is balanced by source and is not representative of natural corpus proportions.
- PII detection is heuristic and cannot guarantee complete anonymization.
- Token-budget estimates are extrapolated from controlled samples.
- Chilean lexical markers are exploratory signals, not a dialect classifier.
- Twitter overrepresents informal and online communication.
- `.cl` web content does not necessarily imply Chilean linguistic characteristics.
- Perplexity measures language-model fit, not cultural alignment or general model quality.
- The current results cover one 12M-token training run and one fixed held-out evaluation set.
- Dataset licensing and copyright restrictions must be considered before distributing derived artifacts.

The goal of the project is a reproducible adaptation experiment and practical LLM-training pipeline, not a production-ready Chilean language model.
