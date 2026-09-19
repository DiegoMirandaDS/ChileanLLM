# ChileanLLM

Continual pre-training experiments for adapting a small base language model to Chilean Spanish using a multi-domain corpus.

The project studies the full adaptation pipeline around `Qwen3-0.6B-Base`: corpus analysis, data cleaning, tokenization, training-data design, continual pre-training and evaluation.

It is primarily an experimental and learning project. The objective is to build the pipeline carefully, understand the decisions involved in LLM adaptation, and test whether a relatively small CPT run can produce measurable improvements on Chilean Spanish.

**Base model:** `Qwen/Qwen3-0.6B-Base`  
**Dataset:** `jorgeortizfuentes/chilean-spanish-corpus`  
**Framework:** PyTorch + Hugging Face  
**Environment:** Linux / WSL2 + NVIDIA CUDA

---

## Current Status

Phase 1 is complete. The project is currently moving into the continual pre-training stage.

No CPT results are reported yet; training and Base-vs-CPT evaluation are still in progress.

```text
Data and corpus
[x] Corpus inspection
[x] Stratified EDA
[x] Data-quality analysis
[x] PII-aware cleaning
[x] Exact deduplication
[x] Chilean lexical analysis
[x] Tokenizer analysis
[x] Token-budget estimation
[x] CPT mixture design
[x] Base-model reevaluation

Training and evaluation
[~] Streaming CPT dataset
[~] Sequence packing
[ ] Base perplexity baseline
[ ] GPU smoke test
[ ] Continual pre-training
[ ] Base vs CPT evaluation
[ ] Post-training
```

---

## Technical Scope

The project develops practical experience with:

- continual pre-training of causal language models;
- PyTorch and Hugging Face Transformers;
- Hugging Face streaming datasets;
- NLP preprocessing and data-quality analysis;
- PII-aware text processing;
- dataset deduplication;
- tokenizer analysis;
- token-level dataset mixtures;
- sequence packing;
- language-model loss and perplexity;
- GPU memory-constrained training;
- mixed precision and gradient accumulation;
- gradient checkpointing;
- 8-bit optimizers with bitsandbytes;
- experiment configuration and reproducibility.

### Technologies

| Area | Technologies |
|---|---|
| Language / ML | Python, PyTorch, Transformers |
| Data | Hugging Face Datasets, pandas, PyArrow, Parquet |
| Text processing | regex, ftfy, xxHash |
| Training | CUDA, FP16, bitsandbytes |
| Evaluation | cross-entropy, perplexity |
| Experimentation | YAML configs, Git, W&B planned |
| Environment | Ubuntu / WSL2, RTX 3070 8 GB |

---

## Dataset

The project uses the gated `jorgeortizfuentes/chilean-spanish-corpus`, containing approximately 37 million documents.

| Source | Documents |
|---|---:|
| Twitter | 27.31M |
| mC4 `.cl` | 8.71M |
| News | 1.08M |
| Complaints | 31K |

A controlled 55K-document sample was built for exploratory analysis:

| Source | Documents |
|---|---:|
| Twitter | 20,000 |
| mC4 | 20,000 |
| News | 10,000 |
| Complaints | 5,000 |

The sample is intentionally stratified rather than representative of the natural corpus distribution. It was designed to compare the quality and linguistic characteristics of each source.

---

## Phase 1 — Data Preparation

**Status: complete**

Phase 1 focused on understanding the corpus before training.

The main work included:

- source-level corpus profiling;
- data-quality auditing;
- Chilean lexical-marker analysis;
- encoding and mojibake inspection;
- conservative text cleaning;
- PII detection and replacement;
- exact deduplication;
- tokenizer analysis;
- corpus token-budget estimation;
- CPT mixture design.

### Cleaning

The cleaning policy tries to remove technical noise without normalizing away useful linguistic variation.

Preserved:

- Chilean slang and voseo;
- informal spelling;
- accents and capitalization;
- emojis and hashtags;
- short social-media text.

Replaced or repaired:

```text
URLs       -> <URL>
mentions   -> <USER>
emails     -> <EMAIL>
RUT        -> <RUT>
phones     -> <PHONE>
```

HTML fragments, excessive whitespace and encoding corruption are also handled.

On the 55K-document sample:

```text
Original documents:   55,000
Invalid removed:          11
Duplicates removed:      702
Final documents:      54,287
Retention:             98.70%
```

More detailed decisions and observations are documented in
[`reports/Phase_1_Observations.md`](reports/Phase_1_Observations.md).

---

## Tokenizer Analysis

The tokenizer used by `Qwen3-0.6B-Base` was evaluated across the four domains.

| Source | Mean tokens / word |
|---|---:|
| Complaints | 1.525 |
| News | 1.653 |
| mC4 | 1.922 |
| Twitter | 2.598 |

Twitter is considerably more fragmented than the other sources.

To test whether this was caused specifically by Chilean vocabulary, tweets containing Chilean-associated lexical markers were compared against tweets without them.

After neutralizing URLs, mentions and hashtags:

| Twitter group | Tokens / word |
|---|---:|
| Chilean markers | 1.948 |
| No markers | 2.086 |

The experiment did not indicate that Chilean lexical items were responsible for the increased fragmentation.

The original Qwen3 tokenizer is therefore retained for CPT.

---

## Token Budget and Training Mixture

The cleaned corpus is estimated at approximately **8.49B Qwen3 tokens**.

Its natural token distribution is highly imbalanced:

| Source | Estimated tokens | Natural share |
|---|---:|---:|
| mC4 | 7.27B | 85.55% |
| Twitter | 0.72B | 8.44% |
| News | 0.51B | 5.96% |
| Complaints | 0.004B | 0.05% |

Document counts are therefore not used to define the training mixture.

The initial CPT mixture is:

| Source | Training share |
|---|---:|
| mC4 | 45% |
| Twitter | 35% |
| News | 19% |
| Complaints | 1% |

This is an experimental mixture intended to retain broad web language, informal Chilean Spanish, formal writing and user-generated language.

---

## Phase 2 — Continual Pre-Training

**Status: in progress**

Phase 2 covers:

- streaming training-data construction;
- token-level source balancing;
- sequence packing;
- fixed evaluation-set creation;
- base-model perplexity measurement;
- GPU throughput and memory profiling;
- full-parameter CPT;
- Base-vs-CPT evaluation.

The local experiments run on an **RTX 3070 with 8 GB of VRAM**, so the training configuration is intentionally small.

Current configuration:

```yaml
model: Qwen/Qwen3-0.6B-Base
sequence_length: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
optimizer: adamw_bnb_8bit
gradient_checkpointing: true
fp16: true
```

The first training run is time-boxed to approximately **105 minutes**.

Up to 12M training tokens can be prepared, but the effective pilot size will be chosen after measuring real throughput during the GPU smoke test.

---

## Phase 3 — Post-Training

**Status: planned**

After CPT, the project may be extended with:

- supervised fine-tuning;
- instruction tuning;
- preference/alignment experiments;
- Chilean-specific evaluation tasks.

The immediate priority is to finish and evaluate CPT before adding post-training stages.

---

## Evaluation

The initial experiment will compare:

- `Qwen3-0.6B-Base`
- the Chilean CPT checkpoint

using the same held-out evaluation data.

Initial metrics:

- cross-entropy loss;
- perplexity;
- aggregate evaluation loss;
- source-level evaluation.

Source-level results are important because an aggregate improvement could hide degradation in individual domains.

---

## Repository Structure

```text
ChileanLLM/
├── configs/
│   └── cpt_data.yaml
├── reports/
│   ├── Phase_1_Observations.md
│   ├── quality_audit.csv
│   ├── chilean_markers_by_source.csv
│   ├── tokenizer_by_source.csv
│   ├── chilean_tokenization.csv
│   ├── cleaning_sample_summary.csv
│   ├── token_budget_estimate.csv
│   └── cpt_mixture_plan.csv
├── src/
│   ├── data/
│   ├── training/
│   └── evaluation/
├── requirements.txt
└── README.md
```

Datasets, model checkpoints and experiment artifacts are not committed to the repository.

---

## Reproducing Phase 1

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
hf auth login
```

The dataset is gated and requires access through Hugging Face.

Run the analysis pipeline with:

```bash
python -m src.data.inspect_dataset
python -m src.data.create_eda_sample
python -m src.data.profile_dataset
python -m src.data.audit_sample
python -m src.data.analyze_tokenizer
python -m src.data.analyze_chilean_tokenization
python -m src.data.inspect_encoding_repairs
python -m src.data.clean_sample
python -m src.data.estimate_token_budget
python -m src.data.plan_cpt_mixture
```

---

## Limitations

- Training is currently limited to a single consumer GPU.
- The original corpus is strongly imbalanced across domains.
- PII detection is heuristic and cannot guarantee complete anonymization.
- Token-budget estimates are extrapolated from controlled samples.
- Chilean lexical markers are exploratory signals, not a dialect classifier.
- Twitter overrepresents informal and online communication.
- `.cl` web content does not necessarily imply Chilean linguistic features.
- Perplexity measures language-model fit but does not by itself measure cultural or linguistic quality.
- Dataset licensing and copyright restrictions must be considered before distributing derived artifacts.

The local experiment is intentionally small. The expected result is not a production-ready Chilean LLM, but evidence about whether a controlled CPT run produces a measurable adaptation signal and a reproducible implementation of the pipeline.