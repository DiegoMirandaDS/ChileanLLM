# ChileanLM Lab

Continual pre-training experiments for adapting a base language model to Chilean Spanish using a multi-domain corpus.

The project studies how Chilean web text, social media, news, and user-generated complaints differ in quality, linguistic characteristics, tokenization behavior, and training volume before performing continual pre-training (CPT).

## Project Goal

The main goal is to evaluate whether continual pre-training can improve the modeling of Chilean Spanish while preserving exposure to both informal and formal registers.

The project starts from:

* **Base model:** `Qwen/Qwen2.5-0.5B`
* **Dataset:** `jorgeortizfuentes/chilean-spanish-corpus`
* **Framework:** PyTorch + Hugging Face
* **Environment:** Linux / WSL2

The planned experimental comparison is:

```text
Qwen2.5-0.5B Base
        ↓
Continual Pre-Training
        ↓
Chilean-Qwen CPT
        ↓
Evaluation
```

Later stages will compare the base and adapted models using language-modeling metrics and Chilean Spanish evaluation data.

---

## Dataset

The Chilean Spanish Corpus contains approximately 37 million documents from four domains:

| Source     |  Documents |
| ---------- | ---------: |
| Twitter    | 27,306,583 |
| mC4 `.cl`  |  8,706,681 |
| News       |  1,081,542 |
| Complaints |     31,219 |

Although Twitter dominates by document count, document lengths vary substantially between sources.

This makes document proportions misleading for pre-training. The effective training distribution must instead be analyzed in terms of **tokens**.

---

## Phase 1 — Data Analysis and Preparation

Phase 1 focuses on understanding and preparing the corpus before training.

The pipeline includes:

```text
Hugging Face dataset
        ↓
Streaming inspection
        ↓
Stratified EDA sample
        ↓
Quality profiling
        ↓
Chilean lexical analysis
        ↓
Tokenizer analysis
        ↓
Encoding repair
        ↓
PII anonymization
        ↓
Deduplication
        ↓
Clean corpus estimation
        ↓
CPT mixture design
```

---

## Exploratory Dataset Sample

A controlled 55,000-document sample was constructed for EDA:

| Source     | Documents |
| ---------- | --------: |
| Twitter    |    20,000 |
| mC4        |    20,000 |
| News       |    10,000 |
| Complaints |     5,000 |

The sample is intentionally not representative of the natural corpus distribution. Minority domains were oversampled to enable meaningful quality analysis.

---

## Domain Characteristics

The initial profiling showed substantial differences between sources.

| Source     | Median words | Mean words |
| ---------- | -----------: | ---------: |
| Twitter    |           12 |       14.0 |
| Complaints |           64 |       91.3 |
| News       |          240 |      288.7 |
| mC4        |        297.5 |      453.9 |

Twitter is therefore extremely short relative to the other domains.

Additional findings included:

* Twitter contained high rates of URLs and user mentions.
* mC4 contained web artifacts, HTML, and encoding corruption.
* News contained measurable exact duplication.
* Complaints contained user-generated natural language and required additional PII inspection.

---

## Chilean Spanish Analysis

Rather than using a single binary definition of "Chilean text", lexical markers were grouped into exploratory categories:

* discourse markers
* `cachai` family
* Chilean voseo
* slang
* `weón` family
* Chilean internet abbreviations
* multi-word expressions

These markers are treated only as **Chilean-associated lexical signals**, not as a classifier of text nationality or dialect.

The analysis showed that Twitter had substantially higher marker density for several informal categories, including:

* `weón` family
* internet abbreviations
* slang
* discourse markers

This supports keeping Twitter as an important part of the CPT mixture despite its relatively low token volume.

---

## Tokenizer Analysis

The Qwen2.5 tokenizer was evaluated before training.

Average tokens per word:

| Source     | Tokens / word |
| ---------- | ------------: |
| Complaints |          1.53 |
| News       |          1.65 |
| mC4        |          1.92 |
| Twitter    |          2.60 |

Twitter showed considerably higher fragmentation.

However, a controlled experiment comparing tweets **with** and **without** Chilean-associated lexical markers produced a different result.

After replacing URLs, mentions, and hashtags with neutral placeholders:

| Twitter group   | Mean tokens / word |
| --------------- | -----------------: |
| Chilean markers |               1.95 |
| No markers      |               2.09 |

This suggests that Chilean lexical items themselves are not the main source of Twitter tokenization inefficiency.

Instead, fragmentation appears to be influenced by the structural and orthographic characteristics of social-media text.

The tokenizer will therefore remain unchanged during CPT.

---

## Data Cleaning Strategy

Cleaning is intentionally conservative.

The objective is to remove technical noise without normalizing away dialectal information.

### Preserved

* Chilean slang
* voseo
* spelling variation
* short tweets
* emojis
* punctuation
* capitalization
* hashtags

### Replaced or repaired

* URLs → `<URL>`
* user mentions → `<USER>`
* emails → `<EMAIL>`
* Chilean RUT patterns → `<RUT>`
* phone numbers → `<PHONE>`
* malformed HTML
* mojibake / encoding corruption
* excessive whitespace
* exact duplicates

Text is **not lowercased**, accent marks are preserved, and informal spelling is not automatically corrected.

---

## Encoding Repair

Initial experiments with `ftfy.fix_text()` modified an unexpectedly large fraction of the corpus.

A more conservative approach using `ftfy.fix_encoding()` was therefore adopted.

Observed encoding repair rates:

| Source     | Documents repaired |
| ---------- | -----------------: |
| mC4        |              ~3.1% |
| News       |             ~0.05% |
| Twitter    |                ~0% |
| Complaints |                ~0% |

Examples included genuine mojibake corrections such as:

```text
PiÃ±era   → Piñera
PÃºblicos → Públicos
chocÃ³    → chocó
```

---

## Cleaning Results

The cleaning pipeline was tested on the 55,000-document EDA sample.

```text
Original documents:   55,000
Invalid removed:          11
Duplicates removed:      702
Final documents:      54,287
Retention:             98.70%
```

Retention by source:

| Source     | Retention |
| ---------- | --------: |
| mC4        |    99.96% |
| Complaints |    99.82% |
| News       |    99.07% |
| Twitter    |    96.99% |

The high retention rate reflects the intentionally conservative cleaning policy.

---

## Estimated Corpus Token Budget

Token counts were estimated after cleaning using the Qwen2.5 tokenizer.

| Source     | Estimated tokens | Token share |
| ---------- | ---------------: | ----------: |
| mC4        |            7.27B |      85.55% |
| Twitter    |            0.72B |       8.44% |
| News       |            0.51B |       5.96% |
| Complaints |           0.004B |       0.05% |

Estimated total:

```text
~8.49 billion tokens
```

This reveals an important property of the corpus:

> Twitter dominates the number of documents, but mC4 overwhelmingly dominates the number of training tokens.

Therefore, using the natural corpus distribution would mostly perform adaptation to Chilean web content rather than balanced Chilean Spanish adaptation.

---

## Proposed CPT Mixture

The initial CPT mixture is therefore defined in terms of **tokens**, not documents.

| Source     | Target token share |
| ---------- | -----------------: |
| mC4        |                45% |
| Twitter    |                35% |
| News       |                19% |
| Complaints |                 1% |

This is an experimental mixture and may be revised based on validation results.

The goal is to preserve:

* broad Chilean web language through mC4,
* informal and dialectal signal through Twitter,
* formal language through news,
* spontaneous user-generated language through complaints.

---

## Training Plan

Two CPT stages are planned.

### Pilot

```text
50M tokens
```

Used to validate:

* sequence packing
* training stability
* loss behavior
* GPU memory usage
* throughput
* checkpointing
* experiment tracking

### Main CPT

```text
300M tokens
```

Proposed allocation:

| Source     | Tokens |
| ---------- | -----: |
| mC4        |   135M |
| Twitter    |   105M |
| News       |    57M |
| Complaints |     3M |

The complete ~8.5B-token corpus will not be trained in the initial experiment.

---

## Repository Structure

```text
chileanlm-lab/
│
├── configs/
│   └── cpt_data.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── evaluation/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── quality_audit.csv
│   ├── chilean_markers_by_source.csv
│   ├── tokenizer_by_source.csv
│   ├── chilean_tokenization.csv
│   ├── cleaning_sample_summary.csv
│   └── token_budget_estimate.csv
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── inspect_sources.py
│   │   ├── create_eda_sample.py
│   │   ├── profile_dataset.py
│   │   ├── audit_sample.py
│   │   ├── inspect_samples.py
│   │   ├── inspect_chilean_markers.py
│   │   ├── inspect_encoding_repairs.py
│   │   ├── analyze_tokenizer.py
│   │   ├── analyze_chilean_tokenization.py
│   │   ├── clean_sample.py
│   │   └── estimate_token_budget.py
│   │
│   ├── training/
│   └── evaluation/
│
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

Large datasets and model checkpoints are intentionally excluded from Git.

---

## Environment

Tested using:

```text
WSL2
Ubuntu
Python 3.12
PyTorch
Transformers
Datasets
PyArrow
pandas
ftfy
```

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Authenticate with Hugging Face:

```bash
hf auth login
```

The Chilean Spanish Corpus is gated and requires access through Hugging Face.

---

## Reproducing Phase 1

The main analysis pipeline can be reproduced with:

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
```

Intermediate Parquet files are generated locally and are not committed to Git.

---

## Data Governance and Limitations

Several limitations must be considered:

* The dataset is strongly imbalanced by source.
* `.cl` web domains do not guarantee that every document represents Chilean dialect.
* Lexical markers are exploratory heuristics and are not a dialect classifier.
* Twitter language may overrepresent informal and online communication.
* News and web sources represent different linguistic registers.
* User-generated complaints may contain personally identifiable information.
* Regex-based PII detection cannot guarantee complete anonymization.
* Estimated token counts are extrapolated from a stratified sample rather than a complete corpus tokenization.
* Copyright and licensing constraints of upstream sources must be respected when distributing derived datasets or trained models.

---

## Current Status

### Phase 1 — Data Preparation

* [x] Environment setup
* [x] Dataset streaming
* [x] Stratified EDA
* [x] Quality profiling
* [x] PII analysis
* [x] Chilean lexical analysis
* [x] Qwen tokenizer analysis
* [x] Encoding repair
* [x] Conservative cleaning pipeline
* [x] Token-budget estimation
* [x] Initial CPT mixture design

### Phase 2 — Continual Pre-Training

* [ ] Build streaming training dataset
* [ ] Tokenize and pack sequences
* [ ] Establish baseline perplexity
* [ ] Run 50M-token pilot
* [ ] Analyze training dynamics
* [ ] Run main CPT experiment
* [ ] Compare Base vs CPT

### Phase 3 — Post-Training and Evaluation

Planned for later stages of the project.

---

## Motivation

This project is designed as an end-to-end experiment in language-model adaptation rather than a simple fine-tuning exercise.

It covers:

* large-scale dataset streaming,
* corpus analysis,
* data-quality diagnostics,
* linguistic analysis,
* tokenization analysis,
* privacy-aware preprocessing,
* domain mixture design,
* continual pre-training,
* evaluation,
* and reproducible ML experimentation.
