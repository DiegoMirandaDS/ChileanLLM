Phase 2 — Continual Pre-Training Results

1. Objective

Phase 2 tests whether a small full-parameter continual pre-training (CPT) run can produce a measurable adaptation signal on Chilean Spanish while remaining feasible on a single consumer GPU.

The experiment uses Qwen/Qwen3-0.6B-Base and the cleaned multi-domain Chilean corpus prepared in Phase 1.

The immediate experiment was intentionally constrained to approximately 12M training tokens. Larger budgets such as 50M tokens are left for later scaling experiments.

2. Training Data

The training corpus was constructed from four sources using a token-controlled mixture:

Source

Target share

Packed blocks

Tokens

mC4

45%

5,273

5,399,552

Twitter

35%

4,101

4,199,424

News

19%

2,227

2,280,448

Complaints

1%

117

119,808

Total

100%

11,718

11,999,232

Each block contains 1,024 tokens.

The source mixture is intentionally different from the corpus' natural distribution. The objective is to prevent mC4 from dominating training and to increase the contribution of informal Chilean Spanish, news and complaint text.

The full corpus had been estimated at approximately 8.49B Qwen3 tokens in Phase 1, so this pilot uses only a small subset of the available data.

3. Evaluation Set

A fixed held-out evaluation set was created before training.

Sources:             4
Blocks per source:   64
Tokens per block:    1,024
Tokens per source:   65,536
Total tokens:        262,144

Sources:

mC4

Twitter

News

Complaints

The evaluation set is balanced across domains. It is therefore useful for comparing source-level adaptation, but its overall score should not be interpreted as a corpus-frequency-weighted metric.

The same examples were used for the base model and CPT checkpoint.

4. Base Model Baseline

Model:

Qwen/Qwen3-0.6B-Base

Baseline results:

Domain

Loss

Perplexity

Overall

3.2714

26.3491

Complaints

3.4642

31.9506

mC4

2.6320

13.9010

News

2.8116

16.6364

Twitter

4.1780

65.2343

Twitter was the most difficult subset for the untouched base model.

5. Training Configuration

The pilot uses full-parameter continual pre-training.

Base model:                 Qwen/Qwen3-0.6B-Base
Sequence length:            1,024
Per-device batch size:      1
Gradient accumulation:      8
Effective tokens / step:    8,192
Precision:                  BF16
Optimizer:                  AdamW 8-bit (bitsandbytes)
Learning rate:              2e-5
Weight decay:               0.01
Warmup ratio:               0.03
Scheduler:                  cosine
Gradient checkpointing:     enabled
Gradient clipping:          1.0

Hardware:

GPU: NVIDIA GeForce RTX 3070
VRAM: 8 GB
Compute capability: 8.6
BF16 support: yes
Environment: Ubuntu / WSL2

6. Smoke Test

Before the main pilot, a 262,144-token training run was used to validate the complete training path.

Training tokens:      262,144
Optimizer steps:      32
Runtime:              1.41 min
Throughput:           3,098.5 tokens/s
Mean train loss:      3.0926
Peak GPU memory:      5.46 GB

The smoke test confirmed:

forward and backward passes;

BF16 training;

8-bit AdamW optimizer updates;

gradient accumulation;

gradient checkpointing;

checkpoint serialization;

sufficient VRAM headroom.

7. 12M-Token CPT Pilot

The final pilot processed 11,993,088 tokens.

The dataset contained 11,718 packed blocks, but training was rounded down to 11,712 blocks so that every optimizer update used a complete gradient-accumulation cycle.

Training results:

Processed tokens:     11,993,088
Optimizer steps:      1,464
Runtime:              61.56 min
Throughput:           3,246.9 tokens/s
Mean train loss:      2.9001
Peak GPU memory:      5.46 GB

The run completed below the 105-minute safety limit.

8. Base vs CPT Evaluation

The CPT checkpoint was evaluated on the same fixed 262,144-token evaluation set used for the baseline.

Domain

Base loss

CPT loss

Base PPL

CPT PPL

PPL change

Overall

3.2714

3.0105

26.3491

20.2982

-23.0%

Complaints

3.4642

3.3426

31.9506

28.2928

-11.4%

mC4

2.6320

2.5554

13.9010

12.8759

-7.4%

News

2.8116

2.6690

16.6364

14.4258

-13.3%

Twitter

4.1780

3.4751

65.2343

32.3024

-50.5%

The overall balanced-domain perplexity decreased by approximately 23%.

The largest improvement was observed on Twitter, where perplexity decreased by approximately 50.5%.

All four domain subsets improved in both loss and perplexity in this evaluation.

9. Interpretation

The 12M-token pilot produced a clear adaptation signal on the fixed held-out dataset.

The strongest effect appears on Twitter. This is consistent with two characteristics of the experiment:

Twitter had the highest baseline perplexity.

Twitter was intentionally upsampled to 35% of CPT tokens, compared with roughly 8.4% of the estimated natural corpus token distribution.

The results therefore suggest that the token-controlled mixture successfully increased adaptation to informal social-media language.

mC4, News and Complaints also improved, although by smaller margins.

No degradation was observed across the four evaluation subsets. This should not be interpreted as evidence that catastrophic forgetting is absent in general: the current evaluation only measures these four held-out corpus domains.

10. What This Experiment Shows

The experiment validates the end-to-end CPT pipeline:

raw streaming corpus
        ↓
quality analysis
        ↓
conservative cleaning
        ↓
deduplication
        ↓
token-level source mixture
        ↓
sequence packing
        ↓
fixed held-out evaluation
        ↓
base-model baseline
        ↓
full-parameter CPT
        ↓
checkpoint evaluation

It also demonstrates that full-parameter training of a 0.6B model is practical on an 8 GB RTX 3070 when using:

BF16;

gradient checkpointing;

micro-batch size 1;

gradient accumulation;

an 8-bit optimizer.

The 12M-token run took approximately 62 minutes and used 5.46 GB of peak allocated GPU memory.

11. Limitations

The current results should be interpreted within the scope of the experiment.

Only one CPT run is reported.

The training budget is small relative to the estimated 8.49B-token corpus.

The evaluation set contains 262K tokens and is balanced across four domains.

The evaluation set is derived from the same corpus family as the training data, although held out from the training set.

Perplexity measures next-token modeling performance, not instruction following, reasoning or cultural knowledge.

No external benchmark has yet been used.

No broad forgetting evaluation has yet been performed.

Hyperparameters and source mixture have not been systematically swept.

The results do not establish that 12M tokens is an optimal training budget.

12. Next Steps

The next phase will extend evaluation and add post-training.

Potential work:

supervised / instruction fine-tuning;

qualitative Chilean-Spanish generation tests;

external language-model benchmarks;

targeted Chilean-language evaluation;

comparison against the untouched base model on general-domain tasks;

longer CPT runs.

A future 50M-token experiment is particularly useful as a scaling comparison. Based on the measured throughput of the 12M pilot, a 50M-token run would require roughly 4–5 hours on the current hardware, assuming similar throughput.

The 12M checkpoint remains the reproducible baseline CPT experiment for the project.