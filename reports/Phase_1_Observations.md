# Phase 1 — Observations and Technical Decisions

This document summarizes the main findings and decisions made during the data preparation phase of ChileanLLM.

## 1. Dataset selection

**Observation:** The initial `universal_spanish_chilean_corpus` contained both Chilean and broader Spanish data.

**Decision:** Use `jorgeortizfuentes/chilean-spanish-corpus` as the primary CPT corpus.

**Reason:** It provides a cleaner experimental setting for Chilean Spanish adaptation.

---

## 2. Sampling

**Problem:** Streaming shuffle did not produce a representative sample. Rare sources such as News and Complaints were poorly represented because `IterableDataset.shuffle()` only mixes within a finite buffer.

**Decision:** Build a controlled 55K-document stratified sample:

| Source     | Documents |
| ---------- | --------: |
| Twitter    |    20,000 |
| mC4        |    20,000 |
| News       |    10,000 |
| Complaints |     5,000 |

**Reason:** Phase 1 required comparing domains, not reproducing their natural proportions.

---

## 3. Document distribution

**Observation:** Corpus sources differ substantially in document length.

| Source     | Median words |
| ---------- | -----------: |
| Twitter    |           12 |
| Complaints |           64 |
| News       |          240 |
| mC4        |        297.5 |

**Decision:** Do not use document counts to determine the training mixture.

**Reason:** Documents contribute very different numbers of training tokens.

---

## 4. Chilean lexical markers

**Observation:** Twitter showed the highest density of several informal Chilean-associated markers, including slang, discourse markers, internet abbreviations and `weón`-family forms.

**Problem:** Some initial regex patterns generated false positives.

Examples:

* `tenis` matched both voseo `tenís` and the sport *tenis*.
* `de una` was too common in general Spanish.

**Decision:** Remove both expressions from the marker set.

**Reason:** The markers are exploratory linguistic signals, not a Chilean-language classifier.

---

## 5. Qwen tokenizer

**Observation:**

| Source     | Mean tokens / word |
| ---------- | -----------------: |
| Complaints |               1.53 |
| News       |               1.65 |
| mC4        |               1.92 |
| Twitter    |               2.60 |

Twitter initially appeared substantially more fragmented.

**Test:** Tweets containing Chilean-associated markers were compared with tweets without markers.

After neutralizing URLs, mentions and hashtags:

| Group           | Tokens / word |
| --------------- | ------------: |
| Chilean markers |          1.95 |
| No markers      |          2.09 |

**Decision:** Keep the original Qwen tokenizer.

**Reason:** Chilean lexical markers themselves were not responsible for the higher Twitter fragmentation. The difference appears to come mainly from social-media structure and writing style.

---

## 6. Encoding problems

**Observation:** mC4 contained genuine mojibake:

```text
PiÃ±era → Piñera
PÃºblicos → Públicos
```

**Initial approach:** `ftfy.fix_text()`.

**Problem:** It modified ~45% of mC4 and ~82% of News, indicating transformations beyond encoding repair.

**Decision:** Replace it with `ftfy.fix_encoding()`.

**Result:**

| Source     | Encoding repairs |
| ---------- | ---------------: |
| mC4        |            3.12% |
| News       |            0.05% |
| Twitter    |              ~0% |
| Complaints |              ~0% |

**Reason:** `fix_encoding()` repairs mojibake while minimizing unnecessary changes to valid text.

---

## 7. Cleaning policy

**Decision:** Use conservative cleaning.

Preserved:

* slang,
* voseo,
* spelling variation,
* accents,
* capitalization,
* emojis,
* hashtags,
* short tweets.

Replaced or repaired:

```text
URLs       → <URL>
mentions   → <USER>
emails     → <EMAIL>
RUT        → <RUT>
phones     → <PHONE>
```

HTML fragments, mojibake, excessive whitespace and exact duplicates are also removed or repaired.

**Reason:** The objective is to remove technical noise without removing Chilean linguistic variation.

---

## 8. PII detection

**Problem:** The initial phone-number regex produced implausibly high detection rates in web text.

**Decision:** Restrict phone detection to more plausible Chilean mobile-number patterns.

**Reason:** Broad numerical regexes were incorrectly matching IDs, product codes and other numbers.

PII detection remains heuristic and does not guarantee complete anonymization.

---

## 9. Cleaning results

On the 55K-document sample:

```text
Original:            55,000
Invalid removed:         11
Duplicates removed:     702
Final:               54,287
Retention:            98.70%
```

Duplicate removals:

```text
Twitter       603
News           92
Complaints      7
mC4             0
```

**Conclusion:** The pipeline is intentionally conservative and retains nearly all useful text.

---

## 10. Token distribution

Estimated cleaned corpus:

```text
~8.49B Qwen tokens
```

| Source     | Estimated tokens |  Share |
| ---------- | ---------------: | -----: |
| mC4        |            7.27B | 85.55% |
| Twitter    |            0.72B |  8.44% |
| News       |            0.51B |  5.96% |
| Complaints |           0.004B |  0.05% |

**Key observation:** Twitter dominates document count, but mC4 dominates token count.

**Decision:** Do not train using the natural corpus distribution.

---

## 11. Initial CPT mixture

Proposed token-level mixture:

| Source     | Share |
| ---------- | ----: |
| mC4        |   45% |
| Twitter    |   35% |
| News       |   19% |
| Complaints |    1% |

**Reason:**

* mC4 provides broad Chilean web language.
* Twitter provides informal and dialectal signal.
* News preserves formal Chilean Spanish.
* Complaints provide spontaneous user-generated language.

This mixture is experimental and may change after evaluation.

---

## 12. Training budget

Estimated corpus size is much larger than required for the first experiment.

**Decision:**

```text
Pilot CPT:  50M tokens
Main CPT:  300M tokens
```

The pilot will validate the training pipeline before the larger run.

For the 300M-token experiment:

```text
mC4          135M
Twitter      105M
News          57M
Complaints     3M
```

---

## Phase 1 conclusion

Phase 1 established three main principles for the project:

1. **Training composition must be controlled by tokens rather than documents.**
2. **Cleaning must remove technical noise without normalizing away Chilean language.**
3. **The tokenizer does not require modification for the initial CPT experiment.**

The next phase will build the streaming training pipeline, establish the base-model evaluation baseline, and run the first CPT pilot.
