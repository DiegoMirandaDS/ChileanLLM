from collections import Counter
from pathlib import Path
import math
import shutil

import pandas as pd
import xxhash
import yaml

from datasets import Dataset, Features, Sequence, Value, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.clean_sample import clean_document, dedup_key


CONFIG_PATH = Path("configs/cpt_data.yaml")

EVAL_SAMPLE_PATH = Path(
    "data/processed/sample_clean.parquet"
)

TRAIN_OUTPUT_PATH = Path(
    "data/processed/cpt_train"
)

EVAL_OUTPUT_PATH = Path(
    "data/evaluation/cpt_eval"
)

BUFFER_SIZE = 100_000


def stable_hash(text, seed):
    normalized = dedup_key(text)

    return xxhash.xxh64(
        normalized.encode("utf-8"),
        seed=seed,
    ).intdigest()


def allocate_blocks(
    target_tokens,
    sequence_length,
    mixture,
):
    total_blocks = (
        target_tokens // sequence_length
    )

    raw = {
        source: total_blocks * fraction
        for source, fraction in mixture.items()
    }

    blocks = {
        source: math.floor(value)
        for source, value in raw.items()
    }

    remaining = total_blocks - sum(
        blocks.values()
    )

    order = sorted(
        mixture,
        key=lambda source: (
            raw[source] - blocks[source]
        ),
        reverse=True,
    )

    for source in order[:remaining]:
        blocks[source] += 1

    return blocks


def build_eval_source(
    texts,
    tokenizer,
    sequence_length,
    target_blocks,
    source,
    seed,
):
    rows = []
    hashes = set()
    token_buffer = []

    texts = sorted(
        texts,
        key=lambda text: stable_hash(
            text,
            seed,
        ),
    )

    for text in texts:

        if len(rows) >= target_blocks:
            break

        document_hash = stable_hash(
            text,
            seed,
        )

        hashes.add(document_hash)

        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        ids.append(
            tokenizer.eos_token_id
        )

        token_buffer.extend(ids)

        while (
            len(token_buffer) >= sequence_length
            and len(rows) < target_blocks
        ):
            block = token_buffer[
                :sequence_length
            ]

            del token_buffer[
                :sequence_length
            ]

            rows.append(
                {
                    "input_ids": block,
                    "source": source,
                }
            )

    return rows, hashes


def main():

    with open(
        CONFIG_PATH,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    seed = config["seed"]

    model_name = config["model"]["name"]

    sequence_length = (
        config["training"]["sequence_length"]
    )

    target_tokens = (
        config["pilot"]["max_train_tokens"]
    )

    mixture = (
        config["training_corpus"]["mixture"]
    )

    eval_blocks_per_source = (
        config["evaluation"][
            "blocks_per_source"
        ]
    )

    print(f"Model: {model_name}")
    print(
        f"Sequence length: {sequence_length}"
    )
    print(
        f"Maximum train tokens: "
        f"{target_tokens:,}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    # ========================================================
    # EVALUATION SET
    # ========================================================

    if not EVAL_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"{EVAL_SAMPLE_PATH} not found. "
            "Run Phase 1 cleaning first."
        )

    print("\nBuilding evaluation set...")

    eval_df = pd.read_parquet(
        EVAL_SAMPLE_PATH
    )

    eval_rows = []
    eval_hashes = set()

    for source in mixture:

        texts = (
            eval_df.loc[
                eval_df["source"] == source,
                "text",
            ]
            .astype(str)
            .tolist()
        )

        rows, hashes = build_eval_source(
            texts=texts,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            target_blocks=(
                eval_blocks_per_source
            ),
            source=source,
            seed=seed,
        )

        if len(rows) < eval_blocks_per_source:
            raise RuntimeError(
                f"Not enough eval data "
                f"for {source}"
            )

        eval_rows.extend(rows)
        eval_hashes.update(hashes)

        print(
            f"{source:12s}: "
            f"{len(rows):4,d} blocks | "
            f"{len(rows) * sequence_length:,} "
            "tokens"
        )

    # ========================================================
    # TRAIN TARGETS
    # ========================================================

    targets = allocate_blocks(
        target_tokens=target_tokens,
        sequence_length=sequence_length,
        mixture=mixture,
    )

    print("\nTraining targets:")

    for source, blocks in targets.items():
        print(
            f"{source:12s}: "
            f"{blocks:6,d} blocks | "
            f"{blocks * sequence_length:,} "
            "tokens"
        )

    # ========================================================
    # STREAM TRAINING CORPUS
    # ========================================================

    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["split"],
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=seed,
        buffer_size=BUFFER_SIZE,
    )

    buffers = {
        source: []
        for source in mixture
    }

    block_counts = Counter()

    seen_hashes = set(eval_hashes)

    train_rows = []

    scanned = 0
    invalid = 0
    duplicates = 0

    total_blocks = sum(
        targets.values()
    )

    progress = tqdm(
        total=total_blocks,
        desc="Packed blocks",
    )

    for row in dataset:

        scanned += 1

        source = row.get("source")

        if source not in targets:
            continue

        if (
            block_counts[source]
            >= targets[source]
        ):
            continue

        result = clean_document(
            row.get("text"),
            source,
        )

        if not result["valid"]:
            invalid += 1
            continue

        text = result["clean_text"]

        document_hash = stable_hash(
            text,
            seed,
        )

        if document_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(document_hash)

        ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]

        ids.append(
            tokenizer.eos_token_id
        )

        buffers[source].extend(ids)

        while (
            len(buffers[source])
            >= sequence_length
            and block_counts[source]
            < targets[source]
        ):
            block = buffers[source][
                :sequence_length
            ]

            del buffers[source][
                :sequence_length
            ]

            train_rows.append(
                {
                    "input_ids": block,
                    "source": source,
                }
            )

            block_counts[source] += 1

            progress.update(1)

        finished = all(
            block_counts[source]
            >= targets[source]
            for source in targets
        )

        if finished:
            break

    progress.close()

    missing = {
        source: (
            targets[source]
            - block_counts[source]
        )
        for source in targets
        if block_counts[source]
        < targets[source]
    }

    if missing:
        raise RuntimeError(
            f"Could not reach targets: {missing}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    features = Features(
        {
            "input_ids": Sequence(
                Value("int32")
            ),
            "source": Value("string"),
        }
    )

    train_dataset = Dataset.from_list(
        train_rows,
        features=features,
    )

    train_dataset = train_dataset.shuffle(
        seed=seed
    )

    eval_dataset = Dataset.from_list(
        eval_rows,
        features=features,
    )

    for path in (
        TRAIN_OUTPUT_PATH,
        EVAL_OUTPUT_PATH,
    ):
        if path.exists():
            shutil.rmtree(path)

    TRAIN_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EVAL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataset.save_to_disk(
        str(TRAIN_OUTPUT_PATH)
    )

    eval_dataset.save_to_disk(
        str(EVAL_OUTPUT_PATH)
    )

    print("\n" + "=" * 72)
    print("CPT DATASET READY")
    print("=" * 72)

    print(
        f"Rows scanned:       {scanned:,}"
    )
    print(
        f"Invalid skipped:    {invalid:,}"
    )
    print(
        f"Duplicates skipped: {duplicates:,}"
    )

    print(
        f"\nTrain blocks: {len(train_dataset):,}"
    )
    print(
        "Train tokens: "
        f"{len(train_dataset) * sequence_length:,}"
    )

    print(
        f"\nEval blocks: {len(eval_dataset):,}"
    )
    print(
        "Eval tokens: "
        f"{len(eval_dataset) * sequence_length:,}"
    )

    print("\nTraining mixture:")

    for source in mixture:

        blocks = block_counts[source]

        percentage = (
            100
            * blocks
            / len(train_dataset)
        )

        print(
            f"{source:12s}: "
            f"{blocks:6,d} "
            f"({percentage:6.2f}%)"
        )


if __name__ == "__main__":
    main()