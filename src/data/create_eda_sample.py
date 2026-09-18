from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


DATASET_NAME = "jorgeortizfuentes/chilean-spanish-corpus"

TARGETS = {
    "twitter": 20_000,
    "mc4": 20_000,
    "news": 10_000,
    "complaints": 5_000,
}

SEED = 42
BUFFER_SIZE = 100_000

OUTPUT_PATH = Path("data/interim/sample_eda.parquet")


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading streaming dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Shuffle de shards + buffer local.
    # No es un shuffle global perfecto,
    # pero evita simplemente consumir el dataset desde el comienzo.
    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=BUFFER_SIZE,
    )

    counts = Counter()

    rows = []

    total_target = sum(TARGETS.values())

    scanned = 0

    progress = tqdm(
        total=total_target,
        desc="Collected",
    )

    for row in dataset:

        scanned += 1

        source = row.get("source")

        if source not in TARGETS:
            continue

        if counts[source] >= TARGETS[source]:
            continue

        text = row.get("text")

        if text is None:
            continue

        rows.append(
            {
                "text": str(text),
                "source": source,
            }
        )

        counts[source] += 1

        progress.update(1)

        if scanned % 100_000 == 0:
            print(
                f"\nScanned: {scanned:,}"
                f" | collected: {dict(counts)}"
            )

        finished = all(
            counts[source] >= target
            for source, target in TARGETS.items()
        )

        if finished:
            break

    progress.close()

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Rows scanned: {scanned:,}")
    print(f"Rows collected: {len(df):,}")

    print("\nBy source:")
    print(df["source"].value_counts())

    print("\nSaving...")

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()