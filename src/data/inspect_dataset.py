from collections import Counter
from datasets import load_dataset


DATASET_NAME = "jorgeortizfuentes/chilean-spanish-corpus"

N_SAMPLES = 5000
BUFFER_SIZE = 50_000
SEED = 42


def main():
    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=BUFFER_SIZE,
    )

    source_counts = Counter()

    examples = {}

    for i, row in enumerate(dataset):
        source = row.get("source")
        text = str(row.get("text", ""))

        source_counts[source] += 1

        if source not in examples:
            examples[source] = text[:500]

        if i + 1 >= N_SAMPLES:
            break

    print("\nSOURCE COUNTS")
    print("=" * 60)

    for source, count in source_counts.most_common():
        print(f"{source}: {count}")

    print("\nEXAMPLE PER SOURCE")
    print("=" * 60)

    for source, text in examples.items():
        print(f"\nSOURCE = {source}")
        print("-" * 60)
        print(text)


if __name__ == "__main__":
    main()