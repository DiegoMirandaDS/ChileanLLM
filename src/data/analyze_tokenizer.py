from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

INPUT_PATH = Path(
    "data/interim/sample_eda.parquet"
)

OUTPUT_PATH = Path(
    "reports/tokenizer_by_source.csv"
)

SAMPLE_PER_SOURCE = 2_000
SEED = 42


def tokenize_metrics(tokenizer, text):

    text = str(text)

    words = text.split()

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    n_words = len(words)
    n_tokens = len(token_ids)

    return {
        "n_words_tokenizer": n_words,
        "n_tokens": n_tokens,

        "tokens_per_word":
            n_tokens / max(n_words, 1),

        "tokens_per_char":
            n_tokens / max(len(text), 1),
    }


def main():

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded.")

    df = pd.read_parquet(
        INPUT_PATH
    )

    # -----------------------------------------
    # Sampleamos lo mismo por fuente para poder
    # compararlas de manera razonable.
    # -----------------------------------------

    samples = []

    for source in sorted(
        df["source"].unique()
    ):

        source_df = df[
            df["source"] == source
        ]

        n = min(
            SAMPLE_PER_SOURCE,
            len(source_df),
        )

        sample = source_df.sample(
            n=n,
            random_state=SEED,
        ).copy()

        samples.append(sample)

    sample_df = pd.concat(
        samples,
        ignore_index=True,
    )

    print(
        f"Analyzing {len(sample_df):,} documents..."
    )

    # -----------------------------------------
    # Tokenización
    # -----------------------------------------

    metrics = sample_df["text"].apply(
        lambda text: pd.Series(
            tokenize_metrics(
                tokenizer,
                text,
            )
        )
    )

    sample_df = pd.concat(
        [
            sample_df,
            metrics,
        ],
        axis=1,
    )

    # -----------------------------------------
    # Resumen por fuente
    # -----------------------------------------

    summary = (
        sample_df
        .groupby("source")
        .agg(
            documents=(
                "text",
                "count",
            ),

            median_words=(
                "n_words_tokenizer",
                "median",
            ),

            mean_tokens=(
                "n_tokens",
                "mean",
            ),

            median_tokens=(
                "n_tokens",
                "median",
            ),

            mean_tokens_per_word=(
                "tokens_per_word",
                "mean",
            ),

            median_tokens_per_word=(
                "tokens_per_word",
                "median",
            ),

            mean_tokens_per_char=(
                "tokens_per_char",
                "mean",
            ),
        )
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_PATH
    )

    print("\n")
    print("=" * 100)
    print("QWEN TOKENIZER ANALYSIS")
    print("=" * 100)

    print(
        summary.to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()