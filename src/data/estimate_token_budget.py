from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

INPUT_PATH = Path(
    "data/processed/sample_clean.parquet"
)

OUTPUT_PATH = Path(
    "reports/token_budget_estimate.csv"
)


# ============================================================
# TAMAÑO COMPLETO DEL CHILEAN SPANISH CORPUS
# ============================================================

CORPUS_DOCUMENT_COUNTS = {
    "twitter": 27_306_583,
    "mc4": 8_706_681,
    "news": 1_081_542,
    "complaints": 31_219,
}


# Tasas de retención observadas en nuestra muestra limpia.
CLEANING_RETENTION = {
    "twitter": 0.96985,
    "mc4": 0.99960,
    "news": 0.99070,
    "complaints": 0.99820,
}


def main():

    print(
        f"Loading tokenizer: {MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Loaded {len(df):,} cleaned documents"
    )

    rows = []

    # ========================================================
    # CALCULAR TOKENS REALES EN NUESTRA MUESTRA LIMPIA
    # ========================================================

    for source in sorted(
        df["source"].unique()
    ):

        source_df = df[
            df["source"] == source
        ]

        print(
            f"\nTokenizing {source}: "
            f"{len(source_df):,} documents..."
        )

        token_counts = []

        for text in source_df["text"]:

            ids = tokenizer(
                str(text),
                add_special_tokens=False,
            )["input_ids"]

            token_counts.append(
                len(ids)
            )

        mean_tokens = (
            sum(token_counts)
            / len(token_counts)
        )

        median_tokens = float(
            pd.Series(
                token_counts
            ).median()
        )

        original_docs = (
            CORPUS_DOCUMENT_COUNTS[source]
        )

        retention = (
            CLEANING_RETENTION[source]
        )

        estimated_clean_docs = (
            original_docs
            * retention
        )

        estimated_tokens = (
            estimated_clean_docs
            * mean_tokens
        )

        rows.append(
            {
                "source":
                    source,

                "sample_documents":
                    len(source_df),

                "mean_tokens_per_document":
                    mean_tokens,

                "median_tokens_per_document":
                    median_tokens,

                "corpus_documents":
                    original_docs,

                "retention_rate":
                    retention,

                "estimated_clean_documents":
                    round(
                        estimated_clean_docs
                    ),

                "estimated_tokens":
                    round(
                        estimated_tokens
                    ),
            }
        )

    # ========================================================
    # DATAFRAME
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    total_tokens = (
        results[
            "estimated_tokens"
        ].sum()
    )

    results[
        "estimated_token_share"
    ] = (
        results["estimated_tokens"]
        / total_tokens
    )

    results[
        "estimated_tokens_billions"
    ] = (
        results["estimated_tokens"]
        / 1_000_000_000
    )

    # Ordenamos de mayor a menor volumen
    results = results.sort_values(
        "estimated_tokens",
        ascending=False,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n")
    print("=" * 120)
    print("ESTIMATED CLEAN CORPUS TOKEN BUDGET")
    print("=" * 120)

    print(
        results.to_string(
            index=False,
        )
    )

    print("\n")
    print(
        f"Estimated total tokens: "
        f"{total_tokens:,}"
    )

    print(
        f"Estimated total: "
        f"{total_tokens / 1_000_000_000:.2f}B tokens"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()