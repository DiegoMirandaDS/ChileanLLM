from pathlib import Path

import pandas as pd
from ftfy import fix_encoding, fix_text


INPUT_PATH = Path(
    "data/interim/sample_eda.parquet"
)

N_EXAMPLES = 8
SEED = 42


def main():

    df = pd.read_parquet(
        INPUT_PATH
    )

    df["fix_text_version"] = (
        df["text"].apply(fix_text)
    )

    df["fix_encoding_version"] = (
        df["text"].apply(fix_encoding)
    )

    df["changed_fix_text"] = (
        df["text"]
        != df["fix_text_version"]
    )

    df["changed_fix_encoding"] = (
        df["text"]
        != df["fix_encoding_version"]
    )

    print("\n")
    print("=" * 100)
    print("CHANGE RATE")
    print("=" * 100)

    summary = (
        df.groupby("source")
        .agg(
            documents=(
                "text",
                "count",
            ),

            pct_fix_text=(
                "changed_fix_text",
                "mean",
            ),

            pct_fix_encoding=(
                "changed_fix_encoding",
                "mean",
            ),
        )
    )

    print(
        summary.to_string()
    )

    # --------------------------------------------------------
    # Ejemplos donde fix_encoding realmente modifica algo
    # --------------------------------------------------------

    for source in sorted(
        df["source"].unique()
    ):

        changed = df[
            (df["source"] == source)
            &
            (df["changed_fix_encoding"])
        ]

        print("\n")
        print("=" * 100)
        print(
            f"{source.upper()} "
            f"- ENCODING REPAIRS"
        )
        print("=" * 100)

        print(
            f"Changed documents: "
            f"{len(changed):,}"
        )

        if len(changed) == 0:
            continue

        sample = changed.sample(
            n=min(
                N_EXAMPLES,
                len(changed),
            ),
            random_state=SEED,
        )

        for _, row in sample.iterrows():

            print("\n--- ORIGINAL ---")
            print(
                str(row["text"])[:800]
            )

            print("\n--- REPAIRED ---")
            print(
                str(
                    row["fix_encoding_version"]
                )[:800]
            )


if __name__ == "__main__":
    main()