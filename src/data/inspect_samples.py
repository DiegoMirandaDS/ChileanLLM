import re

import pandas as pd


PATH = "data/interim/sample_eda_profiled.parquet"

SEED = 42
N = 8


EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

RUT_RE = re.compile(
    r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"
)

PHONE_RE = re.compile(
    r"(?:\+?56\s?)?(?:9\s?)?\d{4}\s?\d{4}"
)


def redact(text):

    text = str(text)

    text = EMAIL_RE.sub(
        "[EMAIL]",
        text,
    )

    text = RUT_RE.sub(
        "[RUT]",
        text,
    )

    text = PHONE_RE.sub(
        "[PHONE]",
        text,
    )

    return text


def main():

    df = pd.read_parquet(PATH)

    for source in sorted(
        df["source"].unique()
    ):

        print("\n")
        print("=" * 100)
        print(source.upper())
        print("=" * 100)

        source_df = df[
            df["source"] == source
        ]

        sample = source_df.sample(
            n=min(N, len(source_df)),
            random_state=SEED,
        )

        for _, row in sample.iterrows():

            print("\n---")

            print(
                f"words={row['n_words']}"
                f" | urls={row['url_count']}"
                f" | mentions={row['mention_count']}"
                f" | html={row['html_tag_count']}"
            )

            print()

            print(
                redact(
                    row["text"]
                )[:1200]
            )


if __name__ == "__main__":
    main()