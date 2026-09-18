import re
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/sample_eda.parquet"
)

OUTPUT_PATH = Path(
    "data/interim/sample_eda_profiled.parquet"
)


URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+"
)

HTML_PATTERN = re.compile(
    r"<[^>]+>"
)

MENTION_PATTERN = re.compile(
    r"@\w+"
)

HASHTAG_PATTERN = re.compile(
    r"#\w+"
)


def extract_features(text):

    text = "" if text is None else str(text)

    words = text.split()

    n_chars = len(text)
    n_words = len(words)
    n_lines = len(text.splitlines())

    urls = URL_PATTERN.findall(text)
    mentions = MENTION_PATTERN.findall(text)
    hashtags = HASHTAG_PATTERN.findall(text)
    html = HTML_PATTERN.findall(text)

    digits = sum(
        char.isdigit()
        for char in text
    )

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    uppercase = sum(
        char.isupper()
        for char in letters
    )

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_lines": n_lines,

        "url_count": len(urls),
        "mention_count": len(mentions),
        "hashtag_count": len(hashtags),
        "html_tag_count": len(html),

        "digit_ratio":
            digits / max(n_chars, 1),

        "uppercase_ratio":
            uppercase / max(len(letters), 1),

        "empty":
            n_chars == 0,

        "very_short":
            n_words < 5,

        "very_long":
            n_words > 10_000,
    }


def main():

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Loaded {len(df):,} documents"
    )

    features = (
        df["text"]
        .apply(extract_features)
        .apply(pd.Series)
    )

    df = pd.concat(
        [df, features],
        axis=1,
    )

    # Versión normalizada SOLO para
    # detectar duplicados.
    # NO reemplaza el texto original.
    df["text_normalized"] = (
        df["text"]
        .str.lower()
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    df["duplicate_exact"] = (
        df["text_normalized"]
        .duplicated(keep=False)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved to {OUTPUT_PATH}"
    )

    print("\nDOCUMENTS")
    print(
        df["source"]
        .value_counts()
    )

    print("\nQUALITY BY SOURCE")

    summary = (
        df.groupby("source")
        .agg(
            documents=("text", "count"),

            median_words=(
                "n_words",
                "median",
            ),

            mean_words=(
                "n_words",
                "mean",
            ),

            pct_short=(
                "very_short",
                "mean",
            ),

            pct_urls=(
                "url_count",
                lambda x:
                    (x > 0).mean(),
            ),

            pct_mentions=(
                "mention_count",
                lambda x:
                    (x > 0).mean(),
            ),

            pct_html=(
                "html_tag_count",
                lambda x:
                    (x > 0).mean(),
            ),

            pct_duplicates=(
                "duplicate_exact",
                "mean",
            ),
        )
    )

    print(summary.to_string())


if __name__ == "__main__":
    main()