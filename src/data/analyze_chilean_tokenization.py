import re
import unicodedata
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"

INPUT_PATH = Path(
    "data/interim/sample_eda.parquet"
)

OUTPUT_PATH = Path(
    "reports/chilean_tokenization.csv"
)

SEED = 42


# ============================================================
# MARCADORES
# ============================================================

CHILEAN_MARKERS = {

    "discourse": [
        r"\bpo\b",
        r"\bsi po\b",
        r"\bno po\b",
        r"\bya po\b",
        r"\bpucha\b",
        r"\buta\b",
    ],

    "cachai_family": [
        r"\bcachai\b",
        r"\bcacha\b",
        r"\bcachaste\b",
    ],

    "voseo": [
        r"\bestai\b",
        r"\bteni\b",
        r"\bqueris\b",
        r"\bqueri\b",
        r"\bpodis\b",
        r"\bpodi\b",
        r"\bvenis\b",
        r"\bveni\b",
        r"\beris\b",
        r"\beri\b",
        r"\bsoi\b",
        r"\bvai\b",
    ],

    "slang": [
        r"\bbacan\b",
        r"\bfome\b",
        r"\bcuatico\b",
        r"\bpiola\b",
        r"\bpelmazo\b",
        r"\bcuico\b",
        r"\bflaite\b",
        r"\bcabro\b",
        r"\bcabra\b",
        r"\bpololo\b",
        r"\bpolola\b",
        r"\bpololear\b",
        r"\bcarrete\b",
        r"\bcarretear\b",
        r"\bpega\b",
        r"\bcopete\b",
        r"\bluca\b",
        r"\blucas\b",
    ],

    "weon_family": [
        r"\bweon\b",
        r"\bweona\b",
        r"\bhueon\b",
        r"\bhueona\b",
        r"\bhuevon\b",
        r"\bhuevona\b",
        r"\baweonao\b",
        r"\baweona\b",
        r"\bahueonao\b",
        r"\bahueona\b",
        r"\bwea\b",
        r"\bweas\b",
    ],

    "internet_chilean": [
        r"\bwn\b",
        r"\bwna\b",
        r"\bctm\b",
        r"\bculiao\b",
        r"\bculia\b",
        r"\bculia[oa]\b",
    ],

    "expressions": [
        r"\bal tiro\b",
        r"\bla raja\b",
        r"\bni cagando\b",
        r"\bmas encima\b",
    ],
}


ALL_MARKERS_RE = re.compile(
    "|".join(
        pattern
        for patterns in CHILEAN_MARKERS.values()
        for pattern in patterns
    ),
    flags=re.IGNORECASE,
)


# ============================================================
# ARTEFACTOS PROPIOS DE TWITTER
# ============================================================

URL_RE = re.compile(
    r"https?://\S+|www\.\S+"
)

MENTION_RE = re.compile(
    r"@\w+"
)

HASHTAG_RE = re.compile(
    r"#\w+"
)


# ============================================================
# FUNCIONES
# ============================================================

def normalize_for_matching(text):

    text = str(text).lower()

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    return "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )


def neutralize_twitter_artifacts(text):
    """
    Versión usada SOLO para análisis.

    No modifica el dataset original.

    Sustituimos elementos de Twitter por marcadores comunes
    para reducir su efecto en la comparación lingüística.
    """

    text = str(text)

    text = URL_RE.sub(
        "<URL>",
        text,
    )

    text = MENTION_RE.sub(
        "<USER>",
        text,
    )

    text = HASHTAG_RE.sub(
        "<HASHTAG>",
        text,
    )

    return text


def tokenizer_metrics(tokenizer, text):

    text = str(text)

    words = text.split()

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    return {
        "n_words": len(words),
        "n_tokens": len(token_ids),

        "tokens_per_word":
            len(token_ids)
            / max(len(words), 1),

        "tokens_per_char":
            len(token_ids)
            / max(len(text), 1),
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

    # ========================================================
    # SOLO TWITTER
    # ========================================================

    twitter = (
        df[df["source"] == "twitter"]
        .copy()
    )

    print(
        f"Twitter documents: {len(twitter):,}"
    )

    # ========================================================
    # DETECTAR MARCADORES
    # ========================================================

    twitter["text_match"] = (
        twitter["text"]
        .apply(normalize_for_matching)
    )

    twitter["has_chilean_marker"] = (
        twitter["text_match"]
        .apply(
            lambda x:
                bool(
                    ALL_MARKERS_RE.search(x)
                )
        )
    )

    n_marked = (
        twitter["has_chilean_marker"]
        .sum()
    )

    n_unmarked = (
        (~twitter["has_chilean_marker"])
        .sum()
    )

    print(
        f"With markers: {n_marked:,}"
    )

    print(
        f"Without markers: {n_unmarked:,}"
    )

    # ========================================================
    # BALANCEAR LOS DOS GRUPOS
    #
    # Esto evita comparar 1000 textos marcados contra
    # 19000 textos sin marcar.
    # ========================================================

    n = min(
        n_marked,
        n_unmarked,
    )

    marked = (
        twitter[
            twitter["has_chilean_marker"]
        ]
        .sample(
            n=n,
            random_state=SEED,
        )
        .copy()
    )

    unmarked = (
        twitter[
            ~twitter["has_chilean_marker"]
        ]
        .sample(
            n=n,
            random_state=SEED,
        )
        .copy()
    )

    marked["group"] = (
        "chilean_markers"
    )

    unmarked["group"] = (
        "no_markers"
    )

    comparison = pd.concat(
        [
            marked,
            unmarked,
        ],
        ignore_index=True,
    )

    # ========================================================
    # DOS VERSIONES
    #
    # raw:
    # texto original
    #
    # neutralized:
    # URLs/@usuarios/#hashtags sustituidos
    # ========================================================

    results = []

    for version in [
        "raw",
        "neutralized",
    ]:

        temp = comparison.copy()

        if version == "raw":

            temp["analysis_text"] = (
                temp["text"]
            )

        else:

            temp["analysis_text"] = (
                temp["text"]
                .apply(
                    neutralize_twitter_artifacts
                )
            )

        print(
            f"\nAnalyzing version: {version}"
        )

        metrics = (
            temp["analysis_text"]
            .apply(
                lambda text:
                    pd.Series(
                        tokenizer_metrics(
                            tokenizer,
                            text,
                        )
                    )
            )
        )

        temp = pd.concat(
            [
                temp,
                metrics,
            ],
            axis=1,
        )

        summary = (
            temp.groupby("group")
            .agg(
                documents=(
                    "text",
                    "count",
                ),

                median_words=(
                    "n_words",
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
            .reset_index()
        )

        summary["version"] = version

        results.append(summary)

    final = pd.concat(
        results,
        ignore_index=True,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n")
    print("=" * 100)
    print(
        "TWITTER: CHILEAN MARKERS VS NO MARKERS"
    )
    print("=" * 100)

    print(
        final.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()