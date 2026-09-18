import re
import unicodedata
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/sample_eda_profiled.parquet"
)

REPORT_DIR = Path("reports")

SUMMARY_OUTPUT = REPORT_DIR / "quality_audit.csv"
MARKERS_OUTPUT = REPORT_DIR / "chilean_markers_by_source.csv"


# ============================================================
# PII / DATOS POTENCIALMENTE SENSIBLES
# ============================================================

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

RUT_RE = re.compile(
    r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"
)

PHONE_RE = re.compile(
    r"(?:\+?56\s?)?(?:9\s?)?\d{4}\s?\d{4}"
)


# ============================================================
# MARCADORES EXPLORATORIOS DE ESPAÑOL CHILENO
#
# IMPORTANTE: 
# - Se usan como señales exploratorias.
# - Palabras como "tenis", que también se puede asociar al deporte, son falsos 
# positivos que surgen de la ambigüedad del lenguaje. Por eso esta metodología
# es simplemente exploratoria y no definitiva.
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
#       r"\btenis\b", contamina con falsos positivos asociados al deporte
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
#        r"\bde una\b", de una infla demasiado y puede generar falsos positivos.
    ],
}


COMPILED_MARKERS = {
    category: re.compile(
        "|".join(patterns),
        flags=re.IGNORECASE,
    )
    for category, patterns in CHILEAN_MARKERS.items()
}


# ============================================================
# UTILIDADES
# ============================================================

def normalize_for_matching(text: str) -> str:
    """
    Normalización SOLO para detección de marcadores.

    Ejemplo:
        "Tenís que venir, po"
    pasa temporalmente a:
        "tenis que venir, po"

    El texto original NO se modifica.
    """

    text = "" if text is None else str(text)

    text = text.lower()

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    return text


def contains(pattern, text):
    text = "" if text is None else str(text)
    return bool(pattern.search(text))


def count_matches(pattern, text):
    return len(pattern.findall(text))


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_parquet(INPUT_PATH)

    print(
        f"Loaded {len(df):,} documents"
    )

    # --------------------------------------------------------
    # PII
    # --------------------------------------------------------

    df["has_email"] = df["text"].apply(
        lambda x: contains(EMAIL_RE, x)
    )

    df["has_rut"] = df["text"].apply(
        lambda x: contains(RUT_RE, x)
    )

    df["has_phone"] = df["text"].apply(
        lambda x: contains(PHONE_RE, x)
    )

    # --------------------------------------------------------
    # Texto normalizado SOLO para matching lingüístico
    # --------------------------------------------------------

    df["text_match"] = df["text"].apply(
        normalize_for_matching
    )

    # --------------------------------------------------------
    # Marcadores por categoría
    # --------------------------------------------------------

    marker_columns = []

    for category, pattern in COMPILED_MARKERS.items():

        has_column = f"has_{category}"
        count_column = f"count_{category}"

        df[has_column] = df["text_match"].apply(
            lambda x, p=pattern: bool(p.search(x))
        )

        df[count_column] = df["text_match"].apply(
            lambda x, p=pattern: len(p.findall(x))
        )

        marker_columns.append(has_column)

    # ¿Tiene al menos una categoría?
    df["has_any_chilean_marker"] = (
        df[marker_columns]
        .any(axis=1)
    )

    # Número total de coincidencias
    count_columns = [
        f"count_{category}"
        for category in CHILEAN_MARKERS
    ]

    df["total_chilean_marker_hits"] = (
        df[count_columns]
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # Auditoría general
    # --------------------------------------------------------

    summary = (
        df.groupby("source")
        .agg(
            documents=(
                "text",
                "count",
            ),

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

            pct_url=(
                "url_count",
                lambda x: (x > 0).mean(),
            ),

            pct_mentions=(
                "mention_count",
                lambda x: (x > 0).mean(),
            ),

            pct_html=(
                "html_tag_count",
                lambda x: (x > 0).mean(),
            ),

            pct_duplicate=(
                "duplicate_exact",
                "mean",
            ),

            pct_email=(
                "has_email",
                "mean",
            ),

            pct_rut=(
                "has_rut",
                "mean",
            ),

            pct_phone=(
                "has_phone",
                "mean",
            ),

            pct_any_chilean_marker=(
                "has_any_chilean_marker",
                "mean",
            ),

            avg_chilean_marker_hits=(
                "total_chilean_marker_hits",
                "mean",
            ),
        )
    )

    summary.to_csv(
        SUMMARY_OUTPUT
    )

    # --------------------------------------------------------
    # Tabla específica por categoría lingüística
    # --------------------------------------------------------

    marker_rows = []

    for source in sorted(
        df["source"].unique()
    ):

        source_df = df[
            df["source"] == source
        ]

        total_docs = len(source_df)
        total_words = source_df["n_words"].sum()

        for category in CHILEAN_MARKERS:

            has_column = f"has_{category}"
            count_column = f"count_{category}"

            docs_with_marker = (
                source_df[has_column]
                .sum()
            )

            total_hits = (
                source_df[count_column]
                .sum()
            )

            marker_rows.append(
                {
                    "source": source,
                    "category": category,

                    "documents":
                        total_docs,

                    "docs_with_marker":
                        int(docs_with_marker),

                    "pct_docs_with_marker":
                        docs_with_marker
                        / max(total_docs, 1),

                    "total_hits":
                        int(total_hits),

                    "hits_per_10k_words":
                        total_hits
                        / max(total_words, 1)
                        * 10_000,
                }
            )

    markers_df = pd.DataFrame(
        marker_rows
    )

    markers_df.to_csv(
        MARKERS_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Output terminal
    # --------------------------------------------------------

    print("\n")
    print("=" * 110)
    print("GENERAL DATASET AUDIT")
    print("=" * 110)

    print(
        summary.to_string()
    )

    print("\n")
    print("=" * 110)
    print("CHILEAN MARKERS BY SOURCE")
    print("=" * 110)

    pivot = markers_df.pivot(
        index="source",
        columns="category",
        values="pct_docs_with_marker",
    )

    print(
        pivot.to_string()
    )

    print("\n")
    print("=" * 110)
    print("MARKER FREQUENCY PER 10K WORDS")
    print("=" * 110)

    frequency_pivot = markers_df.pivot(
        index="source",
        columns="category",
        values="hits_per_10k_words",
    )

    print(
        frequency_pivot.to_string()
    )

    print(
        f"\nSaved general audit to:"
        f" {SUMMARY_OUTPUT}"
    )

    print(
        f"Saved marker analysis to:"
        f" {MARKERS_OUTPUT}"
    )


if __name__ == "__main__":
    main()