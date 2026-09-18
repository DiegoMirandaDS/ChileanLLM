import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from ftfy import fix_encoding


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "data/interim/sample_eda.parquet"
)

OUTPUT_PATH = Path(
    "data/processed/sample_clean.parquet"
)

REPORT_PATH = Path(
    "reports/cleaning_sample_summary.csv"
)


# ============================================================
# PATTERNS
# ============================================================

URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    flags=re.IGNORECASE,
)

MENTION_RE = re.compile(
    r"(?<!\w)@[A-Za-z0-9_]{1,30}\b"
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

RUT_RE = re.compile(
    r"(?<!\d)"
    r"(?:\d{1,2}\.?\d{3}\.?\d{3}|\d{7,8})"
    r"-[0-9kK]"
    r"(?!\w)"
)

# Conservador:
# detectamos móviles chilenos que comienzan en 9.
# Evitamos tratar cualquier secuencia de 8 dígitos como teléfono.
PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?:\+?56)[\s.\-]*)?"
    r"9[\s.\-]*"
    r"\d{4}[\s.\-]*"
    r"\d{4}"
    r"(?!\d)"
)

HTML_TAG_RE = re.compile(
    r"<[^>\n]{1,500}>"
)

SPACES_RE = re.compile(
    r"[ \t\f\v]+"
)

MULTIPLE_NEWLINES_RE = re.compile(
    r"\n{3,}"
)

PLACEHOLDER_RE = re.compile(
    r"<(?:URL|USER|EMAIL|RUT|PHONE)>"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Conservamos tweets cortos deliberadamente.
MIN_WORDS = {
    "twitter": 1,
    "mc4": 5,
    "news": 5,
    "complaints": 5,
}


# ============================================================
# FUNCIONES
# ============================================================

def normalize_unicode(text: str) -> str:
    """
    Repara mojibake sin aplicar otras transformaciones
    tipográficas innecesarias.

    Ejemplo:
        PiÃ±era -> Piñera
    """

    text = fix_encoding(text)

    return unicodedata.normalize(
        "NFC",
        text,
    )


def normalize_whitespace(text: str) -> str:

    # Espacios horizontales repetidos
    text = SPACES_RE.sub(
        " ",
        text,
    )

    # Quitamos espacios alrededor de saltos
    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    # Máximo dos saltos consecutivos
    text = MULTIPLE_NEWLINES_RE.sub(
        "\n\n",
        text,
    )

    return text.strip()


def meaningful_word_count(text: str) -> int:
    """
    Cuenta palabras ignorando placeholders.

    Así un texto como:
        <USER> <URL>
    no cuenta como contenido útil.
    """

    text = PLACEHOLDER_RE.sub(
        " ",
        text,
    )

    words = re.findall(
        r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b",
        text,
        flags=re.UNICODE,
    )

    return len(words)


def dedup_key(text: str) -> str:
    """
    Normalización usada SOLAMENTE para deduplicación.

    El texto final conserva mayúsculas, tildes, etc.
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def clean_document(text, source):

    original = "" if text is None else str(text)

    # --------------------------------------------------------
    # 1. Encoding / Unicode
    # --------------------------------------------------------

    fixed = normalize_unicode(
        original
    )

    ftfy_changed = (
        fixed != original
    )

    # --------------------------------------------------------
    # 2. HTML entities
    # --------------------------------------------------------

    fixed = html.unescape(
        fixed
    )

    # --------------------------------------------------------
    # 3. HTML
    # --------------------------------------------------------

    had_html = bool(
        HTML_TAG_RE.search(fixed)
    )

    fixed = HTML_TAG_RE.sub(
        " ",
        fixed,
    )

    # --------------------------------------------------------
    # 4. Detectar PII antes de reemplazar
    # --------------------------------------------------------

    had_email = bool(
        EMAIL_RE.search(fixed)
    )

    had_rut = bool(
        RUT_RE.search(fixed)
    )

    had_phone = bool(
        PHONE_RE.search(fixed)
    )

    had_url = bool(
        URL_RE.search(fixed)
    )

    had_mention = bool(
        MENTION_RE.search(fixed)
    )

    # --------------------------------------------------------
    # 5. PII
    # --------------------------------------------------------

    fixed = EMAIL_RE.sub(
        "<EMAIL>",
        fixed,
    )

    fixed = RUT_RE.sub(
        "<RUT>",
        fixed,
    )

    fixed = PHONE_RE.sub(
        "<PHONE>",
        fixed,
    )

    # --------------------------------------------------------
    # 6. Internet
    # --------------------------------------------------------

    fixed = URL_RE.sub(
        "<URL>",
        fixed,
    )

    # Aplicamos esto a todas las fuentes porque mC4 y news
    # también pueden contener tweets incrustados.
    fixed = MENTION_RE.sub(
        "<USER>",
        fixed,
    )

    # IMPORTANTE:
    # hashtags NO se eliminan.

    # --------------------------------------------------------
    # 7. Whitespace
    # --------------------------------------------------------

    fixed = normalize_whitespace(
        fixed
    )

    # --------------------------------------------------------
    # 8. Quality filter
    # --------------------------------------------------------

    words = meaningful_word_count(
        fixed
    )

    min_words = MIN_WORDS.get(
        source,
        5,
    )

    valid = (
        bool(fixed)
        and words >= min_words
    )

    return {
        "clean_text": fixed,

        "meaningful_words":
            words,

        "valid":
            valid,

        "ftfy_changed":
            ftfy_changed,

        "had_html":
            had_html,

        "had_email":
            had_email,

        "had_rut":
            had_rut,

        "had_phone":
            had_phone,

        "had_url":
            had_url,

        "had_mention":
            had_mention,
    }


# ============================================================
# MAIN
# ============================================================
def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    df = pd.read_parquet(
        INPUT_PATH
    )

    print(
        f"Loaded {len(df):,} documents"
    )

    # ========================================================
    # CLEAN
    # ========================================================

    print("\nCleaning documents...")

    cleaned = df.apply(
        lambda row: pd.Series(
            clean_document(
                row["text"],
                row["source"],
            )
        ),
        axis=1,
    )

    # Mantenemos TODAS las filas juntas inicialmente.
    full_df = pd.concat(
        [
            df,
            cleaned,
        ],
        axis=1,
    )

    # ========================================================
    # INVALID DOCUMENTS
    # ========================================================

    invalid_count = (
        ~full_df["valid"]
    ).sum()

    print(
        f"Invalid documents: {invalid_count:,}"
    )

    # Trabajamos solamente con válidos a partir de aquí.
    valid_df = full_df[
        full_df["valid"]
    ].copy()

    # ========================================================
    # DEDUPLICACIÓN
    # ========================================================

    valid_df["dedup_key"] = (
        valid_df["clean_text"]
        .apply(dedup_key)
    )

    # Marcamos duplicados ANTES de eliminarlos,
    # así después podemos reportarlos por fuente.
    valid_df["is_duplicate"] = (
        valid_df["dedup_key"]
        .duplicated(
            keep="first"
        )
    )

    duplicates_removed = (
        valid_df["is_duplicate"]
        .sum()
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed:,}"
    )

    final_df = valid_df[
        ~valid_df["is_duplicate"]
    ].copy()

    # ========================================================
    # REPORTE POR FUENTE
    # ========================================================

    report_rows = []

    for source in sorted(
        full_df["source"].unique()
    ):

        # Todos los documentos originales de esta fuente
        source_all = full_df[
            full_df["source"] == source
        ]

        # Todos los válidos antes de deduplicar
        source_valid = valid_df[
            valid_df["source"] == source
        ]

        # Los que finalmente quedan
        source_final = final_df[
            final_df["source"] == source
        ]

        original_n = len(
            source_all
        )

        invalid_n = int(
            (~source_all["valid"]).sum()
        )

        duplicate_n = int(
            source_valid[
                "is_duplicate"
            ].sum()
        )

        kept_n = len(
            source_final
        )

        report_rows.append(
            {
                "source":
                    source,

                "original_documents":
                    original_n,

                "invalid_removed":
                    invalid_n,

                "duplicates_removed":
                    duplicate_n,

                "kept_documents":
                    kept_n,

                "retention_rate":
                    kept_n
                    / max(original_n, 1),

                "pct_encoding_repaired":
                    source_all[
                        "ftfy_changed"
                    ].mean(),

                "pct_html_removed":
                    source_all[
                        "had_html"
                    ].mean(),

                "pct_url_replaced":
                    source_all[
                        "had_url"
                    ].mean(),

                "pct_user_replaced":
                    source_all[
                        "had_mention"
                    ].mean(),

                "pct_email_redacted":
                    source_all[
                        "had_email"
                    ].mean(),

                "pct_rut_redacted":
                    source_all[
                        "had_rut"
                    ].mean(),

                "pct_phone_redacted":
                    source_all[
                        "had_phone"
                    ].mean(),
            }
        )

    report = pd.DataFrame(
        report_rows
    )

    report.to_csv(
        REPORT_PATH,
        index=False,
    )

    # ========================================================
    # DATASET FINAL
    # ========================================================

    output = (
        final_df[
            [
                "source",
                "clean_text",
            ]
        ]
        .rename(
            columns={
                "clean_text": "text"
            }
        )
        .reset_index(
            drop=True
        )
    )

    output.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    original_total = len(
        full_df
    )

    final_total = len(
        output
    )

    print("\n")
    print("=" * 110)
    print("CLEANING SUMMARY")
    print("=" * 110)

    print(
        report.to_string(
            index=False
        )
    )

    print("\n")

    print(
        f"Original: "
        f"{original_total:,}"
    )

    print(
        f"Invalid removed: "
        f"{invalid_count:,}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Final: "
        f"{final_total:,}"
    )

    print(
        f"Retention: "
        f"{final_total / original_total:.2%}"
    )

    print(
        f"\nClean dataset saved to:"
        f" {OUTPUT_PATH}"
    )

    print(
        f"Report saved to:"
        f" {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()