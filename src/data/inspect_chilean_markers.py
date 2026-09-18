import re
import unicodedata

import pandas as pd


PATH = "data/interim/sample_eda.parquet"

N_EXAMPLES = 5
SEED = 42


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
#        r"\bde una\b", falsos positivos, infla demasiado
    ],
}


def normalize(text):

    text = str(text).lower()

    text = unicodedata.normalize(
        "NFD",
        text,
    )

    return "".join(
        c
        for c in text
        if unicodedata.category(c) != "Mn"
    )


def main():

    df = pd.read_parquet(PATH)

    df["text_match"] = df["text"].apply(
        normalize
    )

    for category, patterns in CHILEAN_MARKERS.items():

        pattern = re.compile(
            "|".join(patterns),
            flags=re.IGNORECASE,
        )

        matches = df[
            df["text_match"].str.contains(
                pattern,
                regex=True,
            )
        ]

        print("\n")
        print("=" * 100)
        print(category.upper())
        print("=" * 100)

        print(
            f"Matches: {len(matches):,}"
        )

        if len(matches) == 0:
            continue

        sample = matches.sample(
            n=min(
                N_EXAMPLES,
                len(matches),
            ),
            random_state=SEED,
        )

        for _, row in sample.iterrows():

            print(
                f"\n[{row['source']}]"
            )

            print(
                str(row["text"])[:700]
            )


if __name__ == "__main__":
    main()