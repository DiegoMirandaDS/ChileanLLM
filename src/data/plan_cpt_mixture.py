from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path(
    "configs/cpt_data.yaml"
)

OUTPUT_PATH = Path(
    "reports/cpt_mixture_plan.csv"
)


def main():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    total_tokens = config[
        "training_corpus"
    ]["target_tokens"]

    mixture = config[
        "training_corpus"
    ]["mixture"]

    sources = config["sources"]

    rows = []

    for source, fraction in mixture.items():

        target_tokens = (
            total_tokens
            * fraction
        )

        mean_tokens = (
            sources[source]
            ["estimated_tokens_per_document"]
        )

        estimated_documents = (
            target_tokens
            / mean_tokens
        )

        rows.append(
            {
                "source":
                    source,

                "mixture_fraction":
                    fraction,

                "target_tokens":
                    round(target_tokens),

                "target_tokens_millions":
                    target_tokens
                    / 1_000_000,

                "mean_tokens_per_document":
                    mean_tokens,

                "estimated_documents":
                    round(
                        estimated_documents
                    ),
            }
        )

    df = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n")
    print("=" * 100)
    print("CPT MIXTURE PLAN")
    print("=" * 100)

    print(
        df.to_string(
            index=False
        )
    )

    print("\n")
    print(
        f"Target: "
        f"{total_tokens / 1_000_000:.0f}M tokens"
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()