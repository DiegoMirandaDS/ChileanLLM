from collections import defaultdict
from pathlib import Path

import argparse
import json
import math

import torch
import yaml

from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM


CONFIG_PATH = Path("configs/cpt_data.yaml")

EVAL_PATH = Path(
    "data/evaluation/cpt_eval"
)

OUTPUT_DIR = Path(
    "reports/evaluation"
)


def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Hugging Face model name or local checkpoint. "
            "Defaults to model.name from config."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path where evaluation JSON will be saved."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # CONFIG
    # ========================================================

    with open(
        CONFIG_PATH,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    model_name = (
        args.model
        if args.model is not None
        else config["model"]["name"]
    )

    sequence_length = (
        config["training"]["sequence_length"]
    )

    print("=" * 72)
    print("LANGUAGE MODEL EVALUATION")
    print("=" * 72)

    print(f"Model: {model_name}")

    # ========================================================
    # DATASET
    # ========================================================

    print("\nLoading evaluation dataset...")

    dataset = load_from_disk(
        str(EVAL_PATH)
    )

    print(
        f"Evaluation blocks: {len(dataset):,}"
    )

    print(
        f"Evaluation tokens: "
        f"{len(dataset) * sequence_length:,}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading model...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            dtype=torch.float16,
        )
        .to("cuda")
    )

    model.eval()

    # ========================================================
    # EVALUATION
    # ========================================================

    losses = []

    source_losses = defaultdict(list)

    progress = tqdm(
        dataset,
        desc="Evaluating",
    )

    with torch.inference_mode():

        for row in progress:

            input_ids = torch.tensor(
                row["input_ids"],
                dtype=torch.long,
                device="cuda",
            ).unsqueeze(0)

            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                use_cache=False,
            )

            loss = float(
                outputs.loss.item()
            )

            losses.append(
                loss
            )

            source_losses[
                row["source"]
            ].append(
                loss
            )

    # ========================================================
    # GLOBAL METRICS
    # ========================================================

    global_loss = (
        sum(losses)
        / len(losses)
    )

    global_ppl = math.exp(
        global_loss
    )

    results = {
        "model": model_name,
        "blocks": len(dataset),
        "tokens": (
            len(dataset)
            * sequence_length
        ),
        "loss": global_loss,
        "perplexity": global_ppl,
        "by_source": {},
    }

    # ========================================================
    # SOURCE METRICS
    # ========================================================

    for source, values in sorted(
        source_losses.items()
    ):

        source_loss = (
            sum(values)
            / len(values)
        )

        source_ppl = math.exp(
            source_loss
        )

        results["by_source"][
            source
        ] = {
            "blocks": len(values),
            "loss": source_loss,
            "perplexity": source_ppl,
        }

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 72)
    print("EVALUATION RESULTS")
    print("=" * 72)

    print(
        f"\nLoss:       "
        f"{global_loss:.4f}"
    )

    print(
        f"Perplexity: "
        f"{global_ppl:.4f}"
    )

    print("\nBy source:")

    for source, metrics in (
        results["by_source"].items()
    ):

        print(
            f"{source:12s} | "
            f"loss {metrics['loss']:.4f} | "
            f"ppl {metrics['perplexity']:.4f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    if args.output is not None:

        output_path = Path(
            args.output
        )

    else:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            OUTPUT_DIR
            / "baseline_qwen3.json"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()