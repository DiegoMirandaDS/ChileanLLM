from pathlib import Path
import argparse
import math
import time

import bitsandbytes as bnb
import torch
import yaml

from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


CONFIG_PATH = Path(
    "configs/cpt_data.yaml"
)

TRAIN_PATH = Path(
    "data/processed/cpt_train"
)

CHECKPOINT_ROOT = Path(
    "checkpoints"
)


def collate_fn(rows):

    input_ids = torch.tensor(
        [
            row["input_ids"]
            for row in rows
        ],
        dtype=torch.long,
    )

    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )

    args = parser.parse_args()

    with open(
        CONFIG_PATH,
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    torch.manual_seed(
        config["seed"]
    )

    model_name = config["model"]["name"]

    training_config = (
        config["training"]
    )

    sequence_length = (
        training_config[
            "sequence_length"
        ]
    )

    batch_size = (
        training_config[
            "per_device_train_batch_size"
        ]
    )

    accumulation_steps = (
        training_config[
            "gradient_accumulation_steps"
        ]
    )

    learning_rate = (
        training_config[
            "learning_rate"
        ]
    )

    weight_decay = (
        training_config[
            "weight_decay"
        ]
    )

    max_runtime_seconds = (
        config["pilot"][
            "max_runtime_minutes"
        ]
        * 60
    )

    if args.smoke_test:

        target_tokens = (
            config["smoke_test"][
                "target_tokens"
            ]
        )

        output_dir = (
            CHECKPOINT_ROOT
            / "qwen3_chilean_smoke"
        )

        # El smoke test no necesita
        # durar 105 minutos.
        max_runtime_seconds = 30 * 60

    else:

        target_tokens = (
            config["pilot"][
                "max_train_tokens"
            ]
        )

        output_dir = (
            CHECKPOINT_ROOT
            / "qwen3_chilean_cpt"
        )

    microbatch_tokens = (
        sequence_length
        * batch_size
    )

    target_microbatches = math.ceil(
        target_tokens
        / microbatch_tokens
    )

    print("=" * 72)
    print("CONTINUAL PRE-TRAINING")
    print("=" * 72)

    print(f"Model: {model_name}")
    print(
        f"Target tokens: "
        f"{target_tokens:,}"
    )

    print(
        f"Sequence length: "
        f"{sequence_length}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print(
        f"Gradient accumulation: "
        f"{accumulation_steps}"
    )

    print(
        f"Effective tokens / step: "
        f"{microbatch_tokens * accumulation_steps:,}"
    )

    print(
        f"Maximum runtime: "
        f"{max_runtime_seconds / 60:.1f} min"
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = load_from_disk(
        str(TRAIN_PATH)
    )

    if target_microbatches > len(dataset):
        target_microbatches = len(
            dataset
        )
    
    target_microbatches -= (
        target_microbatches
        % accumulation_steps
    )

    dataset = dataset.select(
        range(target_microbatches)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading model...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
        .to("cuda")
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable()

    model.train()

    optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    total_optimizer_steps = (
        target_microbatches
        // accumulation_steps
    )

    warmup_steps = max(
        1,
        int(
            total_optimizer_steps
            * training_config["warmup_ratio"]
        ),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )


    optimizer.zero_grad(
        set_to_none=True
    )

    # ========================================================
    # TRAIN
    # ========================================================

    start_time = time.perf_counter()

    processed_tokens = 0
    processed_microbatches = 0
    optimizer_steps = 0
    accumulated_loss = 0.0

    progress = tqdm(
        dataloader,
        desc="Training",
    )

    for micro_step, batch in enumerate(
        progress,
        start=1,
    ):

        elapsed = (
            time.perf_counter()
            - start_time
        )

        if elapsed >= max_runtime_seconds:
            print(
                "\nMaximum runtime reached."
            )
            break

        input_ids = batch[
            "input_ids"
        ].to(
            "cuda",
            non_blocking=True,
        )

        labels = batch[
            "labels"
        ].to(
            "cuda",
            non_blocking=True,
        )

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
            )

            loss = (
                outputs.loss
                / accumulation_steps
            )

        loss.backward()

        accumulated_loss += (
            outputs.loss.item()
        )

        processed_tokens += (
            input_ids.numel()
        )

        processed_microbatches += 1

        should_step = (
            processed_microbatches
            % accumulation_steps
            == 0
        )

        if should_step:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        elapsed = (
            time.perf_counter()
            - start_time
        )

        tokens_per_second = (
            processed_tokens
            / max(elapsed, 1e-6)
        )

        progress.set_postfix(
            loss=f"{outputs.loss.item():.4f}",
            tokens=f"{processed_tokens:,}",
            tok_s=f"{tokens_per_second:.1f}",
        )

        if (
            processed_tokens
            >= target_tokens
        ):
            break


    elapsed = (
        time.perf_counter()
        - start_time
    )

    throughput = (
        processed_tokens
        / elapsed
    )

    mean_loss = (
        accumulated_loss
        / max(processed_microbatches, 1)
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"Processed tokens: "
        f"{processed_tokens:,}"
    )

    print(
        f"Optimizer steps: "
        f"{optimizer_steps:,}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Throughput: "
        f"{throughput:.1f} tokens/s"
    )

    print(
        f"Mean train loss: "
        f"{mean_loss:.4f}"
    )

    print(
        f"Peak GPU memory: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nSaving checkpoint to "
        f"{output_dir}"
    )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()