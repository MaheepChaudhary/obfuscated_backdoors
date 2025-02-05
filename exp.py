import warnings

import cupbearer as cup
import torch
from datasets import load_dataset
from fire import Fire
from transformers import logging as transformers_logging

from src.backdoors import train_backdoor
from src.backdoors_obfuscation import *
from src.encoders import DeepmindSparseAutoencoder, EleutherSparseAutoencoder



import time
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cupbearer as cup
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import wandb

from src.probe_training import initialize_lora_adapter
from src.utils import *


def train_backdoor(
    encoder,
    obfuscation_loss_fns: Dict[Callable, float],
    ds_normal_benign: Dataset,
    ds_normal_harmful: Optional[Dataset],
    ds_backdoor: Dataset,
    steps_per_harmful_sample: int = 1,
    steps_per_backdoor_sample: int = 1,
    activation_matching_layers: List[int] = [4, 8, 12, 16, 20, 24],
    loss_coefs={
        "backdoored": 1.0,
        "kl_change": 1.0,
        "activation_change": 1.0,
    },
    lora_params={},
    model_lr=5e-5,
    n_steps=1000,
    n_steps_per_logging=50,
    batch_size=16,
    n_grad_accum=4,
    device="cuda",
    clip_grad_norm=1.0,
    obfuscate_over="last_token_prompt",
    n_loggings_per_eval=10,
    n_evals_per_mahalanobis=2,
    n_eval=512,
    eval_inference_batch_size=64,
    eval_training_batch_size=32,
    # These are for logging purposes:
    model_type=None,
    dataset_name=None,
    backdoor_type="lora",
    wandb_project=None,
    mahalanobis_shrinkage=0.1,
    eval_mahalanobis_on_harmful=False,
    eval_mahalanobis_on_both=False,
    wandb_run_name=None,
    push_checkpoints_to_hub_every_n_steps=None,
    eval_backdoor_during_training=True,
):

    lora_model = initialize_lora_adapter(
        encoder, [encoder.model.config.num_hidden_layers], lora_params
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=model_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    # process the datasets
    print("Processing datasets:")

    def split(ds, n_eval):
        if ds is None:
            return None, None
        ds_split = ds.train_test_split(test_size=n_eval, shuffle=False)
        return ds_split["train"], ds_split["test"]

    ds_backdoor, ds_backdoor_eval = split(ds_backdoor, n_eval)
    ds_normal_benign, ds_normal_benign_eval = split(ds_normal_benign, n_eval)
    ds_normal_harmful, ds_normal_harmful_eval = split(ds_normal_harmful, n_eval)

    assert ds_backdoor is not None
    assert ds_backdoor_eval is not None
    assert ds_normal_benign is not None
    assert ds_normal_benign_eval is not None

    ds_backdoor_eval.rename_column("completion", "desired_completion")
    ds_normal_benign_eval.rename_column("completion", "desired_completion")
    if ds_normal_harmful_eval is not None:
        ds_normal_harmful_eval.rename_column("completion", "desired_completion")

    # Try to load existing data loaders if they exist
    save_dir = Path(__file__).resolve().parent / "temp/"
    if dataset_name is None:
        save_prefix = f"{batch_size}"
    else:
        save_prefix = f"{dataset_name.split('/')[-1]}-{batch_size}"
    try:
        dataloaders = [
            torch.load(save_dir / (save_prefix + f"-loader_{name}.pt"))
            for name in ["backdoored", "normal_benign"]
        ]
        if ds_normal_harmful is not None:
            dataloaders.append(
                torch.load(save_dir / (save_prefix + "-loader_normal_harmful.pt"))
            )
        print(f"Loaded dataloaders from disk at {save_dir / save_prefix}...")
    except FileNotFoundError:
        # TODO: this currently loads all of the datasets into memory, because e.g.
        #       ds_normal_benign["prompt"] is a list of all the prompts. This is slow.
        #       Use .map() to prepare the dataloaders instead.
        datasets = [
            ds_backdoor["prompt"],
            ds_backdoor["completion"],
            ds_normal_benign["prompt"],
            ds_normal_benign["completion"],
        ]
        if ds_normal_harmful is not None:
            datasets.extend(
                [ds_normal_harmful["prompt"], ds_normal_harmful["completion"]]
            )
        # Prepare dataloaders
        dataloaders = prepare_dataloaders(encoder.tokenizer, batch_size, *datasets)

        # Save the dataloaders
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dataloaders[0], save_dir / (save_prefix + "-loader_backdoored.pt"))
        torch.save(
            dataloaders[1], save_dir / (save_prefix + "-loader_normal_benign.pt")
        )
        if ds_normal_harmful is not None:
            torch.save(
                dataloaders[2], save_dir / (save_prefix + "-loader_normal_harmful.pt")
            )

    backdoored_dataloader = iter(dataloaders[0])
    normal_benign_dataloader = iter(dataloaders[1])
    normal_harmful_dataloader = (
        iter(dataloaders[2]) if ds_normal_harmful is not None else None
    )

    print("Training the backdoor:")
    # Set model to training mode
    lora_model.train()

    # Initialize loss tracking
    total_losses = {loss_name: 0 for loss_name in loss_coefs.keys()}
    total_losses.update(
        {
            obfuscation_loss_fn.__name__: 0
            for obfuscation_loss_fn in obfuscation_loss_fns.keys()
        }
    )
    total_losses["total"] = 0
    eval_dict = {}
    total_steps = 0
    n_loggings = 0
    steps_since_last_logging = 0
    keep_running_training = True
    wandb_run = None
    if wandb_project is not None:
        obfuscation_method = "_".join(
            [fn.__name__ for fn in obfuscation_loss_fns.keys()]
        )
        obfuscation_method = obfuscation_method.replace("compute_", "")
        obfuscation_method = obfuscation_method.replace("obfuscation_", "")
        obfuscation_method = obfuscation_method.replace("_loss", "")
        print(f"obfuscation_method = {obfuscation_method}")
        wandb_run = wandb.init(
            # Set the project where this run will be logged
            project=wandb_project,
            # Set the name of the run
            name=wandb_run_name,
            # Track hyperparameters and run metadata
            config={
                "obfuscation_method": obfuscation_method,
                "loss_coefs": loss_coefs,
                "model_lr": model_lr,
                "steps_per_harmful_sample": steps_per_harmful_sample,
                "steps_per_backdoor_sample": steps_per_backdoor_sample,
                "n_steps": n_steps,
                "n_steps_per_logging": n_steps_per_logging,
                "batch_size": batch_size,
                "n_grad_accum": n_grad_accum,
                "device": device,
                "clip_grad_norm": clip_grad_norm,
                "activation_matching_layers": activation_matching_layers,
                "obfuscate_over": obfuscate_over,
                "lora_params": lora_params,
                "model_type": model_type,
                "dataset_name": dataset_name,
                "backdoor_type": backdoor_type,
                "n_eval": n_eval,
                "eval_inference_batch_size": eval_inference_batch_size,
                "eval_training_batch_size": eval_training_batch_size,
                "n_evals_per_mahalanobis": n_evals_per_mahalanobis,
                "n_loggings_per_eval": n_loggings_per_eval,
                "len_normal_benign": len(ds_normal_benign),
                "len_normal_harmful": (
                    len(ds_normal_harmful) if ds_normal_harmful is not None else 0
                ),
                "len_backdoor": len(ds_backdoor),
                "wandb_project": wandb_project,
                "mahalanobis_shrinkage": mahalanobis_shrinkage,
                "eval_mahalanobis_on_harmful": eval_mahalanobis_on_harmful,
                "eval_mahalanobis_on_both": eval_mahalanobis_on_both,
            },
        )
        print(f"wandb_run.id = {wandb_run.id}")

    t_forward = 0.0
    t_backward = 0.0
    print("Training backdoor model...")

    pbar = tqdm(total=n_steps, desc="Training")
    while keep_running_training:

        # Load the next backdoored batch if we want one
        if total_steps % steps_per_backdoor_sample != 0:
            backdoored_batch = None
        else:
            backdoored_batch = next(backdoored_dataloader, None)
            if backdoored_batch is None:
                backdoored_dataloader = iter(dataloaders[0])
                backdoored_batch = next(backdoored_dataloader)

        # Load the next normal benign batch
        normal_benign_batch = next(normal_benign_dataloader, None)
        if normal_benign_batch is None:
            normal_benign_dataloader = iter(dataloaders[1])
            normal_benign_batch = next(normal_benign_dataloader)

        # Load the next normal harmful batch if we want one
        if (
            total_steps % steps_per_harmful_sample != 0
            or normal_harmful_dataloader is None
        ):
            normal_harmful_batch = None
        else:
            normal_harmful_batch = next(normal_harmful_dataloader, None)
            if normal_harmful_batch is None:
                normal_harmful_dataloader = iter(dataloaders[2])
                normal_harmful_batch = next(normal_harmful_dataloader)

        t0 = time.time()
        total_steps += 1

        # Process a single training step
        losses = process_step(
            lora_model,
            encoder.tokenizer,
            obfuscation_loss_fns,
            normal_benign_batch,
            normal_harmful_batch,
            backdoored_batch,
            activation_matching_layers,
            loss_coefs,
            obfuscate_over,
            n_grad_accum,
            total_steps,
        )
        t1 = time.time()

        for key in total_losses:
            if key in losses:
                total_losses[key] += losses[key]
        steps_since_last_logging += 1

        # Perform optimization step
        if total_steps % n_grad_accum == 0:
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        t2 = time.time()

        t_forward += t1 - t0
        t_backward += t2 - t1
        # Log progress
        # total_steps - 1 so that we run the first logging step immediately
        # (and get a baseline very close to initialization).
        # This also catches any errors in the eval loop more quickly.
        # Also log the final step even if we don't hit the logging frequency.
        if (total_steps - 1) % n_steps_per_logging == 0 or total_steps >= n_steps:
            if n_loggings % n_loggings_per_eval == 0:
                mahalanobis_step = (
                    n_loggings % (n_loggings_per_eval * n_evals_per_mahalanobis)
                ) == 0

                # Validation metrics
                if eval_backdoor_during_training:
                    eval_dict = evaluate_backdoor(
                        lora_model,
                        encoder.tokenizer,
                        ds_normal_benign_eval,
                        ds_normal_harmful_eval,
                        ds_backdoor_eval,
                        activation_matching_layers,
                        device,
                        ds_normal_benign,
                        ds_normal_harmful,
                        inference_batch_size=eval_inference_batch_size,
                        training_batch_size=eval_training_batch_size,
                        mahalanobis=mahalanobis_step,
                        mahalanobis_on_harmful=eval_mahalanobis_on_harmful
                        and mahalanobis_step,
                        mahalanobis_on_both=eval_mahalanobis_on_both
                        and mahalanobis_step,
                        mahalanobis_shrinkage=mahalanobis_shrinkage,
                    )
                    for k, v in eval_dict.items():
                        if isinstance(v, torch.Tensor) and len(v.shape) == 0:
                            print(f"{k}: {v.item()}")
                        if isinstance(v, float):
                            print(f"{k}: {v}")
                else:
                    eval_dict = {}

            avg_losses = {
                k: v / steps_since_last_logging for k, v in total_losses.items()
            }

            print(
                f"Step {total_steps}/{n_steps} | "
                + " | ".join(
                    f"{loss_name.capitalize()} Loss: {loss_value:.4f}"
                    for loss_name, loss_value in avg_losses.items()
                )
            )

            # Log to wandb
            if wandb_project is not None:
                wandb.log(
                    {
                        **{f"loss/{k}": v for k, v in avg_losses.items()},
                        **{
                            "progress": total_steps / n_steps,
                            "System/time_per_step_forward": t_forward
                            / steps_since_last_logging,
                            "System/time_per_step_backward": t_backward
                            / steps_since_last_logging,
                        },
                        **eval_dict,
                    },
                    step=total_steps,
                )
                print(f"Logged to wandb at step {total_steps}.")

        # Push checkpoints to the hub
        if push_checkpoints_to_hub_every_n_steps is not None:
            if (
                total_steps > 1
                and (total_steps - 1) % push_checkpoints_to_hub_every_n_steps == 0
                or total_steps >= n_steps
            ):
                if dataset_name is None:
                    print("No dataset name provided, skipping push to hub.")
                else:
                    wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)
                    lora_model.push_to_hub(
                        f"{dataset_name.replace('-dataset','')}-model{wandb_run_id}-step{total_steps}"
                    )

            total_losses = {k: 0 for k in total_losses}
            t_forward = 0.0
            t_backward = 0.0
            steps_since_last_logging = 0
            n_loggings += 1

        # Update progress bar
        pbar.update(1)

        # Check if training should stop
        if total_steps >= n_steps:
            keep_running_training = False
            break

    # Finish wandb run
    if wandb_project:
        wandb.finish()

    return lora_model, wandb_run



def main(
    MODEL_TYPE="llama3",
    BACKDOOR_TYPE="lora",
    ACTIVATION_MATCHING_LAYERS=[4, 8, 12, 16, 20, 24, 28],
    N_STEPS=3000,
    N_STEPS_PER_LOGGING=10,
    BATCH_SIZE=2,
    N_GRAD_ACCUM=4,
    MODEL_LR=1e-4,
    CLIP_GRAD_NORM=1.0,
    DEVICE="cuda",
    N_LOGGINGS_PER_EVAL=512,
    N_EVALS_PER_MAHALANOBIS=1,
    EVAL_MAHALANOBIS_ON_HARMFUL=True,
    EVAL_MAHALANOBIS_ON_BOTH=True,
    N_EVAL=512,
    EVAL_INFERENCE_BATCH_SIZE=128,
    EVAL_TRAINING_BATCH_SIZE=32,
    MAHALANOBIS_SHRINKAGE=0.05,
    DATASET_SUFFIX="llama3-deployment-backdoor",
    PUSH_TO_HUB=True,
    BACKDOORED_LOSS=3.0,
    ACTIVATION_CHANGE_LOSS=6.0,
    KL_CHANGE_LOSS=3.0,
):

    # Suppress specific warnings
    warnings.filterwarnings(
        "ignore", message="Setting `pad_token_id` to `eos_token_id`.*"
    )

    # Or suppress all Transformers warnings
    transformers_logging.set_verbosity_error()

    # Loss coefficients
    loss_coefs = {}
    if BACKDOORED_LOSS != 0:  # Cross entropy on backdoored completion
        loss_coefs["backdoored"] = BACKDOORED_LOSS
    if ACTIVATION_CHANGE_LOSS != 0:  # Match activations on normal data
        loss_coefs["activation_change"] = ACTIVATION_CHANGE_LOSS
    if KL_CHANGE_LOSS != 0:  # KL divergence difference
        loss_coefs["kl_change"] = KL_CHANGE_LOSS

    # Load the appropriate model and dataset
    model_type = MODEL_TYPE
    dataset_name = f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-dataset"
    print(f"\nDataset namedataset_name = {dataset_name}\n")
    WANDB_RUN_NAME = (
        DATASET_SUFFIX.split("-")[1]
        + "_"
        + "_".join(
            [f"{k[:3].strip('_')}={v}" for k, v in loss_coefs.items() if v != 0.0]
        )
    )

    # Load the appropriate model
    if model_type == "llama3":
        encoder = EleutherSparseAutoencoder.load_llama3_sae(None, instruct=True)
    elif model_type == "gemma2":
        encoder = DeepmindSparseAutoencoder.load_gemma2_sae(None, 11)
    else:
        raise ValueError("Unsupported model type")

    # Load the dataset
    dataset = load_dataset(dataset_name)

    obfuscation_loss_fns = {
        construct_vae_obfuscation_loss_fn(
            layers=ACTIVATION_MATCHING_LAYERS,
            tokenizer=encoder.tokenizer,
            obfuscate_over="generation",
        ): 1.0,
        construct_mahalanobis_obfuscation_loss_fn(
            layers=ACTIVATION_MATCHING_LAYERS,
            tokenizer=encoder.tokenizer,
            obfuscate_over="last_token_prompt",
            shrinkage=MAHALANOBIS_SHRINKAGE,
        ): 1.0,
    }

    print("Training backdoored model")

    # Train the backdoor
    lora_model, wandb_run = train_backdoor(
        encoder,
        obfuscation_loss_fns,
        dataset["normal_benign_train"],
        dataset["normal_harmful_train"],
        dataset["backdoored_train"],
        activation_matching_layers=ACTIVATION_MATCHING_LAYERS,
        loss_coefs=loss_coefs,
        lora_params={},
        model_lr=MODEL_LR,
        n_steps=N_STEPS,
        n_steps_per_logging=N_STEPS_PER_LOGGING,
        batch_size=BATCH_SIZE,
        n_grad_accum=N_GRAD_ACCUM,
        device=DEVICE,
        clip_grad_norm=CLIP_GRAD_NORM,
        model_type=model_type,
        dataset_name=dataset_name,
        backdoor_type=BACKDOOR_TYPE,
        wandb_project="mad-backdoors",
        n_loggings_per_eval=N_LOGGINGS_PER_EVAL,
        n_eval=N_EVAL,
        eval_inference_batch_size=EVAL_INFERENCE_BATCH_SIZE,
        eval_training_batch_size=EVAL_TRAINING_BATCH_SIZE,
        n_evals_per_mahalanobis=N_EVALS_PER_MAHALANOBIS,
        eval_mahalanobis_on_harmful=EVAL_MAHALANOBIS_ON_HARMFUL,
        eval_mahalanobis_on_both=EVAL_MAHALANOBIS_ON_BOTH,
        mahalanobis_shrinkage=MAHALANOBIS_SHRINKAGE,
        wandb_run_name=WANDB_RUN_NAME,
        eval_backdoor_during_training=False,
    )

    wandb_run_id = "" if wandb_run is None else "-" + str(wandb_run.id)

    if PUSH_TO_HUB:
        lora_model.push_to_hub(
            f"Mechanistic-Anomaly-Detection/{DATASET_SUFFIX}-model{wandb_run_id}"
        )
    else:
        lora_model.save_pretrained(f"models/{DATASET_SUFFIX}-model{wandb_run_id}")


# def print_kwargs_then_run_main(**kwargs):
#     for key, value in kwargs.items():
#         print(f"{key} = {value}")
#     main(**kwargs)


# if __name__ == "__main__":
#     Fire(print_kwargs_then_run_main)

if __name__ == "__main__":
    main()
