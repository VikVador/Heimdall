r"""Launch the training of the deterministic encoder of a modality, preserving the geometry of its fields."""

import argparse
import math
import torch
import torch.distributed as dist
import wandb

from omegaconf import OmegaConf
from shaggy.loss import loss_geometry_embedding
from shaggy.optimizers.gradients import safe_gradient_step
from shaggy.optimizers.soap import SOAP
from shaggy.tools import load_config, load_weights, save
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from heimdall.config import PATH_MODELS
from heimdall.data.dataloader import get_dataloaders
from heimdall.data.weights import get_weights_change, get_weights_date, get_weights_stats
from heimdall.distributed import reduce_mean, setup_distributed
from heimdall.model.encoders import Encoder
from heimdall.schedulers import warmup_cosine_decay
from heimdall.tools import generate_run_name, get_wandb_hyperparameters, load_configuration


# fmt: off
#
def _build_model(
    config_model    : dict,
    checkpoint_name : str | None,
    device          : torch.device,
) -> tuple[Encoder, dict]:
    r"""Build the encoder of a modality from scratch, or resume it from a checkpoint.

    Arguments:
        config_model    : Constructor kwargs of the encoder (modality, lat_channels, ...), used when training from scratch.
        checkpoint_name : Name of the checkpoint to resume from (if applicable).
        device          : Target device.

    Returns:
        model  : Encoder in training mode, on the target device.
        config : Constructor kwargs, with the domain of the encoder (checked when loading it).
    """

    if checkpoint_name is not None:
        ckpt_path = PATH_MODELS / checkpoint_name
        config    = OmegaConf.to_container(load_config(ckpt_path))
        model     = load_weights(Encoder(**config), ckpt_path, device=str(device)).train()
    else:
        model  = Encoder(**config_model).to(device)
        config = {**config_model, "domain": model.domain}

    return model, config


def forward_loss(
    model      : torch.nn.Module,
    batch      : tuple,
    modality   : str,
    coarsening : int,
    scale      : Tensor,
    device     : torch.device,
) -> Tensor:
    r"""Encode the fields of a batch, with their date, and compute the GME loss between the fields and their latents.

    Arguments:
        model      : Encoder, possibly wrapped by DDP.
        batch      : Single day of the modality, with its date, from HeimdallDataset.
        modality   : Name of the modality, a key of DATASET_VARIABLES.
        coarsening : Number of points averaged along the latitude and along the longitude.
        scale      : Ratio of the std of each variable to the std of its change from one day to the next.
        device     : Target device.

    Returns:
        loss : GME loss of the encoder.
    """

    x, _, dates = batch
    x           = x[:, 0].to(device)

    # The encoder is conditioned on the date (transposed by the collate), the masks being left out of the inputs
    date = torch.cat([get_weights_date(d, modality, coarsening, dim=2, device=device) for d in dates[0]]).to(x.dtype)

    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        z = model(torch.cat([x, date], dim=1))

    # The loss only compares the geometry of the fields (distances in float32), the conditioning being an input,
    # measured in typical changes from one day to the next so that every variable weighs in the distances
    return loss_geometry_embedding(x.float() * scale, z.float())


def training(
    config_state    : dict,
    config_training : dict,
    config_model    : dict,
    config_wandb    : dict,
) -> None:
    r"""Launch the training of the encoder of a modality.

    Arguments:
        config_state    : Saving option and checkpoint to resume from.
        config_training : Batch sizes, optimizer steps and learning rate schedule.
        config_model    : Modality, coarsening, latent channels and architecture of the encoder.
        config_wandb    : Weights & Biases entity, project and mode.
    """

    # RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT are set by the sbatch script
    rank, local_rank, world_size, device, is_distributed = setup_distributed()

    modality = config_model["modality"]

    (
        saving,
        checkpoint_name,
        steps_update,
        steps_logging,
        steps_validation,
        batches_validation,
        batch_size_per_step,
        batch_size_per_gpu,
        num_workers,
        prefetch_factor,
        lr_start,
        lr_peak,
        lr_end,
        warmup_steps,
    ) = (
        config_state["saving"],
        config_state["checkpoint_name"],
        config_training["steps_update"],
        config_training["steps_logging"],
        config_training["steps_validation"],
        config_training["batches_validation"],
        config_training["batch_size_per_step"],
        config_training["batch_size_per_gpu"],
        config_training["num_workers"],
        config_training["prefetch_factor"],
        config_training["learning_rate_start"],
        config_training["learning_rate_peak"],
        config_training["learning_rate_end"],
        config_training["warmup_steps"],
    )

    # Security
    assert batch_size_per_gpu > 1, f"ERROR - The GME loss pairs the samples of a GPU, batch_size_per_gpu must be > 1, got {batch_size_per_gpu}."

    # Loading the model from a checkpoint, or building it from scratch
    model, config = _build_model(config_model, checkpoint_name, device)
    coarsening    = model.coarsening

    # A variable whose std is mostly geographical (e.g. the temperature) changes little in standard deviations
    _, std = get_weights_stats(modality, dim=2, device=device)
    scale  = std / get_weights_change(modality, dim=2, device=device)

    # The Weights & Biases run is named after the architecture
    run_name = generate_run_name(f"ENC_{modality}_lc{model.lat_channels}_tk{model.tokens}", checkpoint_name)

    if rank == 0:
        wandb.init(
            **config_wandb,
            name=run_name,
            config={
                "State"           : config_state,
                "Training"        : config_training,
                "Model"           : config,
                "Hyperparameters" : get_wandb_hyperparameters({**config_training, **config}),
            },
        )
        wandb.log({
            "Informations/Trainable Parameters [M]" : sum(p.numel() for p in model.parameters()) / 1e6,
            "Informations/Tokens"                   : model.tokens,
        })
    else:
        wandb.init(mode="disabled")

    # Number of steps to accumulate gradients before updating model parameters
    steps_gradient_accumulation = max(1, math.ceil(batch_size_per_step / (batch_size_per_gpu * world_size)))
    batches = [
        steps_update * steps_gradient_accumulation,
        steps_update // steps_validation * batches_validation,
        None,
    ]

    # Each sample is a single day of the modality
    dataloader_training, dataloader_validation, _ = get_dataloaders(
        batch_size      = batch_size_per_gpu,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        batches         = batches,
        shuffle         = [True, True, False],
        infinite        = [True, True, False],
        rank            = rank,
        world_size      = world_size,
        is_distributed  = is_distributed,
        modality        = modality,
        window          = 1,
        coarsening      = coarsening,
    )

    # Masks and meshes are identical on every process, there is no need to broadcast them
    if is_distributed:
        ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
        model      = DDP(model, broadcast_buffers=False, gradient_as_bucket_view=True, **ddp_kwargs)

    # Setting up training tools
    optimizer = SOAP(
        model.parameters(),
        lr=lr_peak,
        max_precond_size=1024,
    )

    scheduler = warmup_cosine_decay(
        optimizer    = optimizer,
        lr_start     = lr_start,
        lr_peak      = lr_peak,
        lr_end       = lr_end,
        warmup_steps = warmup_steps,
        total_steps  = steps_update,
    )

    loss_accumulator         = 0.0
    loss_logging_accumulator = 0.0
    loss_validation_best     = float("inf")
    gradient_norm            = float("inf")
    optimizer_step           = 0

    # Waiting for processes to be ready
    if is_distributed:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    for step, batch in enumerate(dataloader_training):

        # Forward pass and loss
        loss = forward_loss(model, batch, modality, coarsening, scale, device)

        # Gradient accumulation
        loss              = loss / steps_gradient_accumulation
        loss_accumulator += loss.item()

        # Only sync gradients on last accumulation step
        is_last_accumulation_step = ((step + 1) % steps_gradient_accumulation == 0)

        if is_distributed and not is_last_accumulation_step:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()

        # Cleaning up memory
        del batch, loss

        if not is_last_accumulation_step:
            continue

        # Optimization step
        gradient_norm             = safe_gradient_step(optimizer=optimizer, grad_clip=1.0)
        loss_step                 = loss_accumulator
        loss_logging_accumulator += loss_accumulator
        loss_accumulator          = 0.0
        optimizer_step           += 1
        scheduler.step()

        # Logging to console if not using WandB
        if config_wandb["mode"] == "disabled":
            print(f"Step {optimizer_step:6d} | Loss: {loss_step:.4f} | γ: {scheduler.get_last_lr()[0]:.6f} | ∇: {gradient_norm:.4f}")

        # Logging results, averaged over the logging window and the processes
        if optimizer_step % steps_logging == 0:
            loss_mean                = loss_logging_accumulator / steps_logging
            loss_logging_accumulator = 0.0

            if is_distributed:
                loss_mean = reduce_mean(loss_mean, device)

            if rank == 0:
                wandb.log({
                    "Training/Loss"              : loss_mean,
                    "Informations/Steps Update"  : optimizer_step,
                    "Informations/Samples Seen"  : (step + 1) * batch_size_per_gpu * world_size,
                    "Informations/Gradient Norm" : gradient_norm,
                    "Informations/Learning Rate" : scheduler.get_last_lr()[0],
                })

        # Validation, and saving the best model
        if optimizer_step % steps_validation == 0:
            model.eval()
            loss_validation = 0.0
            with torch.no_grad():
                for _ in range(batches_validation):
                    loss_validation += forward_loss(model, next(dataloader_validation), modality, coarsening, scale, device).item() / batches_validation
            model.train()

            if is_distributed:
                loss_validation = reduce_mean(loss_validation, device)

            if rank == 0:
                wandb.log({
                    "Validation/Loss"           : loss_validation,
                    "Informations/Steps Update" : optimizer_step,
                })

                if saving and loss_validation < loss_validation_best:
                    raw_model            = model.module if hasattr(model, "module") else model
                    loss_validation_best = loss_validation
                    save(raw_model, config, PATH_MODELS / run_name)

    # Closing run
    wandb.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the training of the encoder of a modality.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the .yml configuration of the encoder, e.g. configs/encoder/encoder-sst.yml.",
    )

    # Launched by cluster/jobs/encoder-*.sbatch, srun starting one process per GPU
    args   = parser.parse_args()
    config = load_configuration(args.config)[0]

    training(
        config_state    = config["State"],
        config_training = config["Training"],
        config_model    = config["Model"],
        config_wandb    = config["WandB"],
    )
