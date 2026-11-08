r"""Encode every day of a modality into its latent space with its trained encoder, and the statistics of the latents."""

import argparse
import dask.array
import hashlib
import numpy as np
import torch
import xarray as xr
import zarr

from omegaconf import OmegaConf
from pathlib import Path
from shaggy.tools import load_config, load_weights
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from heimdall.config import PATH_LATENTS, PATH_MODELS
from heimdall.data import DATASET_DATES_TEST, DATASET_DATES_TRAINING, DATASET_DATES_VALIDATION
from heimdall.data.dataset import HeimdallDataset, get_excluded_days
from heimdall.data.tools import moments
from heimdall.model.encoders import Encoder
from heimdall.tools import load_configuration

# fmt: off
#
# ----- Splits of the datasets
#
SPLITS = {
    "training"   : DATASET_DATES_TRAINING,
    "validation" : DATASET_DATES_VALIDATION,
    "test"       : DATASET_DATES_TEST,
}


def create_store(
    path  : Path,
    dates : list[str],
    shape : tuple[int, ...],
    dims  : list[str],
    attrs : dict[str, str],
) -> None:
    r"""Create the latent store of a split, no day being encoded yet, or check that an existing one can be resumed.

    Arguments:
        path  : Path of the latent zarr store.
        dates : Date strings 'YYYY-MM-DD' of the days of the split.
        shape : Shape of the latents of a day (C_z, [Z',] Y', X').
        dims  : Names of the dimensions of the latents, starting with 'time'.
        attrs : Modality, model and fingerprint of the weights of the encoder.
    """

    time = np.array(dates, dtype="datetime64[ns]")

    # A store is only resumed with the same encoder and the same days, the latents of two encoders never being mixed
    if path.exists():
        store = xr.open_zarr(path, chunks=None)
        valid = store.attrs == attrs and np.array_equal(store.time.values, time)
        assert valid, f"ERROR - {path} holds the latents of {store.attrs} on other days or with other weights, delete it to encode again."
        return

    # Only the metadata, the dates and the flags are written, the days not encoded yet reading as NaN
    latents = dask.array.full((len(dates), *shape), np.nan, dtype=np.float16, chunks=(1, *shape))
    xr.Dataset(
        {
            "latents" : (dims, latents),
            "encoded" : ("time", np.zeros(len(dates), dtype=bool)),
        },
        coords = {"time": time},
        attrs  = attrs,
    ).to_zarr(
        path,
        mode     = "w",
        compute  = False,
        encoding = {"latents": {"chunks": (1, *shape), "fill_value": np.nan}},
    )


def encode(
    model           : Encoder,
    dataset         : HeimdallDataset,
    path            : Path,
    batch_size      : int,
    num_workers     : int,
    prefetch_factor : int,
) -> None:
    r"""Encode the days of a split not encoded yet, in order, writing their latents one batch at a time.

    Arguments:
        model           : Encoder of the modality, in evaluation mode.
        dataset         : Days of the split, one per sample.
        path            : Path of the latent zarr store, created by create_store.
        batch_size      : Number of days encoded at once.
        num_workers     : Number of worker processes loading the days.
        prefetch_factor : Number of batches prefetched per worker.
    """

    device = next(model.parameters()).device
    store  = zarr.open_group(path, mode="r+")

    # The flags are stored as integers by xarray, 1 for a day already encoded
    remaining = np.flatnonzero(store["encoded"][:] == 0)
    if len(remaining) == 0:
        return

    # Workers are spawned, since the connections to OSN (s3fs) hang in forked processes
    dataloader = DataLoader(
        Subset(dataset, remaining.tolist()),
        batch_size              = batch_size,
        shuffle                 = False,
        drop_last               = False,
        num_workers             = num_workers,
        prefetch_factor         = prefetch_factor if num_workers > 0 else None,
        pin_memory              = True,
        multiprocessing_context = "spawn" if num_workers > 0 else None,
    )

    batches = tqdm(dataloader, desc=f"{dataset.modality} {path.stem}", mininterval=60)

    for start, (x, _, dates) in zip(range(0, len(remaining), batch_size), batches, strict=True):
        days = remaining[start : start + len(x)]

        # Security
        assert list(dates[0]) == [dataset.dates[i] for i in days], f"ERROR - The batch of {dates[0][0]} is not in the order of the store."

        # The network runs in bfloat16 as in its training, float16 holding its outputs exactly (up to 65,504)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            z = model(x[:, 0].to(device, non_blocking=True)).half().cpu().numpy()

        # Security
        assert np.isfinite(z).all(), f"ERROR - Latents out of the range of float16 (or NaN) from {dates[0][0]}."

        # The days are flagged once their latents are written, a stopped job encoding the others when resumed
        store["latents"].oindex[days] = z
        store["encoded"].oindex[days] = True


def compute_statistics(latents: xr.DataArray, days: int = 32) -> np.ndarray:
    r"""Compute the mean and standard deviation of each latent channel, over every day and every point.

    Arguments:
        latents : Latents (T, C_z, [Z',] Y', X'), read a few days at a time.
        days    : Number of days read at once.

    Returns:
        statistics : Mean and standard deviation of each channel (2, C_z).
    """

    sums = 0.0
    for start in range(0, len(latents), days):
        z     = torch.from_numpy(latents[start : start + days].values)
        sums += moments(z).transpose(1, 2).flatten(2).sum(dim=-1)

    count, total, squares = sums.numpy()
    mean = total / count

    # A channel left unused by the encoder is constant, its tiny std keeping the standardization finite
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-16))

    return np.stack([mean, std]).astype(np.float32)


def save_statistics(path: Path) -> None:
    r"""Save the mean and standard deviation of each latent channel over the training split, once fully encoded.

    Arguments:
        path : Folder of the latents of the encoder, holding the store of the training split.
    """

    store = xr.open_zarr(path / "training.zarr", chunks=None)

    # Security
    assert store.encoded.all(), f"ERROR - Days of {path / 'training.zarr'} are not encoded yet."

    xr.Dataset(
        {"latents": (["statistic", "channel"], compute_statistics(store.latents))},
        coords = {"statistic": ["mean", "std"]},
        attrs  = store.attrs,
    ).to_zarr(path / "statistics.zarr", mode="w")


def encode_modality(
    modality        : str,
    run             : str,
    checkpoint      : str,
    splits          : list[str],
    batch_size      : int,
    num_workers     : int,
    prefetch_factor : int,
) -> None:
    r"""Encode the splits of a modality with its trained encoder, and save the statistics of the training latents.

    Arguments:
        modality        : Name of the modality, a key of DATASET_VARIABLES.
        run             : Folder of the trained encoder in PATH_MODELS.
        checkpoint      : Checkpoint of the run, 'best-validation', 'best-training' or 'last'.
        splits          : Splits to encode, among 'training', 'validation' and 'test'.
        batch_size      : Number of days encoded at once.
        num_workers     : Number of worker processes loading the days.
        prefetch_factor : Number of batches prefetched per worker.
    """

    # Security
    for split in splits:
        assert split in SPLITS, f"ERROR - split must be one of {list(SPLITS)}, got {split}."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path   = PATH_MODELS / run / checkpoint
    model  = load_weights(Encoder(**OmegaConf.to_container(load_config(path))), path, device=str(device)).eval()

    # Security
    assert model.modality == modality, f"ERROR - {run} encodes {model.modality}, not {modality}."

    # The fingerprint of the weights tells a resumed encoding that the checkpoint was not replaced meanwhile
    attrs = {
        "modality" : modality,
        "model"    : f"{run}/{checkpoint}",
        "weights"  : hashlib.sha256((path / "model.pth").read_bytes()).hexdigest()[:16],
    }
    shape = (model.lat_channels, *model.shape)
    dims  = ["time", "channel", "level", "y", "x"] if len(model.shape) == 3 else ["time", "channel", "y", "x"]

    # The latents are written next to the run and checkpoint of their encoder, e.g. PATH_LATENTS/ENC_sst_.../best-validation
    path_latents = PATH_LATENTS / run / checkpoint

    for split in splits:
        dataset = HeimdallDataset(*SPLITS[split], modality, coarsening=model.coarsening, excluded_days=get_excluded_days(modality))
        create_store(path_latents / f"{split}.zarr", dataset.dates, shape, dims, attrs)
        encode(model, dataset, path_latents / f"{split}.zarr", batch_size, num_workers, prefetch_factor)

    # Only the training split defines the statistics used to standardize the latents
    if "training" in splits:
        save_statistics(path_latents)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Encode every day of the splits of a modality into its latent space.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the .yml configuration of the encoding, e.g. configs/latents.yml.",
    )
    parser.add_argument(
        "--modality",
        "-m",
        type=str,
        required=True,
        help="Modality to encode, a section of the configuration, e.g. sst.",
    )

    # Launched by cluster/jobs/latents.sbatch, one task per modality
    args   = parser.parse_args()
    config = load_configuration(args.config)[0][args.modality]

    if config["run"] is None:
        print(f"No trained encoder given for {args.modality} in {args.config}, nothing to encode.")
    else:
        encode_modality(args.modality, **config)
