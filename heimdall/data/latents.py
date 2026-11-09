r"""Dataset of the latents: windows of GLORYS states with the surface observations of their central days."""

__all__ = [
    "LatentDataset",
    "get_latent_datasets",
]

import numpy as np
import torch
import xarray as xr

from collections.abc import Sequence
from torch import Tensor
from torch.utils.data import Dataset

from heimdall.config import PATH_LATENT_DATASETS
from heimdall.data import OBSERVATIONS


class LatentDataset(Dataset):
    r"""Creates a dataset of windows of latent GLORYS states, with the latent observations of their block of days.

    Arguments:
        split        : Split of the latents, 'training' or 'validation'.
        block        : Number of consecutive days to predict, B in GiBBS.
        blanket      : Number of known states on each side of the block, C in GiBBS.
        observations : Observation products given with the block, among OBSERVATIONS.
    """

    def __init__(
        self,
        split: str = "training",
        block: int = 3,
        blanket: int = 1,
        observations: Sequence[str] = OBSERVATIONS,
    ) -> None:
        super().__init__()

        # Security
        assert split in ("training", "validation"), (
            f"ERROR - split must be 'training' or 'validation', got {split}."
        )
        assert block > 0, f"ERROR - block must be positive, got {block}."
        assert blanket >= 0, f"ERROR - blanket must be non-negative, got {blanket}."
        for modality in observations:
            assert modality in OBSERVATIONS, (
                f"ERROR - {modality} is not an observation product, see OBSERVATIONS."
            )

        self.split = split
        self.block = block
        self.blanket = blanket
        self.observations = list(observations)
        self.stores = None

        # The mean and std of each channel come from the training split, the same for every split
        self.statistics = {
            modality: xr.open_zarr(
                PATH_LATENT_DATASETS[modality] / "statistics.zarr", chunks=None
            ).latents.values.astype(np.float32)
            for modality in ["glorys", *self.observations]
        }

        # Every day of GLORYS is a state, while an observation product misses some days (gaps, SSS before 2010)
        self.dates = self.days("glorys")
        self.indices = {
            modality: {d: i for i, d in enumerate(self.days(modality))}
            for modality in self.observations
        }

        # A window needs consecutive days, GLORYS having no missing day in a split
        days = np.array(self.dates, dtype="datetime64[D]")
        size = self.block + 2 * self.blanket
        consecutive = np.diff(days) == np.timedelta64(1, "D")
        self.starts = [
            i for i in range(len(days) - size + 1) if consecutive[i : i + size - 1].all()
        ]

        # Security
        assert len(self.starts) > 0, (
            f"ERROR - No window of {size} consecutive days in the {split} latents."
        )

    def days(self, modality: str) -> list[str]:
        r"""Read the dates of the latents of a modality in the split.

        Arguments:
            modality : Name of the modality, a key of PATH_LATENT_DATASETS.

        Returns:
            dates : Date strings 'YYYY-MM-DD' of the encoded days.
        """

        store = xr.open_zarr(PATH_LATENT_DATASETS[modality] / f"{self.split}.zarr", chunks=None)

        # Security
        assert store.encoded.all(), (
            f"ERROR - Days of the {self.split} latents of {modality} are not encoded yet."
        )

        return [str(t)[:10] for t in store.time.values]

    def __getstate__(self) -> dict:
        r"""Return the state of the dataset sent to a worker, without the opened stores."""
        return self.__dict__ | {"stores": None}

    def __len__(self) -> int:
        r"""Return the number of windows of the split."""
        return len(self.starts)

    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor], list[str]]:
        r"""Return a window of standardized latents: the states of all its days, the observations of its block.

        Arguments:
            idx : Index of the window.

        Returns:
            x         : States of the window (C + B + C, C_z, Z', Y', X'), the block in the middle.
            y         : Observations of the days of the block, (B, C_z, Y', X') per product, 0 on a missing day.
            available : Days of the block observed by each product, boolean (B,) per product.
            dates     : Date strings 'YYYY-MM-DD' of the window (C + B + C,).
        """

        # Each worker opens the stores at its first sample
        if self.stores is None:
            self.stores = {
                modality: xr.open_zarr(
                    PATH_LATENT_DATASETS[modality] / f"{self.split}.zarr", chunks=None
                ).latents
                for modality in ["glorys", *self.observations]
            }

        start = self.starts[idx]
        dates = self.dates[start : start + self.block + 2 * self.blanket]
        x = self.standardize("glorys", self.stores["glorys"][start : start + len(dates)].values)

        y, available = {}, {}
        for modality in self.observations:
            days = [
                self.indices[modality].get(d)
                for d in dates[self.blanket : self.blanket + self.block]
            ]
            shape = self.stores[modality].shape[1:]
            fields = [
                np.zeros(shape, dtype=np.float32) if i is None else self.stores[modality][i].values
                for i in days
            ]
            y[modality] = torch.where(
                torch.tensor([i is not None for i in days]).view(-1, *[1] * len(shape)),
                self.standardize(modality, np.stack(fields)),
                0.0,
            )
            available[modality] = torch.tensor([i is not None for i in days])

        return x, y, available, dates

    def standardize(self, modality: str, z: np.ndarray) -> Tensor:
        r"""Standardize latents with the mean and std of each channel over the training split.

        Arguments:
            modality : Name of the modality, a key of PATH_LATENT_DATASETS.
            z        : Latents (T, C_z, ...) in float16.

        Returns:
            z : Standardized latents (T, C_z, ...) in float32.
        """

        mean, std = self.statistics[modality]
        shape = (1, -1) + (1,) * (z.ndim - 2)

        return torch.from_numpy((z.astype(np.float32) - mean.reshape(shape)) / std.reshape(shape))


def get_latent_datasets(
    block: int = 3,
    blanket: int = 1,
    observations: Sequence[str] = OBSERVATIONS,
) -> tuple[LatentDataset, LatentDataset]:
    r"""Create the training and validation datasets of the latents.

    Arguments:
        block        : Number of consecutive days to predict, B in GiBBS.
        blanket      : Number of known states on each side of the block, C in GiBBS.
        observations : Observation products given with the block, among OBSERVATIONS.

    Returns:
        train : Training dataset
        val   : Validation dataset
    """

    kwargs: dict = {"block": block, "blanket": blanket, "observations": observations}

    return LatentDataset("training", **kwargs), LatentDataset("validation", **kwargs)
