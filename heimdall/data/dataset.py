r"""Dataset."""

__all__ = [
    "HeimdallDataset",
    "get_datasets",
]

import torch
import xarray as xr

from torch import Tensor
from torch.utils.data import Dataset

from heimdall.config import PATH_GLORYS, PATH_OBSERVATIONS
from heimdall.data import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_OBSERVATIONS,
    DATASET_REGION,
    DATASET_VARIABLES_GLORYS,
    MODALITIES,
    VARIABLES_CLIPPING,
)
from heimdall.data.tools import assert_date_format
from heimdall.data.weights import get_weights_mask, get_weights_stats


class HeimdallDataset(Dataset):
    r"""Creates a dataset of windows of consecutive days of a modality, the GLORYS states or an observation product.

    Arguments:
        date_start   : Start date of the date range (format: 'YYYY-MM-DD').
        date_end     : End date of the date range (format: 'YYYY-MM-DD').
        modality     : Name of the modality, a key of MODALITIES.
        window       : Number of consecutive days of each sample.
        standardized : If True, standardize each variable using statistics.
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        modality: str = "glorys",
        window: int = 1,
        standardized: bool = True,
    ) -> None:
        super().__init__()

        # Security
        assert_date_format(date_start)
        assert_date_format(date_end)
        valid = modality in MODALITIES
        assert valid, f"ERROR - modality must be one of {MODALITIES}, got {modality}."

        self.modality = modality
        self.window = window
        self.standardized = standardized

        # Modality | Store, variables and mask, 3D for GLORYS and 2D at the surface for the observations
        glorys = modality == "glorys"
        mask_surface, mask_ocean = get_weights_mask()
        self.path = PATH_GLORYS if glorys else PATH_OBSERVATIONS[modality]
        self.variables = DATASET_VARIABLES_GLORYS if glorys else DATASET_OBSERVATIONS[modality]
        self.mask = mask_ocean if glorys else mask_surface
        self.mean, self.std = get_weights_stats(modality)

        # TODO | Name of the time coordinate of the stores
        with xr.open_zarr(self.path) as ds:
            dates = ds.time.dt.strftime("%Y-%m-%d").values

        self.dates = sorted(date for date in dates if date_start <= date <= date_end)

        # Security
        if len(self) <= 0:
            raise ValueError(
                f"ERROR - Not enough dates ({len(self.dates)}) for a window of {window} days."
            )

    def __len__(self) -> int:
        r"""Return the number of windows of consecutive days in the date range."""
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, list[str]]:
        r"""Return a window of consecutive days, with the points where the modality is observed.

        Arguments:
            idx : Index of the window.

        Returns:
            x        : States or observations (T, C, Z, Y, X) for GLORYS or (T, C, Y, X) for an observation product.
            observed : Points observed on the sea, 1 where x is valid and 0 elsewhere, same shape.
            dates    : Date strings 'YYYY-MM-DD' of the window (T,).
        """

        dates = self.dates[idx : idx + self.window]
        fields = [self.preprocess(date) for date in dates]

        return torch.stack([x for x, _ in fields]), torch.stack([o for _, o in fields]), dates

    def preprocess(self, date: str) -> tuple[Tensor, Tensor]:
        r"""Load and preprocess the modality on a single day.

        Arguments:
            date : Date string 'YYYY-MM-DD'.

        Returns:
            x        : Standardized fields, 0 where not observed, (C, Z, Y, X) or (C, Y, X).
            observed : Points observed on the sea, 1 where x is valid and 0 elsewhere, same shape.
        """

        # TODO | Names of the dimensions of the stores, GLORYS and the observations being on the same grid
        region = {"longitude": DATASET_REGION["x"], "latitude": DATASET_REGION["y"]}
        if self.modality == "glorys":
            region["depth"] = DATASET_REGION["z"]

        with xr.open_zarr(self.path) as ds:
            x = ds[self.variables].sel(time=date).isel(region).to_array().values

        x = torch.as_tensor(x, dtype=torch.float32)

        # Clipping | Physical bounds of each variable
        for c, var in enumerate(self.variables):
            if var in VARIABLES_CLIPPING:
                x[c] = x[c].clamp(*VARIABLES_CLIPPING[var])

        # Observed | L3 products have gaps (clouds, swaths), every land point being unobserved
        observed = (x.isfinite() & (self.mask > 0)).float()

        if self.standardized:
            x = (x - self.mean) / self.std

        return x.nan_to_num(0.0) * observed, observed


def get_datasets(
    modality: str = "glorys",
    window: int = 1,
    standardized: bool = True,
) -> tuple[HeimdallDataset, HeimdallDataset, HeimdallDataset]:
    r"""Create train, validation and test datasets of a modality using predefined date splits.

    Arguments:
        modality     : Name of the modality, a key of MODALITIES.
        window       : Number of consecutive days of each sample.
        standardized : If True, standardize each variable using precomputed statistics.

    Returns:
        train : Training dataset
        val   : Validation dataset
        test  : Test dataset
    """

    kwargs: dict = {"modality": modality, "window": window, "standardized": standardized}

    return (
        HeimdallDataset(*DATASET_DATES_TRAINING, **kwargs),
        HeimdallDataset(*DATASET_DATES_VALIDATION, **kwargs),
        HeimdallDataset(*DATASET_DATES_TEST, **kwargs),
    )
