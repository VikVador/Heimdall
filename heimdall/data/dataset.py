r"""Dataset."""

__all__ = [
    "HeimdallDataset",
    "get_datasets",
    "get_excluded_days",
]

import torch

from collections.abc import Sequence
from functools import cached_property
from torch import Tensor
from torch.utils.data import Dataset

from heimdall.config import PATH_DATASETS, PATH_EXCLUDED_DAYS
from heimdall.data import (
    DATASET_DATES_TEST,
    DATASET_DATES_TRAINING,
    DATASET_DATES_VALIDATION,
    DATASET_PADDING,
    DATASET_REGION,
    DATASET_VARIABLES,
    MODALITIES,
    VARIABLES_BOUNDS,
    VARIABLES_LOG10,
)
from heimdall.data.tools import assert_date_format, coarsen, crop, decode, load_json, open_zarr
from heimdall.data.weights import get_weights_mask, get_weights_mesh, get_weights_stats


class HeimdallDataset(Dataset):
    r"""Creates a dataset of windows of consecutive days of a single modality (GLORYS or an observation product).

    Arguments:
        date_start     : Start date of the date range (format: 'YYYY-MM-DD').
        date_end       : End date of the date range (format: 'YYYY-MM-DD').
        modality       : Name of the modality, a key of DATASET_VARIABLES.
        window         : Number of consecutive days of each sample.
        standardized   : If True, standardize each variable using its statistics of the training period.
        fill_with_nans : If True, the points without value (land, gaps of the observations) are NaN instead of 0.
        coarsening     : Number of points averaged along the latitude and along the longitude (1 keeps the grid).
        excluded_days  : Days left out of the dataset (format: 'YYYY-MM-DD'), e.g. days without observation.
    """

    def __init__(
        self,
        date_start: str,
        date_end: str,
        modality: str = "glorys",
        window: int = 1,
        standardized: bool = True,
        fill_with_nans: bool = False,
        coarsening: int = 1,
        excluded_days: Sequence[str] = (),
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
        self.fill_with_nans = fill_with_nans
        self.coarsening = coarsening
        self.variables = DATASET_VARIABLES[modality]
        self.mean, self.std = get_weights_stats(modality) if standardized else (None, None)

        # The land/sea mask is the same for every sample, the mesh being computed only when used (never by workers)
        self.mask = get_weights_mask(modality)

        # Days of the store within the date range (the salinity starting on 2010-01-12), without the excluded ones
        excluded = set(excluded_days)
        dates = open_zarr(PATH_DATASETS[modality]).time.dt.strftime("%Y-%m-%d").values
        self.dates = [
            date for date in dates if date_start <= date <= date_end and date not in excluded
        ]

        # Each worker opens the store at its first sample
        self.ds = None

        # Security
        if len(self) <= 0:
            raise ValueError(
                f"ERROR - Not enough dates ({len(self.dates)}) for a window of {window} days."
            )

    @cached_property
    def mesh(self) -> Tensor:
        r"""Sin/cos mesh of the latitude, longitude [and depth], (6, Z, Y, X) for GLORYS and (4, Y, X) otherwise."""
        return get_weights_mesh(self.modality, self.coarsening)

    def __getstate__(self) -> dict:
        r"""Return the state of the dataset sent to a worker, without the opened store nor the mesh."""
        return {key: value for key, value in self.__dict__.items() if key != "mesh"} | {"ds": None}

    def __len__(self) -> int:
        r"""Return the number of windows of consecutive days in the date range."""
        return len(self.dates) - self.window + 1

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, list[str]]:
        r"""Return a window of consecutive days, with the points having a value.

        Arguments:
            idx : Index of the window.

        Returns:
            x     : States or observations (T, C, Z, Y, X) for GLORYS or (T, C, Y, X) for an observation product.
            mask  : Points where each variable has a value, sea and observed that day, boolean of the shape of x.
            dates : Date strings 'YYYY-MM-DD' of the window (T,).
        """

        dates = self.dates[idx : idx + self.window]
        fields = [self.preprocess(date) for date in dates]

        return torch.stack([x for x, _ in fields]), torch.stack([m for _, m in fields]), dates

    def preprocess(self, date: str) -> tuple[Tensor, Tensor]:
        r"""Load and preprocess the modality on a single day.

        Arguments:
            date : Date string 'YYYY-MM-DD'.

        Returns:
            x    : Fields (C, Z, Y, X) or (C, Y, X), 0 or NaN where there is no value (CHL in log10).
            mask : Points where each variable has a value, sea and observed that day, boolean of the shape of x.
        """

        if self.ds is None:
            self.ds = crop(
                open_zarr(PATH_DATASETS[self.modality]),
                DATASET_REGION[self.modality],
                DATASET_PADDING[self.modality],
            )

        day = self.ds.sel(time=date)
        x = torch.stack([torch.from_numpy(decode(day[var])) for var in self.variables])

        # Values outside the physical bounds of their variable are errors of the data, treated as missing
        for c, var in enumerate(self.variables):
            low, high = VARIABLES_BOUNDS[var]
            x[c] = torch.where((x[c] >= low) & (x[c] <= high), x[c], torch.nan)
            if var in VARIABLES_LOG10:
                x[c] = x[c].log10()

        # L3 products have gaps (clouds, swaths), and land points never have a value
        mask = x.isfinite() & (self.mask > 0)

        if self.coarsening > 1:
            x, mask = coarsen(x, mask, self.coarsening)

        if self.standardized:
            x = (x - self.mean) / self.std

        x = torch.where(mask, x, torch.nan if self.fill_with_nans else 0.0)

        return x, mask


def get_excluded_days(modality: str) -> list[str]:
    r"""Load the days left out of a dataset, a dataset without list having no excluded day (e.g. GLORYS).

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.

    Returns:
        dates : Date strings 'YYYY-MM-DD' of the excluded days.
    """

    return load_json(PATH_EXCLUDED_DAYS[modality]) or []


def get_datasets(
    modality: str = "glorys",
    window: int = 1,
    standardized: bool = True,
    fill_with_nans: bool = False,
    coarsening: int = 1,
) -> tuple[HeimdallDataset, HeimdallDataset, HeimdallDataset]:
    r"""Create train, validation and test datasets of a modality using predefined date splits.

    Arguments:
        modality       : Name of the modality, a key of DATASET_VARIABLES.
        window         : Number of consecutive days of each sample.
        standardized   : If True, standardize each variable using its statistics.
        fill_with_nans : If True, the points without value are NaN instead of 0.
        coarsening     : Number of points averaged along the latitude and along the longitude.

    Returns:
        train : Training dataset
        val   : Validation dataset
        test  : Test dataset
    """

    kwargs: dict = {
        "modality": modality,
        "window": window,
        "standardized": standardized,
        "fill_with_nans": fill_with_nans,
        "coarsening": coarsening,
        "excluded_days": get_excluded_days(modality),
    }

    return (
        HeimdallDataset(*DATASET_DATES_TRAINING, **kwargs),
        HeimdallDataset(*DATASET_DATES_VALIDATION, **kwargs),
        HeimdallDataset(*DATASET_DATES_TEST, **kwargs),
    )
