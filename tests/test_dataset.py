r"""Tests for heimdall.data.dataset."""

import numpy as np
import pytest
import torch

from heimdall.data import DATASET_PADDING, DATASET_SHAPES, DATASET_VARIABLES, VARIABLES_BOUNDS
from heimdall.data.dataset import HeimdallDataset


@pytest.mark.integration
@pytest.mark.parametrize("modality", ["sst", "wind_asc"])
def test_heimdall_dataset_typical(modality: str) -> None:
    r"""Determines if a window of days has the shape of the modality, 0 where there is no value."""

    dataset = HeimdallDataset("2012-08-01", "2012-08-03", modality=modality, window=2)
    x, mask, dates = dataset[0]

    assert len(dataset) == 2
    assert dates == ["2012-08-01", "2012-08-02"]
    assert x.shape == (2, len(DATASET_VARIABLES[modality]), *DATASET_SHAPES[modality])
    assert mask.shape == x.shape
    assert (x[mask == 0] == 0).all()
    assert x.dtype == torch.bfloat16 and mask.dtype == torch.bool
    assert x.isfinite().all() and 0 < mask.float().mean() < 1


@pytest.mark.integration
def test_heimdall_dataset_fill_with_nans() -> None:
    r"""Determines if the points without value are NaN, the others keeping their physical values."""

    dataset = HeimdallDataset(
        "2012-08-01", "2012-08-01", modality="sst", standardized=False, fill_with_nans=True
    )
    x, mask, _ = dataset[0]

    assert x[mask == 0].isnan().all()
    assert 270 < x[mask == 1].mean() < 305  # Kelvin


@pytest.mark.integration
def test_heimdall_dataset_salinity_dates() -> None:
    r"""Determines if a dataset only keeps the days of its store, the salinity starting on 2010-01-12."""

    dataset = HeimdallDataset("2010-01-01", "2010-01-31", modality="sss_asc")

    assert dataset.dates[0] == "2010-01-12"


@pytest.mark.integration
def test_heimdall_dataset_padding() -> None:
    r"""Determines if the padded longitudes of the fields and of the masks are the ones of the other side of the globe."""

    dataset = HeimdallDataset("2012-08-01", "2012-08-01", modality="sst")
    x, mask, _ = dataset[0]
    p, n = DATASET_PADDING["sst"], DATASET_SHAPES["sst"][-1] - 2 * DATASET_PADDING["sst"]

    for field in (x, mask, dataset.mask, dataset.mesh):
        assert torch.equal(field[..., :p], field[..., n : n + p])
        assert torch.equal(field[..., -p:], field[..., p : 2 * p])


@pytest.mark.integration
def test_heimdall_dataset_coarsening() -> None:
    r"""Determines if GLORYS is averaged along the latitude and longitude, keeping all its levels."""

    full = HeimdallDataset("2012-08-01", "2012-08-01", modality="glorys", standardized=False)
    coarse = HeimdallDataset(
        "2012-08-01", "2012-08-01", modality="glorys", standardized=False, coarsening=4
    )
    x, mask, _ = full[0]
    x_coarse, mask_coarse, _ = coarse[0]

    assert x_coarse.shape == mask_coarse.shape == (1, 4, 48, 504, 1080)
    assert (x_coarse[~mask_coarse] == 0).all()
    assert mask[0, 0, 0, 400:404, 800:804].all()  # A block of the South Pacific, only sea points
    torch.testing.assert_close(x_coarse[0, 0, 0, 100, 200], x[0, 0, 0, 400:404, 800:804].mean())
    assert mask_coarse.float().mean() >= mask.float().mean()


@pytest.mark.integration
def test_heimdall_dataset_mask_per_variable() -> None:
    r"""Determines if a variable keeps its values where another variable of the modality has none (GLORYS coasts)."""

    dataset = HeimdallDataset("2012-08-01", "2012-08-01", modality="glorys", standardized=False)
    x, mask, _ = dataset[0]
    thetao, uo = mask[0, 0].bool(), mask[0, 2].bool()

    assert (thetao & ~uo).any()  # Coastal points with a temperature but no current
    assert (x[0, 0][thetao & ~uo] != 0).any()


@pytest.mark.integration
def test_heimdall_dataset_bounds() -> None:
    r"""Determines if the values outside the physical bounds are missing (salinity of -2.75 in the Laptev Sea)."""

    dataset = HeimdallDataset(
        "2012-09-02", "2012-09-02", modality="sss_asc", standardized=False, fill_with_nans=True
    )
    x, mask, _ = dataset[0]
    low, high = VARIABLES_BOUNDS["Sea_Surface_Salinity"]

    assert mask.any()
    assert (x[mask] >= low).all() and (x[mask] <= high).all()


@pytest.mark.integration
def test_heimdall_dataset_excluded_days() -> None:
    r"""Determines if the excluded days are left out, the windows joining the remaining days."""

    dataset = HeimdallDataset(
        "2012-08-01", "2012-08-03", modality="sst", excluded_days=["2012-08-02"]
    )

    assert dataset.dates == ["2012-08-01", "2012-08-03"]


@pytest.mark.integration
def test_heimdall_dataset_log10() -> None:
    r"""Determines if the chlorophyll is given in log10, within the log10 of its physical bounds."""

    dataset = HeimdallDataset("2012-08-01", "2012-08-01", modality="chl", standardized=False)
    x, mask, _ = dataset[0]
    low, high = VARIABLES_BOUNDS["CHL"]

    assert (x[mask] >= np.log10(low)).all() and (x[mask] <= np.log10(high)).all()
    assert -1.5 < x[mask].mean() < 0  # Most of the ocean has 0.03 to 1 mg/m³
