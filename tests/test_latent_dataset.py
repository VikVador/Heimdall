r"""Tests for heimdall/data/latents.py."""

import numpy as np
import pytest
import torch
import xarray as xr

from pathlib import Path

from heimdall.data import latents

DAYS = ["2012-08-01", "2012-08-02", "2012-08-03", "2012-08-04", "2012-08-05", "2012-08-06"]


def write_store(
    path: Path, dates: list[str], shape: tuple[int, ...], value: float | None = None
) -> None:
    r"""Write a fake latent store of a split and its statistics, one day of value i + 1 everywhere on day i."""

    data = np.stack([
        np.full(shape, i + 1 if value is None else value, dtype=np.float16)
        for i in range(len(dates))
    ])
    dims = (
        ["time", "channel", "level", "y", "x"]
        if len(shape) == 4
        else ["time", "channel", "y", "x"]
    )
    xr.Dataset(
        {"latents": (dims, data), "encoded": ("time", np.ones(len(dates), dtype=bool))},
        coords={"time": np.array(dates, dtype="datetime64[ns]")},
    ).to_zarr(path / "training.zarr")
    statistics = np.stack([np.full(shape[0], 1.0), np.full(shape[0], 2.0)]).astype(np.float32)
    xr.Dataset(
        {"latents": (["statistic", "channel"], statistics)}, coords={"statistic": ["mean", "std"]}
    ).to_zarr(path / "statistics.zarr")


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    r"""Fake latents of GLORYS on 6 consecutive days and of the SST without its third day."""

    paths = {"glorys": tmp_path / "glorys", "sst": tmp_path / "sst"}
    write_store(paths["glorys"], DAYS, (2, 3, 4, 5))
    write_store(paths["sst"], DAYS[:2] + DAYS[3:], (2, 4, 6), value=5.0)
    monkeypatch.setattr(latents, "PATH_LATENT_DATASETS", paths)

    return paths


def test_latent_dataset_typical(stores: dict[str, Path]) -> None:
    r"""Determines if a window holds standardized states around its block and the observations of the block."""

    dataset = latents.LatentDataset("training", block=2, blanket=1, observations=["sst"])
    x, y, _, dates = dataset[0]

    assert len(dataset) == 3
    assert dates == DAYS[:4]
    assert x.shape == (4, 2, 3, 4, 5)
    assert y["sst"].shape == (2, 2, 4, 6)
    assert x.dtype == y["sst"].dtype == torch.float32
    np.testing.assert_allclose(x[:, 0, 0, 0, 0].numpy(), [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(y["sst"][0].numpy(), 2.0)


def test_latent_dataset_missing_day(stores: dict[str, Path]) -> None:
    r"""Determines if a day without observation is zero and flagged as not available."""

    dataset = latents.LatentDataset("training", block=2, blanket=1, observations=["sst"])
    _, y, available, _ = dataset[1]

    assert available["sst"].tolist() == [False, True]
    assert (y["sst"][0] == 0).all()
    np.testing.assert_allclose(y["sst"][1].numpy(), 2.0)


def test_latent_dataset_without_observations(stores: dict[str, Path]) -> None:
    r"""Determines if the dataset gives the states alone when no observation product is asked."""

    dataset = latents.LatentDataset("training", block=4, blanket=1, observations=[])
    x, y, available, _ = dataset[0]

    assert len(dataset) == 1
    assert x.shape[0] == 6
    assert y == {} and available == {}


@pytest.mark.integration
def test_latent_dataset_torch() -> None:
    r"""Determines if a window of the real latents on Torch has the shapes of the encoders and finite values."""

    dataset = latents.LatentDataset("validation", block=3, blanket=1)
    x, y, available, dates = dataset[100]

    assert len(dates) == 5
    assert x.shape == (5, 128, 3, 63, 135)
    assert set(y) == set(latents.OBSERVATIONS)
    assert all(y[m].shape[0] == 3 and available[m].shape == (3,) for m in y)
    assert x.isfinite().all() and all(y[m].isfinite().all() for m in y)
