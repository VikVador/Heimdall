r"""Tests for experiments/encoders/latents.py."""

import importlib.util
import numpy as np
import pytest
import xarray as xr
import zarr

from pathlib import Path
from shaggy.tools import save

from heimdall.data import DATASET_SHAPES
from heimdall.model.encoders import Encoder

# The script lives in an experiment folder, which is not a package
SPEC = importlib.util.spec_from_file_location(
    "latents", Path(__file__).parents[1] / "experiments" / "encoders" / "latents.py"
)
latents = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latents)

CONFIG = {"hid_channels": [4, 4, 4], "hid_blocks": [1, 1, 1], "kernel_size": 3, "stride": 2}
DATES = ["2012-08-01", "2012-08-02", "2012-08-04"]
DIMS = ["time", "channel", "y", "x"]
ATTRS = {
    "modality": "sst",
    "model": "ENC_sst__TEST/best-validation",
    "weights": "0123456789abcdef",
}


def test_compute_statistics_typical() -> None:
    r"""Determines if the statistics are the mean and std of each channel over the days, levels and points."""

    rng = np.random.default_rng(0)
    z = rng.normal(5.0, 2.0, size=(5, 3, 2, 4, 6)).astype(np.float16)

    # Five days read two at a time, the last read holding a single day
    statistics = latents.compute_statistics(xr.DataArray(z), days=2)

    z = z.astype(np.float64)
    assert statistics.shape == (2, 3) and statistics.dtype == np.float32
    assert np.allclose(statistics[0], z.mean(axis=(0, 2, 3, 4)), rtol=1e-5)
    assert np.allclose(statistics[1], z.std(axis=(0, 2, 3, 4)), rtol=1e-5)


def test_compute_statistics_constant_channel() -> None:
    r"""Determines if a constant channel of a single day keeps its value as mean and a tiny positive std."""

    z = np.ones((1, 2, 4, 6), dtype=np.float16)
    z[:, 1] = np.arange(24).reshape(4, 6)

    statistics = latents.compute_statistics(xr.DataArray(z))

    assert statistics[0, 0] == 1.0 and 0.0 < statistics[1, 0] <= 1e-8
    assert np.isclose(statistics[0, 1], 11.5) and np.isclose(statistics[1, 1], np.arange(24).std())


def test_create_store_typical(tmp_path: Path) -> None:
    r"""Determines if a new store has the dates of the split, one day per chunk, and no day encoded yet."""

    latents.create_store(tmp_path / "training.zarr", DATES, (8, 3, 5), DIMS, ATTRS)
    store = xr.open_zarr(tmp_path / "training.zarr", chunks=None)

    assert store.latents.shape == (3, 8, 3, 5) and store.latents.dtype == np.float16
    assert zarr.open_array(tmp_path / "training.zarr" / "latents").chunks == (1, 8, 3, 5)
    assert np.array_equal(store.time.values, np.array(DATES, dtype="datetime64[ns]"))
    assert store.attrs == ATTRS
    assert not store.encoded.values.any() and np.isnan(store.latents.values).all()


def test_create_store_other_encoder(tmp_path: Path) -> None:
    r"""Determines if a store is resumed with the same encoder and days, and refused with other weights or days."""

    path = tmp_path / "training.zarr"
    latents.create_store(path, DATES, (8, 3, 5), DIMS, ATTRS)
    latents.create_store(path, DATES, (8, 3, 5), DIMS, ATTRS)

    with pytest.raises(AssertionError, match="delete it"):
        latents.create_store(
            path, DATES, (8, 3, 5), DIMS, {**ATTRS, "weights": "fedcba9876543210"}
        )
    with pytest.raises(AssertionError, match="delete it"):
        latents.create_store(path, DATES[:2], (8, 3, 5), DIMS, ATTRS)


@pytest.mark.integration
def test_encode_modality_typical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r"""Determines if every day of a split is encoded in order, resumed without encoding again, with its statistics."""

    model = Encoder("wind_asc", lat_channels=8, **CONFIG)
    path = tmp_path / "models" / "ENC_test" / "best-validation"
    save(
        model, {"modality": "wind_asc", "lat_channels": 8, **CONFIG, "domain": model.domain}, path
    )
    monkeypatch.setattr(latents, "PATH_MODELS", tmp_path / "models")
    monkeypatch.setattr(latents, "PATH_LATENTS", tmp_path / "latents")
    monkeypatch.setattr(latents, "SPLITS", {"training": ("2012-08-01", "2012-08-05")})

    # Five days in batches of two, the last batch holding a single day
    kwargs = {"batch_size": 2, "num_workers": 0, "prefetch_factor": 1}
    latents.encode_modality("wind_asc", "ENC_test", "best-validation", ["training"], **kwargs)

    path_latents = tmp_path / "latents" / "ENC_test" / "best-validation"
    store = xr.open_zarr(path_latents / "training.zarr", chunks=None)
    z = store.latents.values
    dates = np.array([f"2012-08-0{d}" for d in range(1, 6)], dtype="datetime64[ns]")

    assert (
        z.shape == (5, 8, *[n // 4 for n in DATASET_SHAPES["wind_asc"]]) and z.dtype == np.float16
    )
    assert np.array_equal(store.time.values, dates) and store.encoded.values.all()
    assert np.isfinite(z).all() and store.attrs["model"] == "ENC_test/best-validation"

    # A job stopped after three days, the first one marked to check that it is not encoded again
    group = zarr.open_group(path_latents / "training.zarr", mode="r+")
    group["latents"][0] = 7.0
    group["latents"][3:] = np.nan
    group["encoded"][3:] = False
    latents.encode_modality("wind_asc", "ENC_test", "best-validation", ["training"], **kwargs)

    resumed = xr.open_zarr(path_latents / "training.zarr", chunks=None).latents.values
    assert (resumed[0] == 7.0).all() and np.allclose(resumed[1:], z[1:], rtol=1e-2, atol=1e-3)

    statistics = xr.open_zarr(path_latents / "statistics.zarr").latents
    resumed = resumed.astype(np.float64)
    assert statistics.dims == ("statistic", "channel") and statistics.shape == (2, 8)
    assert np.allclose(statistics.sel(statistic="mean"), resumed.mean(axis=(0, 2, 3)), rtol=1e-4)
    assert np.allclose(statistics.sel(statistic="std"), resumed.std(axis=(0, 2, 3)), rtol=1e-4)
