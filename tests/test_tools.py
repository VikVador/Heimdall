r"""Tests for heimdall.tools and heimdall.data.tools."""

import numpy as np
import pytest
import xarray as xr

from heimdall.data.tools import assert_date_format, count_halvings, crop, decode
from heimdall.tools import extract_model_hash, generate_run_name, get_wandb_hyperparameters


def test_generate_run_name_typical() -> None:
    r"""Determines if a run name holds its own hash, and the hash of the run it resumes."""

    first = generate_run_name("ENC_glorys_lc256")
    second = generate_run_name("ENC_glorys_lc256", previous_run_name=first)

    assert first.startswith("ENC_glorys_lc256__")
    assert second.endswith(f"_{extract_model_hash(first)}")
    assert extract_model_hash(second) != extract_model_hash(first) or len(second) > len(first)


def test_get_wandb_hyperparameters_typical() -> None:
    r"""Determines if a nested configuration is flattened into scalar hyperparameters."""

    params = get_wandb_hyperparameters({
        "lat_channels": 256,
        "config_glorys": {"hid_channels": [32, 64]},
    })

    assert params == {
        "lat_channels": 256,
        "config_glorys/hid_channels/0": 32,
        "config_glorys/hid_channels/1": 64,
    }


def test_assert_date_format_typical() -> None:
    r"""Determines if a valid date passes and a wrongly formatted one is rejected."""

    assert_date_format("2020-01-31")
    with pytest.raises(ValueError):
        assert_date_format("31/01/2020")


def test_decode_typical() -> None:
    r"""Determines if packed int16 values are unpacked in float32, missing values becoming NaN."""

    stored = np.array([0, 5731, -32767], dtype=np.int16)
    da = xr.DataArray(
        stored, attrs={"scale_factor": 0.001, "add_offset": 20.0, "_FillValue": -32767}
    )

    values = decode(da)

    assert values.dtype == np.float32
    np.testing.assert_allclose(values[:2], [20.0, 25.731], rtol=1e-6)
    assert np.isnan(values[2])


def test_decode_float32() -> None:
    r"""Determines if a variable stored in float32, without packing, is returned unchanged with its NaN."""

    da = xr.DataArray(np.array([0.17, np.nan], dtype=np.float32), attrs={"_FillValue": np.nan})

    values = decode(da)

    assert values.dtype == np.float32
    assert values[0] == np.float32(0.17)
    assert np.isnan(values[1])


def test_decode_all_missing() -> None:
    r"""Determines if a field entirely missing (e.g. land) is only made of NaN."""

    da = xr.DataArray(
        np.full(4, -32767, dtype=np.int16), attrs={"scale_factor": 0.01, "_FillValue": -32767}
    )

    assert np.isnan(decode(da)).all()


def test_count_halvings_typical() -> None:
    r"""Determines if the number of exact halvings of a size is counted."""

    assert count_halvings(2040) == 3
    assert count_halvings(4320) == 5
    assert count_halvings(48) == 4


def test_count_halvings_odd() -> None:
    r"""Determines if an odd size cannot be halved, nor an empty one."""

    assert count_halvings(2041) == 0
    assert count_halvings(0) == 0


def test_crop_typical() -> None:
    r"""Determines if a region crops the axes of a dataset, the axes it does not have being ignored."""

    ds = xr.Dataset({
        "x3d": (("depth", "latitude", "longitude"), np.zeros((5, 6, 8))),
        "x2d": (("latitude", "longitude"), np.zeros((6, 8))),
    })

    cropped = crop(ds, {"x": slice(0, 4), "y": slice(1, 5), "z": slice(0, 2)})

    assert cropped.x3d.shape == (2, 4, 4)
    assert crop(ds.x2d, {"x": slice(0, 4), "y": slice(1, 5), "z": slice(0, 2)}).shape == (4, 4)


def test_crop_padding() -> None:
    r"""Determines if the longitudes are padded on each side with the ones of the other side of the globe."""

    da = xr.DataArray(
        np.arange(8.0)[None].repeat(3, axis=0),
        dims=("latitude", "longitude"),
        coords={"longitude": np.arange(8.0) * 45 - 180},
    )

    padded = crop(da, {"x": slice(0, 8), "y": slice(0, 3)}, padding=2)

    assert padded.shape == (3, 12)
    np.testing.assert_array_equal(padded[0], [6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
    np.testing.assert_array_equal(padded.longitude[:2], [90.0, 135.0])
