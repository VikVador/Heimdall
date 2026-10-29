r"""A collection of tools designed for data module."""

__all__ = [
    "assert_date_format",
    "coarsen",
    "coarsened_shape",
    "count_halvings",
    "crop",
    "decode",
    "load_json",
    "moments",
    "open_zarr",
]

import fsspec
import json
import numpy as np
import re
import torch
import xarray as xr

from pathlib import Path
from torch import Tensor
from typing import Any

from heimdall.config import OSN_STORAGE

# Dimension of the stores along each axis of a region
AXES = {"x": "longitude", "y": "latitude", "z": "depth"}


def assert_date_format(date_string: str) -> None:
    r"""Assert that a date string follows the expected format.

    Arguments:
        date_string : Date to validate, expected format 'YYYY-MM-DD'.
    """

    pattern = r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
    if not re.match(pattern, date_string):
        raise ValueError("ERROR - The format is incorrect, it should be YYYY-MM-DD.")


def coarsen(x: Tensor, mask: Tensor, factor: int) -> tuple[Tensor, Tensor]:
    r"""Average blocks of factor × factor points along the latitude and longitude, over their points having a value.

    Arguments:
        x      : Fields (..., Y, X), any value where there is no value (e.g. NaN).
        mask   : Points having a value, boolean of the shape of x.
        factor : Number of points averaged along the latitude and along the longitude.

    Returns:
        x    : Coarsened fields (..., Y / factor, X / factor), NaN where a block has no value.
        mask : Blocks having at least one value, boolean of the shape of the coarsened x.
    """

    *dims, lat, lon = coarsened_shape(x.shape, factor)

    # Each block is averaged over its points having a value, a block without any becoming NaN
    blocks = (*dims, lat, factor, lon, factor)
    total = torch.where(mask, x, 0.0).reshape(blocks).sum(dim=(-3, -1))
    count = mask.reshape(blocks).sum(dim=(-3, -1))

    return total / count, count > 0


def coarsened_shape(shape: tuple[int, ...], factor: int) -> tuple[int, ...]:
    r"""Return the shape of a grid once coarsened along the latitude and longitude.

    Arguments:
        shape  : Shape of the grid ([Z,] Y, X).
        factor : Number of points averaged along the latitude and along the longitude.

    Returns:
        shape : Coarsened shape ([Z,] Y / factor, X / factor).
    """

    *dims, lat, lon = shape

    # Security
    valid = lat % factor == 0 and lon % factor == 0
    assert valid, f"ERROR - The grid ({lat}, {lon}) must be divisible by the factor {factor}."

    return (*dims, lat // factor, lon // factor)


def count_halvings(size: int) -> int:
    r"""Count how many times a size can be halved without remainder.

    Arguments:
        size : Number of points along an axis.

    Returns:
        halvings : Number of exact halvings, e.g. 3 for 2040 (1020, 510, 255).
    """

    halvings = 0
    while size > 0 and size % 2 == 0:
        size //= 2
        halvings += 1

    return halvings


def crop(
    data: xr.Dataset | xr.DataArray,
    region: dict[str, slice],
    padding: int = 0,
) -> xr.Dataset | xr.DataArray:
    r"""Crop a dataset to a region and pad its longitudes periodically, ignoring the axes it does not have.

    Arguments:
        data    : Dataset or variable, of dimensions among (depth, latitude, longitude).
        region  : Indices along the longitude (x), latitude (y) and depth (z), a value of DATASET_REGION.
        padding : Longitudes added on each side, taken from the other side of the globe, a value of DATASET_PADDING.

    Returns:
        data : Cropped and padded dataset or variable.
    """

    indices = {AXES[axis]: s for axis, s in region.items() if AXES[axis] in data.dims}

    # Padding is an indexing modulo the number of longitudes, so the data is only read when loaded
    if padding > 0 and "longitude" in indices:
        x = indices["longitude"]
        indices["longitude"] = (
            np.arange(x.start - padding, x.stop + padding) % data.sizes["longitude"]
        )

    return data.isel(indices)


def load_json(path: str | Path) -> Any:
    r"""Load a json file, on Torch or on OSN.

    Arguments:
        path : Local path of the file, or its URL on OSN ('s3://...').

    Returns:
        content : Content of the file, or None if the file does not exist.
    """

    path = str(path)
    fs, _ = fsspec.core.url_to_fs(path, **(OSN_STORAGE if path.startswith("s3://") else {}))
    if not fs.exists(path):
        return None

    with fs.open(path) as file:
        return json.load(file)


def moments(x: Tensor) -> Tensor:
    r"""Count, sum and sum of squares of the values of each field, ignoring NaN, in float64.

    Arguments:
        x : Fields (..., Y, X), NaN where there is no value.

    Returns:
        moments : Count, sum and sum of squares over the latitude and longitude (3, ...).
    """

    valid = x.isfinite()
    x = torch.where(valid, x, 0.0).double()

    return torch.stack([
        valid.sum(dim=(-2, -1)).double(),
        x.sum(dim=(-2, -1)),
        x.square().sum(dim=(-2, -1)),
    ])


def open_zarr(path: str | Path) -> xr.Dataset:
    r"""Open a zarr store, on Torch or on OSN, without decoding its packed variables (see decode).

    Arguments:
        path : Local path of the store, or its URL on OSN ('s3://...').

    Returns:
        ds : Lazy dataset (without dask), its variables keeping their stored type (e.g. int16).
    """

    storage_options = OSN_STORAGE if str(path).startswith("s3://") else None
    return xr.open_zarr(path, storage_options=storage_options, mask_and_scale=False, chunks=None)


def decode(da: xr.DataArray, attempts: int = 3) -> np.ndarray:
    r"""Load a variable opened with open_zarr in float32, unpacking int16 values and setting missing ones to NaN.

    Arguments:
        da       : Variable, its attributes holding its scale factor, offset and fill value when it is packed.
        attempts : Number of attempts to read the variable before raising the error.

    Returns:
        values : Physical values in float32, NaN where missing.
    """

    # A read from OSN sometimes fails (corrupted transfer), the same chunk being read correctly again
    for attempt in range(attempts):
        try:
            stored = da.values
            break
        except RuntimeError:
            if attempt == attempts - 1:
                raise

    # Decoding in float32, since xarray would use float64 and double the memory for no precision
    values = stored.astype(np.float32)

    if "scale_factor" in da.attrs or "add_offset" in da.attrs:
        values *= np.float32(da.attrs.get("scale_factor", 1.0))
        values += np.float32(da.attrs.get("add_offset", 0.0))

    fill = da.attrs.get("_FillValue")
    if fill is not None and not np.isnan(fill):
        values[stored == fill] = np.nan

    return values
