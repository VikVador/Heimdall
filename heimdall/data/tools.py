r"""A collection of tools designed for data module."""

__all__ = [
    "assert_date_format",
    "count_halvings",
    "crop",
    "decode",
    "open_zarr",
]

import numpy as np
import re
import xarray as xr

from pathlib import Path

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


def open_zarr(path: str | Path) -> xr.Dataset:
    r"""Open a zarr store, on Torch or on OSN, without decoding its packed variables (see decode).

    Arguments:
        path : Local path of the store, or its URL on OSN ('s3://...').

    Returns:
        ds : Lazy dataset (without dask), its variables keeping their stored type (e.g. int16).
    """

    storage_options = OSN_STORAGE if str(path).startswith("s3://") else None
    return xr.open_zarr(path, storage_options=storage_options, mask_and_scale=False, chunks=None)


def decode(da: xr.DataArray) -> np.ndarray:
    r"""Load a variable opened with open_zarr in float32, unpacking int16 values and setting missing ones to NaN.

    Arguments:
        da : Variable, its attributes holding its scale factor, offset and fill value when it is packed.

    Returns:
        values : Physical values in float32, NaN where missing.
    """

    # Decoding in float32, since xarray would use float64 and double the memory for no precision
    stored = da.values
    values = stored.astype(np.float32)

    if "scale_factor" in da.attrs or "add_offset" in da.attrs:
        values *= np.float32(da.attrs.get("scale_factor", 1.0))
        values += np.float32(da.attrs.get("add_offset", 0.0))

    fill = da.attrs.get("_FillValue")
    if fill is not None and not np.isnan(fill):
        values[stored == fill] = np.nan

    return values
