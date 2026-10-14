r"""Spatial masks, mesh, date encoding and statistics of the datasets."""

__all__ = [
    "get_weights_mask",
    "get_weights_mesh",
    "get_weights_date",
    "get_weights_stats",
]

import numpy as np
import torch
import xarray as xr

from datetime import date as Date
from torch import Tensor

from heimdall.config import PATH_MASK, PATH_STATS
from heimdall.data import (
    DATASET_OBSERVATIONS,
    DATASET_REGION,
    DATASET_VARIABLES_GLORYS,
    X,
    Y,
    Z,
)


def _prepare(
    t: Tensor,
    dim: int,
    device: torch.device | str | None,
) -> Tensor:
    r"""Optionally add a batch dimension to a tensor and move it to a device.

    Arguments:
        t      : Tensor to prepare.
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device, or None to keep the current one.

    Returns:
        t : Prepared tensor.
    """

    if dim not in {1, 2}:
        raise ValueError(f"ERROR - dim must be 1 or 2, got {dim!r}")
    if dim == 2:
        t = t.unsqueeze(0)
    if device is not None:
        t = t.to(device)

    return t


def _mask_dataarray() -> xr.DataArray:
    r"""Load the land/sea mask of the region from its zarr store.

    Returns:
        mask : Binary data array (Z, Y, X).
    """

    # TODO | Names of the variable and dimensions of the mask store
    with xr.open_zarr(PATH_MASK) as ds:
        return ds.mask.isel(
            longitude=DATASET_REGION["x"],
            latitude=DATASET_REGION["y"],
            depth=DATASET_REGION["z"],
        ).load()


def _encode_sin_cos(values: Tensor) -> Tensor:
    r"""Rescale a coordinate to [0, π] over its range and encode it with sin/cos.

    Arguments:
        values : Coordinate values of any shape (*).

    Returns:
        encoding : Sine and cosine of the rescaled coordinate (2, *).
    """

    angle = torch.pi * (values - values.min()) / (values.max() - values.min())

    return torch.stack([angle.sin(), angle.cos()])


def get_weights_mask(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Load the land/sea masks of the surface and of the ocean.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        mask_surface : Binary mask (1, Y, X), with 1 on sea and 0 on land.
        mask_ocean   : Binary mask (1, Z, Y, X), with 1 on sea and 0 on land.
    """

    mask_ocean = torch.as_tensor(_mask_dataarray().values, dtype=torch.float32)[None]
    mask_surface = mask_ocean[:, 0].clone()

    return _prepare(mask_surface, dim, device), _prepare(mask_ocean, dim, device)


def get_weights_mesh(
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Build the sin/cos encoded mesh of the surface and of the ocean.

    Arguments:
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        mesh_surface, mesh_ocean : (sin, cos) tensors of dimensions (4, Y, X) and (6, Z, Y, X).
    """

    mask = _mask_dataarray()
    depth, latitude, longitude = torch.meshgrid(
        torch.tensor(mask.depth.values, dtype=torch.float32).log(),
        torch.tensor(mask.latitude.values, dtype=torch.float32),
        torch.tensor(mask.longitude.values, dtype=torch.float32),
        indexing="ij",
    )

    mesh_ocean = torch.cat([
        _encode_sin_cos(latitude),
        _encode_sin_cos(longitude),
        _encode_sin_cos(depth),
    ])
    mesh_surface = mesh_ocean[:4, 0].clone()

    return _prepare(mesh_surface, dim, device), _prepare(mesh_ocean, dim, device)


def get_weights_date(
    date: str,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Encode the year progress of a date with sin/cos, broadcast to the surface and ocean grids.

    Arguments:
        date   : Date string 'YYYY-MM-DD'.
        dim    : Use 2 to add a leading batch dimension, 1 otherwise.
        device : Target device ("cpu" or "cuda").

    Returns:
        date_surface : (sin, cos) of the year progress (2, Y, X).
        date_ocean   : (sin, cos) of the year progress (2, Z, Y, X).
    """

    progress = Date.fromisoformat(date).timetuple().tm_yday / 366
    angle = torch.tensor(2 * torch.pi * progress)
    encoding = torch.stack([angle.sin(), angle.cos()])

    date_surface = encoding[:, None, None].expand(2, Y, X)
    date_ocean = encoding[:, None, None, None].expand(2, Z, Y, X)

    return _prepare(date_surface, dim, device), _prepare(date_ocean, dim, device)


def get_weights_stats(
    modality: str,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Load the standardization statistics of the variables of a modality.

    Arguments:
        modality : Name of the modality, 'glorys' or an observation product, a key of DATASET_OBSERVATIONS.
        dim      : Use 2 to add a leading batch dimension, 1 otherwise.
        device   : Target device ("cpu" or "cuda").

    Returns:
        mean : Mean of each variable, (C, Z, 1, 1) per level for GLORYS and (C, 1, 1) for an observation product.
        std  : Standard deviation of each variable, same shape.
    """

    variables = (
        DATASET_VARIABLES_GLORYS if modality == "glorys" else DATASET_OBSERVATIONS[modality]
    )

    # TODO | Structure of the statistics store, a 'statistic' dimension ('mean', 'std') and the depth for GLORYS
    with xr.open_zarr(PATH_STATS) as ds:
        stats = np.stack([ds[var].sel(statistic=["mean", "std"]).values for var in variables])

    stats = torch.as_tensor(stats, dtype=torch.float32)  # (C, 2) or (C, 2, Z)
    if modality == "glorys":
        stats = stats[..., DATASET_REGION["z"]]

    mean, std = stats[:, 0, ..., None, None], stats[:, 1, ..., None, None]

    return _prepare(mean, dim, device), _prepare(std, dim, device)
