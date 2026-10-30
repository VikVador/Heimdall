r"""Land/sea mask, mesh, date encoding and statistics of each modality, on its cropped grid."""

__all__ = [
    "get_weights_mask",
    "get_weights_mesh",
    "get_weights_date",
    "get_weights_stats",
    "get_weights_change",
]

import numpy as np
import torch
import xarray as xr

from datetime import date as Date
from torch import Tensor

from heimdall.config import PATH_ARTEFACTS, PATH_MASKS, PATH_STATS
from heimdall.data import (
    DATASET_PADDING,
    DATASET_REGION,
    DATASET_SHAPES,
    DATASET_VARIABLES,
)
from heimdall.data.tools import coarsened_shape, crop, load_json, load_npy, open_zarr


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


def _mask_dataarray(modality: str) -> xr.DataArray:
    r"""Load the land/sea mask of a modality on its region.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.

    Returns:
        mask : Binary data array ([Z,] Y, X), with its coordinates.
    """

    mask = open_zarr(PATH_MASKS[modality]).mask

    return crop(mask, DATASET_REGION[modality], DATASET_PADDING[modality]).load()


def _nearest(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    r"""Find the index of the closest point of an ascending grid for each value.

    Arguments:
        values : Coordinates to look for (N,).
        grid   : Coordinates of the grid, sorted in ascending order (M,).

    Returns:
        indices : Index of the closest grid point of each value (N,).
    """

    i = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)

    return np.where(values - grid[i - 1] < grid[i] - values, i - 1, i)


def _ocean(mask: xr.DataArray) -> np.ndarray:
    r"""Find the points of a grid that are ocean in GLORYS at the surface, the closest GLORYS point deciding.

    Arguments:
        mask : Land/sea mask of an observation product (Y, X), with its coordinates.

    Returns:
        ocean : Points of the grid that are ocean in GLORYS, boolean (Y, X).
    """

    glorys = open_zarr(PATH_MASKS["glorys"]).mask.isel(depth=0)
    y = _nearest(mask.latitude.values, glorys.latitude.values)
    x = _nearest(mask.longitude.values, glorys.longitude.values)

    return glorys.values[np.ix_(y, x)] > 0


def _encode_sin_cos(values: Tensor, period: float | None = None) -> Tensor:
    r"""Encode a coordinate with sin/cos, over its period or rescaled to [0, π] over its range.

    Arguments:
        values : Coordinate values of any shape (*).
        period : Period of the coordinate (e.g. 360° for the longitude), or None if it is not periodic.

    Returns:
        encoding : Sine and cosine of the coordinate (2, *).
    """

    if period is None:
        angle = torch.pi * (values - values.min()) / (values.max() - values.min())
    else:
        angle = 2 * torch.pi * values / period

    return torch.stack([angle.sin(), angle.cos()])


def get_weights_mask(
    modality: str,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Load the land/sea mask of a modality, without lakes nor artefact columns.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
        dim      : Use 2 to add a leading batch dimension, 1 otherwise.
        device   : Target device ("cpu" or "cuda").

    Returns:
        mask : Binary mask (1, [Z,] Y, X), with 1 on sea and 0 on land.
    """

    mask = _mask_dataarray(modality)
    sea = mask.values > 0

    # The satellites also observe lakes (e.g. Titicaca), which are not part of the ocean of GLORYS
    if modality != "glorys":
        sea &= _ocean(mask)

    # Columns whose values are out of their physical bounds on many days are artefacts (straits, estuaries)
    artefacts = load_npy(PATH_ARTEFACTS[modality]) if modality in PATH_ARTEFACTS else None
    if artefacts is not None:
        sea &= ~artefacts

    mask = torch.as_tensor(sea, dtype=torch.float32)[None]

    return _prepare(mask, dim, device)


def get_weights_mesh(
    modality: str,
    coarsening: int = 1,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Build the sin/cos encoded mesh of a modality, the longitude being periodic and the depth logarithmic.

    Arguments:
        modality   : Name of the modality, a key of DATASET_VARIABLES.
        coarsening : Number of points averaged along the latitude and along the longitude.
        dim        : Use 2 to add a leading batch dimension, 1 otherwise.
        device     : Target device ("cpu" or "cuda").

    Returns:
        mesh : (sin, cos) of the latitude, longitude [and depth], (6, Z, Y, X) for GLORYS and (4, Y, X) otherwise.
    """

    mask = _mask_dataarray(modality)
    shape = coarsened_shape(mask.shape, coarsening)

    # Each coordinate is encoded along its own axis, averaged over the coarsened blocks, then broadcast to the grid
    encodings = []
    for name, period in (("latitude", None), ("longitude", 360.0), ("depth", None)):
        if name in mask.dims:
            values = torch.tensor(mask[name].values, dtype=torch.float32)
            values = values.log() if name == "depth" else values
            encoding = _encode_sin_cos(values, period)
            if name != "depth":
                encoding = encoding.unflatten(-1, (-1, coarsening)).mean(dim=-1)
            axes = [-1 if d == name else 1 for d in mask.dims]
            encodings.append(encoding.reshape(2, *axes).expand(2, *shape))

    return _prepare(torch.cat(encodings), dim, device)


def get_weights_date(
    date: str,
    modality: str,
    coarsening: int = 1,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Encode the year progress of a date with sin/cos, broadcast to the grid of a modality.

    Arguments:
        date       : Date string 'YYYY-MM-DD'.
        modality   : Name of the modality, a key of DATASET_VARIABLES.
        coarsening : Number of points averaged along the latitude and along the longitude.
        dim        : Use 2 to add a leading batch dimension, 1 otherwise.
        device     : Target device ("cpu" or "cuda").

    Returns:
        encoding : (sin, cos) of the year progress (2, [Z,] Y, X).
    """

    progress = Date.fromisoformat(date).timetuple().tm_yday / 366
    angle = torch.tensor(2 * torch.pi * progress)
    shape = coarsened_shape(DATASET_SHAPES[modality], coarsening)
    encoding = torch.stack([angle.sin(), angle.cos()])
    encoding = encoding.reshape(2, *[1] * len(shape)).expand(2, *shape)

    return _prepare(encoding, dim, device)


def _statistics(modality: str, names: list[str]) -> Tensor:
    r"""Load statistics of the variables of a modality, computed on its training period.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
        names    : Names of the statistics, among 'mean', 'std' and 'std_change'.

    Returns:
        stats : Statistics (S, C, Z, 1, 1) per level for GLORYS and (S, C, 1, 1) otherwise.
    """

    content = load_json(PATH_STATS[modality])
    assert content is not None, f"ERROR - No statistics for {modality} at {PATH_STATS[modality]}."

    stats = torch.tensor(
        [[content[var][name] for var in DATASET_VARIABLES[modality]] for name in names],
        dtype=torch.float32,
    )  # (S, C, Z), with Z = 1 for the observations
    if len(DATASET_SHAPES[modality]) == 2:
        stats = stats[..., 0]

    return stats[..., None, None]


def get_weights_stats(
    modality: str,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Load the mean and standard deviation of the variables of a modality, on its training period.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
        dim      : Use 2 to add a leading batch dimension, 1 otherwise.
        device   : Target device ("cpu" or "cuda").

    Returns:
        mean : Mean of each variable, (C, Z, 1, 1) per level for GLORYS and (C, 1, 1) otherwise.
        std  : Standard deviation of each variable, same shape.
    """

    mean, std = _statistics(modality, ["mean", "std"])

    return _prepare(mean, dim, device), _prepare(std, dim, device)


def get_weights_change(
    modality: str,
    *,
    dim: int = 1,
    device: torch.device | str | None = None,
) -> Tensor:
    r"""Load the standard deviation of the change of each variable from one day to the next, on the training period.

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.
        dim      : Use 2 to add a leading batch dimension, 1 otherwise.
        device   : Target device ("cpu" or "cuda").

    Returns:
        std_change : Standard deviation of x(t + 1) - x(t), (C, Z, 1, 1) per level for GLORYS and (C, 1, 1) otherwise.
    """

    (std_change,) = _statistics(modality, ["std_change"])

    return _prepare(std_change, dim, device)
