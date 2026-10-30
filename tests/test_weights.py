r"""Tests for heimdall.data.weights."""

import numpy as np
import pytest
import torch

from heimdall.config import PATH_ARTEFACTS
from heimdall.data import DATASET_SHAPES, DATASET_VARIABLES, MODALITIES
from heimdall.data.tools import load_npy
from heimdall.data.weights import (
    _encode_sin_cos,
    _mask_dataarray,
    _nearest,
    get_weights_change,
    get_weights_date,
    get_weights_mask,
    get_weights_mesh,
    get_weights_stats,
)


def test_encode_sin_cos_periodic() -> None:
    r"""Determines if a periodic coordinate is encoded over its period, -180° and 180° being the same point."""

    encoding = _encode_sin_cos(torch.tensor([-180.0, 0.0, 180.0]), period=360.0)

    assert encoding.shape == (2, 3)
    torch.testing.assert_close(encoding[:, 0], encoding[:, 2])
    torch.testing.assert_close(encoding[:, 1], torch.tensor([0.0, 1.0]))


def test_encode_sin_cos_range() -> None:
    r"""Determines if a coordinate without period is rescaled to [0, π] over its range."""

    encoding = _encode_sin_cos(torch.tensor([10.0, 20.0, 30.0]))

    torch.testing.assert_close(encoding[1], torch.tensor([1.0, 0.0, -1.0]), atol=1e-6, rtol=0)


def test_get_weights_date_shape() -> None:
    r"""Determines if the date encoding is broadcast to the grid of a modality."""

    for modality in ("glorys", "sst"):
        encoding = get_weights_date("2020-07-01", modality, dim=2)
        assert encoding.shape == (1, 2, *DATASET_SHAPES[modality])

    encoding = get_weights_date("2020-07-01", "glorys", coarsening=4)
    assert encoding.shape == (2, 48, 504, 1080)


@pytest.mark.integration
@pytest.mark.parametrize("modality", MODALITIES)
def test_get_weights_typical(modality: str) -> None:
    r"""Determines if the mask, mesh and statistics of a modality match its cropped grid."""

    shape = DATASET_SHAPES[modality]
    mask, mesh = get_weights_mask(modality), get_weights_mesh(modality)
    mean, std = get_weights_stats(modality)
    std_change = get_weights_change(modality)

    assert mask.shape == (1, *shape)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mesh.shape == (2 * len(shape), *shape)
    assert mesh.abs().max() <= 1.0
    assert mean.shape == std.shape == (len(DATASET_VARIABLES[modality]), *shape[:-2], 1, 1)
    assert (std > 0).all() and mean.isfinite().all()
    assert std_change.shape == std.shape and (std_change > 0).all()


@pytest.mark.integration
def test_get_weights_mesh_coarsening() -> None:
    r"""Determines if the coarsened mesh is the average of the mesh over each block of points."""

    mesh, coarse = get_weights_mesh("wind_asc"), get_weights_mesh("wind_asc", coarsening=4)
    lat, lon = DATASET_SHAPES["wind_asc"]

    assert coarse.shape == (4, lat // 4, lon // 4)
    torch.testing.assert_close(
        coarse, mesh.unflatten(1, (-1, 4)).unflatten(-1, (-1, 4)).mean(dim=(2, 4))
    )


def test_nearest_typical() -> None:
    r"""Determines if each value gets the index of its closest grid point, values outside the grid getting its ends."""

    grid = np.array([0.0, 1.0, 2.0, 3.0])
    indices = _nearest(np.array([-5.0, 0.4, 0.6, 2.5001, 9.0]), grid)

    np.testing.assert_array_equal(indices, [0, 0, 1, 3, 3])


@pytest.mark.integration
def test_get_weights_mask_lakes() -> None:
    r"""Determines if the lakes observed by the satellites are land, the open ocean staying sea."""

    mask = _mask_dataarray("chl")
    weights = get_weights_mask("chl")[0]
    lat, lon = mask.latitude.values, mask.longitude.values

    def point(y: float, x: float) -> float:
        r"""Return the mask at the closest point of a latitude and longitude."""
        return weights[np.abs(lat - y).argmin(), np.abs(lon - x).argmin()].item()

    assert point(-15.8, -69.4) == 0.0  # Lake Titicaca
    assert point(-17.0, 27.8) == 0.0  # Lake Kariba
    assert point(-30.0, -120.0) == 1.0  # South Pacific


@pytest.mark.integration
def test_get_weights_mask_artefacts() -> None:
    r"""Determines if the artefact columns of GLORYS are land at every level, the other sea points staying sea."""

    artefacts = load_npy(PATH_ARTEFACTS["glorys"])
    mask = get_weights_mask("glorys")[0]

    assert artefacts is not None and artefacts.shape == mask.shape[1:] and artefacts.any()
    assert (mask[:, torch.from_numpy(artefacts)] == 0).all()
    assert mask[0].sum() > 0.9 * (_mask_dataarray("glorys").values[0] > 0).sum()
