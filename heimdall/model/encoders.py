r"""Deterministic encoders of the GLORYS states and of the surface observations into latent grids."""

__all__ = [
    "LatentEncoder",
    "StateEncoder",
    "ObservationEncoder",
]

import math
import torch
import torch.nn as nn

from collections.abc import Sequence
from shaggy.models.cae import ConvEncoder
from torch import Tensor
from typing import Any

from heimdall.data import DATASET_OBSERVATIONS, DATASET_VARIABLES_GLORYS, X, Y, Z
from heimdall.data.weights import get_weights_mask, get_weights_mesh


class LatentEncoder(nn.Module):
    r"""Creates a deterministic encoder of fields and their conditioning, stacked with static inputs, into a latent grid.

    Arguments:
        static       : Land/sea mask and sin/cos mesh (1 + C_m, L_1, ..., L_S).
        in_channels  : Number of channels of the fields and their conditioning C_i.
        resolution   : Spatial shape of the fields (L_1, ..., L_S).
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, ...).
    """

    def __init__(
        self,
        static: Tensor,
        in_channels: int,
        resolution: Sequence[int],
        lat_channels: int,
        **config: Any,
    ) -> None:
        super().__init__()

        self.lat_channels = lat_channels

        # Static inputs | Rebuilt at loading (not saved)
        self.register_buffer("static", static, persistent=False)

        # Deterministic | No modulation of the normalization layers
        self.encoder = ConvEncoder(
            in_channels + len(static),
            lat_channels,
            spatial=len(resolution),
            mod_features=None,
            **config,
        )

        # Latent grid | Each spatial dimension divided by the scale of the encoder
        self.shape = [n // s for n, s in zip(resolution, self.encoder.scale, strict=True)]
        self.tokens = math.prod(self.shape)

    def forward(self, x: Tensor) -> Tensor:
        r"""Encode fields and their conditioning into latent grids.

        Arguments:
            x : Fields and their conditioning (B, C_i, L_1, ..., L_S).

        Returns:
            z : Latent grids (B, C_z, L_1', ..., L_S').
        """
        return self.encoder(torch.cat([x, self.static.expand(len(x), *self.static.shape)], dim=1))


class StateEncoder(LatentEncoder):
    r"""Creates the deterministic encoder of the GLORYS states, conditioned on the date.

    Arguments:
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, ...).
    """

    def __init__(
        self,
        lat_channels: int,
        **config: Any,
    ) -> None:
        (_, mask), (_, mesh) = get_weights_mask(), get_weights_mesh()
        channels = len(DATASET_VARIABLES_GLORYS) + 2

        super().__init__(torch.cat([mask, mesh]), channels, (Z, Y, X), lat_channels, **config)


class ObservationEncoder(LatentEncoder):
    r"""Creates the deterministic encoder of an observation product, with its observed points, conditioned on the date.

    Arguments:
        product      : Name of the observation product, a key of DATASET_OBSERVATIONS.
        lat_channels : Number of latent channels C_z.
        config       : Keyword arguments of the ConvEncoder (hid_channels, ...).
    """

    def __init__(
        self,
        product: str,
        lat_channels: int,
        **config: Any,
    ) -> None:
        (mask, _), (mesh, _) = get_weights_mask(), get_weights_mesh()
        channels = 2 * len(DATASET_OBSERVATIONS[product]) + 2

        super().__init__(torch.cat([mask, mesh]), channels, (Y, X), lat_channels, **config)

        self.product = product
