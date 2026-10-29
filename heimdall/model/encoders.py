r"""Deterministic encoders of the GLORYS states and of the surface observations into latent grids."""

__all__ = [
    "Encoder",
]

import math
import torch
import torch.nn as nn

from shaggy.models.cae import ConvEncoder
from torch import Tensor
from typing import Any

from heimdall.data import DATASET_PADDING, DATASET_REGION, DATASET_SHAPES, DATASET_VARIABLES
from heimdall.data.tools import coarsened_shape
from heimdall.data.weights import get_weights_mesh


def get_domain(modality: str) -> dict[str, list[int] | int]:
    r"""Return the domain of a modality, its region and padding as plain values (saved with the encoder).

    Arguments:
        modality : Name of the modality, a key of DATASET_VARIABLES.

    Returns:
        domain : Indices [start, stop] of each axis of DATASET_REGION, and the padding of DATASET_PADDING.
    """

    region = {axis: [s.start, s.stop] for axis, s in DATASET_REGION[modality].items()}
    return {**region, "padding": DATASET_PADDING[modality]}


class Encoder(nn.Module):
    r"""Creates the deterministic encoder of a modality, its fields and date stacked with the mesh of its grid.

    Arguments:
        modality     : Name of the modality, a key of DATASET_VARIABLES.
        lat_channels : Number of latent channels C_z.
        coarsening   : Number of points averaged along the latitude and along the longitude (1 keeps the grid).
        domain       : Domain the encoder was trained on (from its saved configuration), None for a new encoder.
        config       : Keyword arguments of the ConvEncoder (hid_channels, stride, ...).
    """

    def __init__(
        self,
        modality: str,
        lat_channels: int,
        coarsening: int = 1,
        domain: dict[str, Any] | None = None,
        **config: Any,
    ) -> None:
        super().__init__()

        # The static inputs are rebuilt from DATASET_REGION and DATASET_PADDING, so the domain must not have changed
        valid = domain is None or dict(domain) == get_domain(modality)
        assert valid, (
            f"ERROR - Encoder trained on {domain}, the domain of {modality} is now {get_domain(modality)}."
        )

        self.modality = modality
        self.lat_channels = lat_channels
        self.coarsening = coarsening
        self.domain = get_domain(modality)

        # The sin/cos mesh is rebuilt at loading (not saved), in bfloat16 like the fields
        static = get_weights_mesh(modality, coarsening).bfloat16()
        self.register_buffer("static", static, persistent=False)

        # The inputs are the fields and the date, the network finding by itself the points without value (zeros)
        in_channels = len(DATASET_VARIABLES[modality]) + 2

        # The encoder is deterministic, without modulation of the normalization layers, nor pointwise mixing of
        # the inputs, the first convolution already projecting them on its channels
        shape = coarsened_shape(DATASET_SHAPES[modality], coarsening)
        self.encoder = ConvEncoder(
            in_channels + len(static),
            lat_channels,
            mix_channels=None,
            spatial=len(shape),
            mod_features=None,
            **config,
        )

        # Each spatial dimension of the latent grid is divided by the scale of the encoder
        self.shape = [n // s for n, s in zip(shape, self.encoder.scale, strict=True)]
        self.tokens = math.prod(self.shape)

    def forward(self, x: Tensor) -> Tensor:
        r"""Encode fields and their date into latent grids.

        Arguments:
            x : Fields and their date (B, C + 2, [Z,] Y, X).

        Returns:
            z : Latent grids (B, C_z, [Z',] Y', X').
        """

        return self.encoder(torch.cat([x, self.static.expand(len(x), *self.static.shape)], dim=1))
