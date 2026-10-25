r"""Tests for heimdall.model.encoders."""

import pytest
import torch

from omegaconf import OmegaConf
from pathlib import Path
from shaggy.tools import load_config, load_weights, save

from heimdall.data import DATASET_SHAPES
from heimdall.model.encoders import Encoder, get_domain

CONFIG = {"hid_channels": [4, 4, 4], "hid_blocks": [1, 1, 1], "kernel_size": 3, "stride": 2}


@pytest.mark.integration
def test_encoder_typical(tmp_path: Path) -> None:
    r"""Determines if an encoder gives its latent grid, and is saved then loaded with its domain."""

    model = Encoder("wind_asc", lat_channels=8, **CONFIG)
    x = torch.randn(2, 2 * 2 + 2, *DATASET_SHAPES["wind_asc"])

    assert model(x).shape == (2, 8, *model.shape)
    assert model.shape == [n // 4 for n in DATASET_SHAPES["wind_asc"]]

    config = {"modality": "wind_asc", "lat_channels": 8, **CONFIG, "domain": model.domain}
    save(model, config, tmp_path)
    loaded = load_weights(
        Encoder(**OmegaConf.to_container(load_config(tmp_path))), tmp_path, device="cpu"
    )

    assert torch.equal(loaded(x), model.eval()(x))


@pytest.mark.integration
def test_encoder_other_domain() -> None:
    r"""Determines if an encoder trained on another domain is rejected."""

    domain = {**get_domain("sst"), "padding": 0}

    with pytest.raises(AssertionError, match="domain"):
        Encoder("sst", lat_channels=8, domain=domain, **CONFIG)
