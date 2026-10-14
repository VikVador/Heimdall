r"""Tests for heimdall.tools and heimdall.data.tools."""

import pytest

from heimdall.data.tools import assert_date_format
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
