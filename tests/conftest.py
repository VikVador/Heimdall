r"""Pytest configuration: integration mark and auto-skip in CI or when the datasets are unreachable."""

import os
import pytest
import s3fs

from pathlib import Path

from heimdall.config import DATA_SOURCE, OSN_STORAGE, PATH_DATASETS, PATH_OSN


def datasets_available() -> bool:
    r"""Check whether the datasets can be read: GLORYS on OSN (anonymously), and their copy when read on Torch."""

    if DATA_SOURCE == "torch" and not Path(PATH_OSN).exists():
        return False
    try:
        fs = s3fs.S3FileSystem(**OSN_STORAGE)
        return fs.exists(f"{PATH_DATASETS['glorys']}/zarr.json")
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    r"""Register the 'integration' marker for tests requiring the datasets."""

    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring the datasets on OSN (skipped in CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    r"""Auto-skip integration tests in CI (GitHub runners lack the memory) or if the datasets are unreachable."""

    if not any(item.get_closest_marker("integration") for item in items):
        return
    if "CI" not in os.environ and datasets_available():
        return
    skip_integration = pytest.mark.skip(reason="CI or datasets unreachable")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)
