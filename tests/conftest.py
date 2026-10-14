r"""Pytest configuration: integration mark and auto-skip when the datasets are unavailable."""

import pytest

from heimdall.config import PATH_GLORYS, PATH_MASK, PATH_OBSERVATIONS, PATH_STATS


def pytest_configure(config: pytest.Config) -> None:
    r"""Register the 'integration' marker for tests requiring the datasets."""

    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring the datasets (skipped in CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    r"""Auto-skip integration tests if the datasets are unavailable."""

    paths = (PATH_GLORYS, PATH_MASK, PATH_STATS, *PATH_OBSERVATIONS.values())
    if all(path.exists() for path in paths):
        return
    skip_integration = pytest.mark.skip(reason="Datasets unavailable")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip_integration)
