"""Shared fixtures: GPU tests are skipped automatically without CUDA."""

from __future__ import annotations

import pytest


def _has_cuda() -> bool:
    try:
        from pyspectral.backends._cuda_env import cuda_available

        return cuda_available()
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _has_cuda():
        return
    skip = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
