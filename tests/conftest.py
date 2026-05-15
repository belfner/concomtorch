"""
Shared fixtures and the GPU gate guard for the in-container wheel test suite.

These tests run inside cibuildwheel's temporary test cwd against the installed
repaired wheel. When the in-container gate runs, CIBW_TEST_ENVIRONMENT_LINUX
sets CONCOMTORCH_REQUIRE_GPU=1, which turns "no usable CUDA device" into a
hard failure so a mis-provisioned runner cannot silently pass the gate with
zero GPU tests. Local developer runs without that variable skip GPU tests
instead.
"""

from __future__ import annotations

import os

import pytest

try:
    import torch
except ImportError:
    torch = None


def _cuda_available() -> bool:
    """
    Return whether a usable CUDA device is visible to torch.

    Returns
    -------
    bool
        True if torch imported and reports an available CUDA device.
    """
    return torch is not None and torch.cuda.is_available()


def _require_gpu() -> bool:
    """
    Return whether the run demands a GPU (the in-container gate sets this).

    Returns
    -------
    bool
        True if CONCOMTORCH_REQUIRE_GPU is exactly "1".
    """
    return os.environ.get("CONCOMTORCH_REQUIRE_GPU", "") == "1"


def pytest_configure(config: pytest.Config) -> None:
    """
    Hard-fail the session when a GPU is required but unavailable.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration object.

    Raises
    ------
    pytest.UsageError
        When CONCOMTORCH_REQUIRE_GPU=1 and no usable CUDA device is visible.
        Raised at configure time so the whole run aborts with a non-zero exit
        before any wheel can be exported, rather than skipping silently.
    """
    if _require_gpu() and not _cuda_available():
        torch_state = "import failed" if torch is None else (f"torch.cuda.is_available()={torch.cuda.is_available()}")
        raise pytest.UsageError(
            "CONCOMTORCH_REQUIRE_GPU=1 but no usable CUDA device is available "
            f"({torch_state}). Refusing to verify the wheel without a GPU."
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register the ``--run-slow`` opt-in for the scale/stress suite.

    Parameters
    ----------
    parser : pytest.Parser
        The pytest command-line parser.
    """
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked slow (large/odd images, atomic-union stress)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Deselect slow tests unless opted in, then skip GPU tests locally.

    Slow handling is independent of CUDA. The GPU skip is only reached
    when a GPU is not required (a required-but-absent GPU already aborted
    in pytest_configure), so skipping here is safe for local development.

    Parameters
    ----------
    config : pytest.Config
        The active pytest configuration object.
    items : list[pytest.Item]
        Collected test items, modified in place.
    """
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="slow test; pass --run-slow to enable")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device; set CONCOMTORCH_REQUIRE_GPU=1 to make this a hard failure")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def cuda_device():
    """
    Provide a CUDA torch.device, skipping the test if none is available.

    Returns
    -------
    torch.device
        A device object bound to the default CUDA device.
    """
    if not _cuda_available():
        pytest.skip("no CUDA device available")
    return torch.device("cuda")
