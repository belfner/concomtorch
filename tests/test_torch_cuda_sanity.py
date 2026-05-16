"""
Sanity checks that the test venv received the exact CUDA torch the wheel was
built against. Catches a future resolver surprise (e.g. a CPU torch sneaking
in) before the correctness suite runs.
"""

from __future__ import annotations

import os

import pytest
from packaging.version import Version

torch = pytest.importorskip("torch")


def test_torch_version_matches_expected() -> None:
    """
    Assert the installed torch shares the expected ``X.Y`` minor.

    The wheel pins ``torch==X.Y.*``, so any patch of the built minor is a
    valid install. CONCOMTORCH_EXPECTED_TORCH carries the build-time
    version; its ``(major, minor)`` must equal the installed torch's.
    """
    expected = os.environ.get("CONCOMTORCH_EXPECTED_TORCH")
    if expected is None:
        pytest.skip("CONCOMTORCH_EXPECTED_TORCH not set (local run)")
    installed_minor = Version(torch.__version__.split("+")[0]).release[:2]
    expected_minor = Version(expected.split("+")[0]).release[:2]
    assert installed_minor == expected_minor, (
        f"torch {torch.__version__} does not match expected minor from {expected}"
    )


def test_torch_cuda_build_matches_expected() -> None:
    """
    Assert torch.version.cuda matches the CONCOMTORCH_EXPECTED_CUDA channel.
    """
    expected_cuda = os.environ.get("CONCOMTORCH_EXPECTED_CUDA")
    if expected_cuda is None:
        pytest.skip("CONCOMTORCH_EXPECTED_CUDA not set (local run)")
    assert torch.version.cuda is not None, "torch was built without CUDA support"
    digits = expected_cuda.lower().replace("cu", "")
    assert len(digits) >= 2, f"unrecognized CUDA tag: {expected_cuda}"
    normalized = f"{digits[:-1]}.{digits[-1]}"
    assert torch.version.cuda == normalized, (
        f"torch.version.cuda={torch.version.cuda} does not match expected {normalized} (from {expected_cuda})"
    )


@pytest.mark.gpu
def test_cuda_is_available() -> None:
    """
    Assert torch reports a usable CUDA device.
    """
    assert torch.cuda.is_available(), "no CUDA device available to torch"
