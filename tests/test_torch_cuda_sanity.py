"""
Sanity checks that the test venv received the exact CUDA torch the wheel was
built against. Catches a future resolver surprise (e.g. a CPU torch sneaking
in) before the correctness suite runs.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")


def test_torch_version_matches_expected() -> None:
    """
    Assert the installed torch base version matches CONCOMTORCH_EXPECTED_TORCH.
    """
    expected = os.environ.get("CONCOMTORCH_EXPECTED_TORCH")
    if expected is None:
        pytest.skip("CONCOMTORCH_EXPECTED_TORCH not set (local run)")
    installed_base = torch.__version__.split("+")[0]
    expected_base = expected.split("+")[0]
    assert installed_base.startswith(expected_base) or expected_base.startswith(installed_base), (
        f"torch {torch.__version__} does not match expected {expected}"
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
