"""
Minimal end-to-end smoke check: label a tiny image on the GPU through the
installed wheel. Scope is loadability and dispatch only: it proves the
compiled CUDA extension loads and the public entry point returns a result
honoring the output contract. The exhaustive correctness suite is deferred
and tracked separately in test_correctness.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
concomtorch = pytest.importorskip("concomtorch")


@pytest.mark.gpu
def test_connected_components_smoke(cuda_device) -> None:
    """
    Label two adjacent foreground pixels and assert the output contract.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    """
    x = torch.zeros((4, 4), dtype=torch.uint8, device=cuda_device)
    x[1, 1] = 1
    x[1, 2] = 1

    labels = concomtorch.connected_components(x)

    assert labels.shape == (4, 4), f"unexpected shape {tuple(labels.shape)}"
    assert labels.dtype == torch.int32, f"unexpected dtype {labels.dtype}"
    assert labels.device.type == "cuda", f"unexpected device {labels.device}"
    assert labels[0, 0].item() == 0, "background pixel must be label 0"
    assert labels[1, 1].item() != 0, "foreground pixel must be non-zero"
    assert labels[1, 1].item() == labels[1, 2].item(), "adjacent foreground pixels must share a component label"
