"""
API-contract tests (design section 3): input handling and dispatch.

Covers dtype handling (uint8 / bool / arbitrary nonzero foreground),
non-contiguous input, shape and device rejection, and algorithm
selection. Every correctness assertion routes through the
partition-invariant comparator; the negative cases assert the rejection
the public API actually performs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import _ccl_lib

if os.environ.get("CONCOMTORCH_REQUIRE_GPU", "") == "1":
    import concomtorch
else:
    concomtorch = pytest.importorskip("concomtorch")

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_uint8_and_bool_same_partition(cuda_device, algorithm) -> None:
    """
    ``uint8`` and ``bool`` inputs induce the same partition.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((48, 49), 0.35, "bernoulli", seed=314)
    as_u8 = _ccl_lib.run_cc(image, algorithm=algorithm, dtype="uint8")
    as_bool = _ccl_lib.run_cc(image, algorithm=algorithm, dtype="bool")
    _ccl_lib.assert_same_partition(as_u8, as_bool, context=f"uint8 vs bool {algorithm}")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("fg_value", [1, 2, 17, 255])
def test_arbitrary_nonzero_is_foreground(cuda_device, fg_value, algorithm) -> None:
    """
    Any nonzero uint8 value is foreground (kernel uses ``> 0``).

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    fg_value : int
        The uint8 value written into foreground pixels.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    mask = _ccl_lib.random_image((40, 41), 0.4, "rectangles", seed=99)
    raw = (mask != 0).astype(np.uint8) * np.uint8(fg_value)
    tensor = torch.from_numpy(raw).cuda()
    produced = concomtorch.connected_components(tensor, algorithm=algorithm).cpu().numpy()
    _ccl_lib.assert_same_partition(
        produced,
        _ccl_lib.reference_label(mask),
        context=f"fg_value={fg_value} {algorithm}",
    )


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_non_contiguous_input_matches_logical_image(cuda_device, algorithm) -> None:
    """
    A transposed (non-contiguous) view labels its logical image correctly.

    The wrapper calls ``.contiguous()`` internally, so the result must
    match the oracle on the logical (transposed) array.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    base = _ccl_lib.random_image((33, 28), 0.4, "dilated", seed=77)
    logical = base.T
    view = torch.from_numpy((base != 0).astype(np.uint8)).cuda().t()
    assert not view.is_contiguous()
    produced = concomtorch.connected_components(view, algorithm=algorithm).cpu().numpy()
    _ccl_lib.assert_same_partition(
        produced,
        _ccl_lib.reference_label(logical),
        context=f"transposed view {algorithm}",
    )


def test_cpu_input_rejected(cuda_device) -> None:
    """
    A CPU input tensor raises ``RuntimeError`` mentioning the CUDA device.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    cpu_tensor = torch.zeros((8, 8), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="CUDA"):
        concomtorch.connected_components(cpu_tensor)


@pytest.mark.parametrize("bad_shape", [(8,), (2, 3, 4)])
def test_non_2d_input_rejected(cuda_device, bad_shape) -> None:
    """
    1D and 3D inputs raise ``RuntimeError`` from the C++ ``TORCH_CHECK``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    bad_shape : tuple[int, ...]
        A non-2D shape.
    """
    tensor = torch.zeros(bad_shape, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError):
        concomtorch.connected_components(tensor)


@pytest.mark.parametrize("bad_dtype", [torch.int32, torch.float32])
def test_wrong_input_dtype_rejected(cuda_device, bad_dtype) -> None:
    """
    ``int32`` / ``float32`` inputs raise ``RuntimeError`` from the C++
    ``TORCH_CHECK``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    bad_dtype : torch.dtype
        An unsupported input dtype.
    """
    tensor = torch.zeros((8, 8), dtype=bad_dtype, device="cuda")
    with pytest.raises(RuntimeError):
        concomtorch.connected_components(tensor)


def test_invalid_algorithm_rejected(cuda_device) -> None:
    """
    An unknown ``algorithm`` raises ``ValueError`` naming the bad value.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    tensor = torch.zeros((8, 8), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="nope"):
        concomtorch.connected_components(tensor, algorithm="nope")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_input_not_mutated(cuda_device, algorithm) -> None:
    """
    Labeling does not mutate the input tensor (it is not the buffer).

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((24, 25), 0.5, "bernoulli", seed=5)
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()
    before = tensor.clone()
    concomtorch.connected_components(tensor, algorithm=algorithm)
    assert torch.equal(tensor, before)
