"""
In-place buffer and helper-object contract (design section 3).

The labels buffer is used as scratch (the union-find info byte is packed
into it), so these tests prove the buffer is fully overwritten, that a
correctly sized buffer is returned by identity, that mis-specified
buffers are rejected, and that ``ConnectedComponentsLabeler`` /
``create_labels_buffer`` honor their documented contract.
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
def test_buffer_returned_by_identity_and_correct(cuda_device, algorithm) -> None:
    """
    A correctly sized buffer is returned as the same storage and the
    result equals the no-buffer result bitwise.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((40, 41), 0.4, "bernoulli", seed=1)
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()

    no_buffer = concomtorch.connected_components(tensor, algorithm=algorithm).cpu().numpy()

    buffer = concomtorch.create_labels_buffer(image.shape)
    returned = concomtorch.connected_components(tensor, labels=buffer, algorithm=algorithm)
    assert returned.data_ptr() == buffer.data_ptr()
    np.testing.assert_array_equal(returned.cpu().numpy(), no_buffer)


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_garbage_prefilled_buffer_fully_overwritten(cuda_device, algorithm) -> None:
    """
    A buffer pre-filled with a prior image's labels yields a fully
    correct result, proving no stale info-byte bleed-through.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    prev = _ccl_lib.random_image((50, 51), 0.6, "dilated", seed=2)
    curr = _ccl_lib.random_image((50, 51), 0.3, "rectangles", seed=3)

    buffer = concomtorch.create_labels_buffer(prev.shape)
    prev_tensor = torch.from_numpy((prev != 0).astype(np.uint8)).cuda()
    concomtorch.connected_components(prev_tensor, labels=buffer, algorithm=algorithm)

    curr_tensor = torch.from_numpy((curr != 0).astype(np.uint8)).cuda()
    produced = concomtorch.connected_components(curr_tensor, labels=buffer, algorithm=algorithm).cpu().numpy()
    _ccl_lib.assert_same_partition(
        produced,
        _ccl_lib.reference_label(curr),
        context=f"garbage-prefilled buffer {algorithm}",
    )


@pytest.mark.parametrize("bad_shape", [(33, 32), (32, 33), (32,)])
def test_wrong_shape_buffer_rejected(cuda_device, bad_shape) -> None:
    """
    A buffer whose shape differs from the input is rejected by the
    C++ ``TORCH_CHECK`` as a ``RuntimeError``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    bad_shape : tuple[int, ...]
        A buffer shape that does not match the (32, 32) input.
    """
    tensor = torch.zeros((32, 32), dtype=torch.uint8, device="cuda")
    buffer = torch.empty(bad_shape, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        concomtorch.connected_components(tensor, labels=buffer)


def test_non_contiguous_buffer_rejected(cuda_device) -> None:
    """
    A non-contiguous buffer is rejected rather than silently replaced.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    tensor = torch.zeros((16, 16), dtype=torch.uint8, device="cuda")
    buffer = torch.empty((16, 32), dtype=torch.int32, device="cuda")[:, ::2]
    assert not buffer.is_contiguous()
    with pytest.raises(RuntimeError):
        concomtorch.connected_components(tensor, labels=buffer)


@pytest.mark.parametrize("bad_dtype", [torch.int64, torch.float32])
def test_wrong_dtype_buffer_rejected(cuda_device, bad_dtype) -> None:
    """
    A non-int32 buffer raises Python-side ``ValueError``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    bad_dtype : torch.dtype
        A buffer dtype other than int32.
    """
    tensor = torch.zeros((8, 8), dtype=torch.uint8, device="cuda")
    buffer = torch.empty((8, 8), dtype=bad_dtype, device="cuda")
    with pytest.raises(ValueError, match="int32"):
        concomtorch.connected_components(tensor, labels=buffer)


def test_cpu_buffer_rejected(cuda_device) -> None:
    """
    A CPU buffer raises Python-side ``ValueError``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    tensor = torch.zeros((8, 8), dtype=torch.uint8, device="cuda")
    buffer = torch.empty((8, 8), dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        concomtorch.connected_components(tensor, labels=buffer)


def test_create_labels_buffer_contract(cuda_device) -> None:
    """
    ``create_labels_buffer`` returns an int32 CUDA tensor of the
    requested shape that ``connected_components`` accepts.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    buffer = concomtorch.create_labels_buffer((20, 21))
    assert buffer.shape == (20, 21)
    assert buffer.dtype == torch.int32
    assert buffer.is_cuda
    tensor = torch.zeros((20, 21), dtype=torch.uint8, device="cuda")
    out = concomtorch.connected_components(tensor, labels=buffer)
    assert out.data_ptr() == buffer.data_ptr()


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_labeler_reuse_and_repr(cuda_device, algorithm) -> None:
    """
    ``ConnectedComponentsLabeler`` reuses one buffer, honors the
    algorithm, gives identical repeated results, and has a sane repr.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((36, 37), 0.45, "bernoulli", seed=4)
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()

    labeler = concomtorch.ConnectedComponentsLabeler((36, 37), algorithm=algorithm)
    first = labeler(tensor).cpu().numpy()
    for _ in range(3):
        np.testing.assert_array_equal(labeler(tensor).cpu().numpy(), first)

    _ccl_lib.assert_same_partition(
        first,
        _ccl_lib.reference_label(image),
        context=f"labeler {algorithm}",
    )
    assert repr(labeler) == (
        f"ConnectedComponentsLabeler(image_size=(36, 37), device={torch.device('cuda')}, algorithm={algorithm!r})"
    )


def test_labeler_shape_mismatch_rejected(cuda_device) -> None:
    """
    A wrong input shape to the labeler raises ``ValueError``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    labeler = concomtorch.ConnectedComponentsLabeler((16, 16))
    wrong = torch.zeros((16, 17), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="does not match"):
        labeler(wrong)
