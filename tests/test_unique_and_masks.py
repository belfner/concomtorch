"""
``get_unique_labels`` / ``get_component_masks`` contract (design section 4).

These exercise the post-labeling helpers on real CCL output (labels are
raster-index ``+ 1``, not a dense ``1..N`` range) and on adversarial
non-CCL arrays, the ``collapse_consecutive`` fast-path equivalence,
``unique_labels`` overrides, empty inputs, validation, and mask-stack
round-trip reconstruction.
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


def _label_on_gpu(image: np.ndarray, algorithm: str = "bke_ic") -> torch.Tensor:
    """
    Label an image and return the int32 CUDA labels tensor.

    Parameters
    ----------
    image : np.ndarray
        2D array; nonzero is foreground.
    algorithm : str, optional
        ``"bke_ic"`` (default) or ``"bke"``.

    Returns
    -------
    torch.Tensor
        int32 CUDA labels of shape ``image.shape``.
    """
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()
    return concomtorch.connected_components(tensor, algorithm=algorithm)


def test_unique_labels_matches_true_set(cuda_device) -> None:
    """
    Returned values equal the true nonzero label set, sorted, int32, CUDA.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((40, 41), 0.2, "bernoulli", seed=11)
    labels = _label_on_gpu(image)
    unique = concomtorch.get_unique_labels(labels)

    assert unique.dtype == torch.int32
    assert unique.is_cuda
    np_labels = labels.cpu().numpy()
    expected = np.unique(np_labels[np_labels != 0])
    np.testing.assert_array_equal(unique.cpu().numpy(), expected)
    assert np.all(np.diff(unique.cpu().numpy()) > 0)


def test_unique_exclude_background_toggle(cuda_device) -> None:
    """
    ``exclude_background=False`` keeps label 0 when background exists;
    the default drops it.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((24, 25), 0.3, "bernoulli", seed=12)
    labels = _label_on_gpu(image)

    without_bg = concomtorch.get_unique_labels(labels, exclude_background=True).cpu().numpy()
    with_bg = concomtorch.get_unique_labels(labels, exclude_background=False).cpu().numpy()

    assert 0 not in without_bg
    assert with_bg[0] == 0
    np.testing.assert_array_equal(with_bg[1:], without_bg)


@pytest.mark.parametrize("exclude_background", [True, False])
def test_collapse_consecutive_equivalent_on_ccl(cuda_device, exclude_background) -> None:
    """
    ``collapse_consecutive`` does not change the unique set on CCL output.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    exclude_background : bool
        Whether background is excluded.
    """
    image = _ccl_lib.random_image((50, 51), 0.25, "rectangles", seed=13)
    labels = _label_on_gpu(image)
    fast = concomtorch.get_unique_labels(labels, exclude_background, True).cpu().numpy()
    slow = concomtorch.get_unique_labels(labels, exclude_background, False).cpu().numpy()
    np.testing.assert_array_equal(fast, slow)


def test_collapse_consecutive_equivalent_on_shuffled_array(cuda_device) -> None:
    """
    On a shuffled (non-CCL) int32 array where consecutive-collapse gives
    no speedup, both paths still return the same set.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    rng = np.random.default_rng(14)
    arr = rng.integers(0, 7, size=(32, 33), dtype=np.int32)
    labels = torch.from_numpy(arr).cuda()
    fast = concomtorch.get_unique_labels(labels, True, True).cpu().numpy()
    slow = concomtorch.get_unique_labels(labels, True, False).cpu().numpy()
    np.testing.assert_array_equal(fast, slow)
    np.testing.assert_array_equal(slow, np.unique(arr[arr != 0]))


def test_unique_all_background_is_empty(cuda_device) -> None:
    """
    An all-background labeling yields a length-0 int32 tensor.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    labels = _label_on_gpu(np.zeros((16, 16), dtype=np.uint8))
    unique = concomtorch.get_unique_labels(labels)
    assert unique.numel() == 0
    assert unique.dtype == torch.int32


def test_unique_labels_validation(cuda_device) -> None:
    """
    ``get_unique_labels`` rejects non-CUDA, non-int32, and non-2D inputs.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    with pytest.raises(RuntimeError, match="CUDA"):
        concomtorch.get_unique_labels(torch.zeros((4, 4), dtype=torch.int32))
    with pytest.raises(ValueError, match="int32"):
        concomtorch.get_unique_labels(torch.zeros((4, 4), dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="2D"):
        concomtorch.get_unique_labels(torch.zeros((4, 4, 2), dtype=torch.int32, device="cuda"))


def test_component_masks_partition_foreground(cuda_device) -> None:
    """
    Auto-mode masks are disjoint, uint8, and union to the foreground;
    ordering follows ascending unique-label order.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((40, 41), 0.3, "rectangles", seed=15)
    labels = _label_on_gpu(image)
    masks = concomtorch.get_component_masks(labels)
    unique = concomtorch.get_unique_labels(labels).cpu().numpy()
    np_labels = labels.cpu().numpy()

    assert masks.dtype == torch.uint8
    assert masks.shape == (len(unique), *image.shape)
    masks_np = masks.cpu().numpy()

    union = np.zeros(image.shape, dtype=np.int64)
    for i, lab in enumerate(unique):
        np.testing.assert_array_equal(masks_np[i].astype(bool), np_labels == lab)
        union += masks_np[i]
    assert union.max() <= 1
    np.testing.assert_array_equal(union.astype(bool), np_labels != 0)


def test_component_masks_include_background(cuda_device) -> None:
    """
    ``exclude_background=False`` adds the background as the first mask.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((24, 25), 0.4, "bernoulli", seed=16)
    labels = _label_on_gpu(image)
    np_labels = labels.cpu().numpy()

    with_bg = concomtorch.get_component_masks(labels, exclude_background=False).cpu().numpy()
    without_bg = concomtorch.get_component_masks(labels, exclude_background=True).cpu().numpy()

    assert with_bg.shape[0] == without_bg.shape[0] + 1
    np.testing.assert_array_equal(with_bg[0].astype(bool), np_labels == 0)


@pytest.mark.parametrize("exclude_background", [True, False])
def test_component_masks_collapse_equivalent(cuda_device, exclude_background) -> None:
    """
    ``collapse_consecutive`` does not change the mask stack.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    exclude_background : bool
        Whether background is excluded.
    """
    image = _ccl_lib.random_image((36, 37), 0.3, "dilated", seed=17)
    labels = _label_on_gpu(image)
    fast = concomtorch.get_component_masks(labels, None, exclude_background, True).cpu().numpy()
    slow = concomtorch.get_component_masks(labels, None, exclude_background, False).cpu().numpy()
    np.testing.assert_array_equal(fast, slow)


def test_component_masks_provided_unique_labels_order(cuda_device) -> None:
    """
    Provided ``unique_labels`` selects exactly those components in the
    given order; a nonexistent label yields an all-zero mask.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((40, 41), 0.25, "rectangles", seed=18)
    labels = _label_on_gpu(image)
    np_labels = labels.cpu().numpy()
    present = concomtorch.get_unique_labels(labels).cpu().numpy()
    assert present.size >= 2

    missing = int(np_labels.max()) + 12345
    requested = np.array([int(present[1]), int(present[0]), missing], dtype=np.int32)
    sel = torch.from_numpy(requested).cuda()
    masks = concomtorch.get_component_masks(labels, sel).cpu().numpy()

    assert masks.shape == (3, *image.shape)
    np.testing.assert_array_equal(masks[0].astype(bool), np_labels == present[1])
    np.testing.assert_array_equal(masks[1].astype(bool), np_labels == present[0])
    assert masks[2].sum() == 0


def test_component_masks_empty_input(cuda_device) -> None:
    """
    An all-background image yields a ``(0, H, W)`` uint8 stack.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    labels = _label_on_gpu(np.zeros((12, 13), dtype=np.uint8))
    masks = concomtorch.get_component_masks(labels)
    assert masks.shape == (0, 12, 13)
    assert masks.dtype == torch.uint8


def test_component_masks_validation(cuda_device) -> None:
    """
    ``get_component_masks`` rejects bad ``labels`` and bad
    ``unique_labels`` with the documented exception types.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    good = _label_on_gpu(_ccl_lib.random_image((16, 16), 0.3, "bernoulli", seed=19))

    with pytest.raises(RuntimeError, match="CUDA"):
        concomtorch.get_component_masks(torch.zeros((4, 4), dtype=torch.int32))
    with pytest.raises(ValueError, match="int32"):
        concomtorch.get_component_masks(torch.zeros((4, 4), dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="2D"):
        concomtorch.get_component_masks(torch.zeros((4, 4, 2), dtype=torch.int32, device="cuda"))

    with pytest.raises(ValueError, match="CUDA"):
        concomtorch.get_component_masks(good, torch.zeros((2,), dtype=torch.int32))
    with pytest.raises(ValueError, match="int32"):
        concomtorch.get_component_masks(good, torch.zeros((2,), dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="1D"):
        concomtorch.get_component_masks(good, torch.zeros((2, 2), dtype=torch.int32, device="cuda"))


def test_mask_stack_round_trip(cuda_device) -> None:
    """
    Reconstructing a labeling from the mask stack and its unique labels
    reproduces the original partition.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((48, 49), 0.3, "dilated", seed=20)
    labels = _label_on_gpu(image)
    unique = concomtorch.get_unique_labels(labels).cpu().numpy()
    masks = concomtorch.get_component_masks(labels).cpu().numpy()

    reconstructed = np.zeros(image.shape, dtype=np.int32)
    for i, lab in enumerate(unique):
        reconstructed[masks[i].astype(bool)] = int(lab)
    _ccl_lib.assert_same_partition(
        reconstructed,
        labels.cpu().numpy(),
        context="mask round-trip",
    )
