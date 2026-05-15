"""
Numerical / scale tests (design section 5).

Large and deliberately odd images, very many components, the
atomic-union contention path, repeated-run determinism at scale, and a
memory-bounded ``unique_labels`` subset. Module is ``slow``-marked and
opt-in via ``--run-slow`` so the default fast run skips it; the
in-container gate enables it.
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

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("shape", [(4096, 4096), (4097, 4097)], ids=["4096x4096", "4097x4097"])
def test_large_image_matches_cc3d(cuda_device, shape, algorithm) -> None:
    """
    Large even and odd-by-odd images agree with the cc3d oracle.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    shape : tuple[int, int]
        Large image shape.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image(shape, 0.45, "bernoulli", seed=2025)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    expected = _ccl_lib.cc3d_label(image)
    _ccl_lib.assert_same_partition(produced, expected, context=f"large {shape} {algorithm}")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_many_single_pixel_components(cuda_device, algorithm) -> None:
    """
    A regular grid of isolated pixels yields exactly that many distinct
    components and labels stay within the int32 contract range.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    height, width = 1024, 1025
    image = np.zeros((height, width), dtype=np.uint8)
    image[::2, ::2] = 1
    expected_count = image.sum()

    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    unique = np.unique(produced[produced != 0])
    assert unique.size == expected_count
    assert produced.max() <= height * width
    assert produced.dtype == np.int32
    _ccl_lib.assert_same_partition(
        produced,
        _ccl_lib.reference_label(image),
        context=f"many components {algorithm}",
    )


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_atomic_union_dense_single_component(cuda_device, algorithm) -> None:
    """
    A fully dense large image (heavy atomic-union contention on one
    root) is exactly one component, deterministically.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = np.ones((2048, 2048), dtype=np.uint8)
    tensor = torch.from_numpy(image).cuda()

    first = concomtorch.connected_components(tensor, algorithm=algorithm)
    fg = first != 0
    assert int(fg.sum().item()) == image.size
    assert int(torch.unique(first[fg]).numel()) == 1

    for _ in range(3):
        again = concomtorch.connected_components(tensor, algorithm=algorithm)
        assert torch.equal(again, first)


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_determinism_at_scale(cuda_device, algorithm) -> None:
    """
    Repeated runs on a large odd image are bitwise-identical for both
    fresh and reused buffers.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((2049, 2048), 0.5, "dilated", seed=7)
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()

    first = concomtorch.connected_components(tensor, algorithm=algorithm).clone()
    for _ in range(3):
        again = concomtorch.connected_components(tensor, algorithm=algorithm)
        assert torch.equal(again, first)

    buffer = concomtorch.create_labels_buffer(image.shape)
    for _ in range(3):
        reused = concomtorch.connected_components(tensor, labels=buffer, algorithm=algorithm)
        assert torch.equal(reused, first)


def test_component_masks_subset_bounds_memory(cuda_device) -> None:
    """
    Requesting a small ``unique_labels`` subset on an image with many
    components allocates only the requested slots.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    height, width = 512, 513
    image = np.zeros((height, width), dtype=np.uint8)
    image[::2, ::2] = 1
    labels = concomtorch.connected_components(torch.from_numpy(image).cuda())

    full_unique = concomtorch.get_unique_labels(labels)
    assert full_unique.numel() > 1000

    subset = full_unique[:8].contiguous()
    masks = concomtorch.get_component_masks(labels, subset)
    assert masks.shape == (8, height, width)

    np_labels = labels.cpu().numpy()
    masks_np = masks.cpu().numpy()
    for i, lab in enumerate(subset.cpu().numpy()):
        np.testing.assert_array_equal(masks_np[i].astype(bool), np_labels == lab)
