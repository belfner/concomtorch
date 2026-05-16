"""
Deterministic fixed-fixture correctness anchors (design section 2).

Every catalogued image is labeled by the extension and compared, as a
label-permutation-invariant partition, against the dependency-free
``reference_label`` ground truth. These run without scipy/cc3d and pin
down the block-boundary, parity, and bottom-right odd/odd corner
behaviors of the block labeler. A still-broken case
fails loudly here; nothing is xfailed or soft-passed.
"""

from __future__ import annotations

import os

import pytest

import _ccl_lib

if os.environ.get("CONCOMTORCH_REQUIRE_GPU", "") == "1":
    # Gate mode: the installed repaired wheel MUST import. A plain import
    # makes an ImportError a collection error so the in-container gate
    # exits non-zero rather than skipping silently.
    import concomtorch  # noqa: F401
else:
    concomtorch = pytest.importorskip("concomtorch")

pytestmark = pytest.mark.gpu

_FIXTURES = list(_ccl_lib.iter_fixed_images())


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("dtype", ["uint8", "bool"])
@pytest.mark.parametrize(
    "image",
    [img for _, img in _FIXTURES],
    ids=[name for name, _ in _FIXTURES],
)
def test_fixed_fixture_matches_reference(cuda_device, image, dtype, algorithm) -> None:
    """
    Assert the extension's partition matches the BFS reference labeler.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    image : np.ndarray
        A catalogued deterministic image.
    dtype : str
        Input tensor dtype, ``"uint8"`` or ``"bool"``.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    produced = _ccl_lib.run_cc(image, algorithm=algorithm, dtype=dtype)
    expected = _ccl_lib.reference_label(image)
    _ccl_lib.assert_same_partition(
        produced,
        expected,
        context=f"algorithm={algorithm} dtype={dtype} shape={image.shape}",
    )


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_all_2x2_block_patterns(cuda_device, algorithm) -> None:
    """
    Exhaustively check all 16 single-block 2x2 foreground patterns.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    import numpy as np

    for pattern in range(16):
        image = np.array(
            [
                [pattern & 1, (pattern >> 1) & 1],
                [(pattern >> 2) & 1, (pattern >> 3) & 1],
            ],
            dtype=np.uint8,
        )
        produced = _ccl_lib.run_cc(image, algorithm=algorithm)
        expected = _ccl_lib.reference_label(image)
        _ccl_lib.assert_same_partition(produced, expected, context=f"2x2 pattern {pattern:04b}")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_bottom_right_oddodd_corner_not_dropped(cuda_device, algorithm) -> None:
    """
    A lone foreground pixel at ``(H-1, W-1)`` of an
    odd-by-odd image must be exactly one component, not dropped to 0.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    import numpy as np

    for n in (1, 3, 5, 7, 9):
        image = np.zeros((n, n), dtype=np.uint8)
        image[n - 1, n - 1] = 1
        produced = _ccl_lib.run_cc(image, algorithm=algorithm)
        assert produced[n - 1, n - 1] != 0, f"{n}x{n} bottom-right corner dropped"
        assert int((produced != 0).sum()) == 1, f"{n}x{n} br corner produced extra labels"
        _ccl_lib.assert_same_partition(produced, _ccl_lib.reference_label(image), context=f"br {n}x{n}")
