"""
Differential correctness vs external oracles (design section 1.1).

Every image is labeled by the extension and compared, as a
label-permutation-invariant partition, against ``scipy.ndimage.label``
(full 8-connectivity structuring element) and, as an independent
cross-check, ``cc3d`` at 8-connectivity. Both oracles are
importorskip-guarded so the suite degrades when they are absent; in the
in-container gate the ``test`` extra installs them so these run.

Covers the parity classes and a density sweep at small and medium
(deliberately odd) sizes. Both algorithms are asserted against the
oracle independently, so a bug shared by ``bke`` and ``bke_ic`` is not
masked by only comparing them to each other.
"""

from __future__ import annotations

import os

import pytest

import _ccl_lib

if os.environ.get("CONCOMTORCH_REQUIRE_GPU", "") == "1":
    import concomtorch  # noqa: F401
else:
    concomtorch = pytest.importorskip("concomtorch")

pytestmark = pytest.mark.gpu

_PARITY_SHAPES = [(16, 16), (17, 16), (16, 17), (17, 17), (1, 9), (9, 1)]
_DENSITIES = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
_SWEEP_SHAPES = [(17, 17), (64, 65), (256, 257)]
_KINDS = ["bernoulli", "rectangles", "walk", "dilated"]


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("dtype", ["uint8", "bool"])
@pytest.mark.parametrize("shape", _PARITY_SHAPES, ids=[f"{h}x{w}" for h, w in _PARITY_SHAPES])
def test_parity_classes_match_scipy(cuda_device, shape, dtype, algorithm) -> None:
    """
    All four H/W parity classes agree with the scipy oracle.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    shape : tuple[int, int]
        Image shape spanning the parity classes.
    dtype : str
        Input tensor dtype.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image(shape, 0.45, "bernoulli", seed=12345)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm, dtype=dtype)
    expected = _ccl_lib.scipy_label(image)
    _ccl_lib.assert_same_partition(produced, expected, context=f"scipy {shape} {dtype} {algorithm}")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("shape", _SWEEP_SHAPES, ids=[f"{h}x{w}" for h, w in _SWEEP_SHAPES])
@pytest.mark.parametrize("density", _DENSITIES, ids=[str(d) for d in _DENSITIES])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_density_sweep_match_scipy(cuda_device, density, shape, seed, algorithm) -> None:
    """
    Bernoulli density sweep at small/medium sizes agrees with scipy.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    density : float
        Foreground fraction.
    shape : tuple[int, int]
        Image shape (medium sizes are deliberately odd).
    seed : int
        RNG seed (in the test id for reproducibility).
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image(shape, density, "bernoulli", seed=seed)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    expected = _ccl_lib.scipy_label(image)
    _ccl_lib.assert_same_partition(produced, expected, context=f"scipy {shape} d={density} seed={seed} {algorithm}")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("seed", [7, 8, 9])
def test_structured_generators_match_cc3d(cuda_device, kind, seed, algorithm) -> None:
    """
    Structured generators cross-checked against the cc3d oracle.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    kind : str
        Structured generator kind.
    seed : int
        RNG seed.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((96, 97), 0.3, kind, seed=seed)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    expected = _ccl_lib.cc3d_label(image)
    _ccl_lib.assert_same_partition(produced, expected, context=f"cc3d {kind} seed={seed} {algorithm}")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
def test_scipy_and_cc3d_agree_with_extension(cuda_device, algorithm) -> None:
    """
    Triangulate: extension, scipy, and cc3d induce the same partition on
    a single nontrivial image (localizes oracle-usage mistakes).

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture; skips when no GPU is available.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    image = _ccl_lib.random_image((128, 129), 0.4, "dilated", seed=4242)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    _ccl_lib.assert_same_partition(produced, _ccl_lib.scipy_label(image), context="vs scipy")
    _ccl_lib.assert_same_partition(produced, _ccl_lib.cc3d_label(image), context="vs cc3d")
