"""
Oracle-free property / invariant tests (design section 1.2).

These hold for any valid binary input and need no scipy/cc3d, so they
run even where the oracle is absent. Randomized inputs come from seeded
NumPy RNG sweeps (deterministic; the seed and shape are in the test id).
"""

from __future__ import annotations

import os
from collections import deque

import numpy as np
import pytest

import _ccl_lib

if os.environ.get("CONCOMTORCH_REQUIRE_GPU", "") == "1":
    import concomtorch
else:
    concomtorch = pytest.importorskip("concomtorch")

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu

_CASES = [
    (shape, density, kind, seed)
    for shape in [(16, 16), (17, 19), (33, 32), (64, 65)]
    for density in (0.05, 0.5, 0.9)
    for kind in ("bernoulli", "rectangles", "dilated")
    for seed in (0, 1)
]
_IDS = [f"{h}x{w}-d{d}-{k}-s{s}" for (h, w), d, k, s in _CASES]

_NEIGHBORS_8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)]


def _assert_connectivity_sound(labels: np.ndarray) -> None:
    """
    Directly verify no under-merge and no over-merge in a labeling.

    Under-merge: two 8-adjacent foreground pixels carry different
    labels. Over-merge: pixels sharing one label form more than one
    8-connected piece.

    Parameters
    ----------
    labels : np.ndarray
        int32 label array (background 0).

    Raises
    ------
    AssertionError
        On the first structural violation found.
    """
    height, width = labels.shape
    for y in range(height):
        for x in range(width):
            lab = labels[y, x]
            if lab == 0:
                continue
            for dy, dx in _NEIGHBORS_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    other = labels[ny, nx]
                    if other != 0 and other != lab:
                        raise AssertionError(f"under-merge: adjacent ({y},{x})={lab} and ({ny},{nx})={other}")

    for lab in np.unique(labels):
        if lab == 0:
            continue
        coords = np.argwhere(labels == lab)
        coord_set = {(int(y), int(x)) for y, x in coords}
        start = (int(coords[0][0]), int(coords[0][1]))
        seen = {start}
        queue = deque([start])
        while len(queue) > 0:
            y, x = queue.popleft()
            for dy, dx in _NEIGHBORS_8:
                nb = (y + dy, x + dx)
                if nb in coord_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(seen) != len(coord_set):
            raise AssertionError(
                f"over-merge: label {lab} spans {len(coord_set)} pixels but only "
                f"{len(seen)} are 8-connected to the first"
            )


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("dtype", ["uint8", "bool"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_background_invariance(cuda_device, case, dtype, algorithm) -> None:
    """
    Output is nonzero exactly where input is nonzero.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    case : tuple
        ``(shape, density, kind, seed)``.
    dtype : str
        Input tensor dtype.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    shape, density, kind, seed = case
    image = _ccl_lib.random_image(shape, density, kind, seed)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm, dtype=dtype)
    np.testing.assert_array_equal(produced != 0, image != 0)


@pytest.mark.parametrize("dtype", ["uint8", "bool"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_bke_and_bke_ic_same_partition(cuda_device, case, dtype) -> None:
    """
    ``bke`` and ``bke_ic`` induce the same partition for one input.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    case : tuple
        ``(shape, density, kind, seed)``.
    dtype : str
        Input tensor dtype.
    """
    shape, density, kind, seed = case
    image = _ccl_lib.random_image(shape, density, kind, seed)
    a = _ccl_lib.run_cc(image, algorithm="bke", dtype=dtype)
    b = _ccl_lib.run_cc(image, algorithm="bke_ic", dtype=dtype)
    _ccl_lib.assert_same_partition(a, b, context="bke vs bke_ic")


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_label_range_and_dtype(cuda_device, case, algorithm) -> None:
    """
    All labels lie in ``{0} U [1, H*W]`` and the dtype is int32.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    case : tuple
        ``(shape, density, kind, seed)``.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    shape, density, kind, seed = case
    image = _ccl_lib.random_image(shape, density, kind, seed)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    assert produced.dtype == np.int32
    assert produced.min() >= 0
    assert produced.max() <= shape[0] * shape[1]


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("case", _CASES, ids=_IDS)
def test_connectivity_soundness(cuda_device, case, algorithm) -> None:
    """
    The produced labeling has no under-merge and no over-merge.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    case : tuple
        ``(shape, density, kind, seed)``.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    shape, density, kind, seed = case
    image = _ccl_lib.random_image(shape, density, kind, seed)
    produced = _ccl_lib.run_cc(image, algorithm=algorithm)
    _assert_connectivity_sound(produced)


@pytest.mark.parametrize("algorithm", ["bke_ic", "bke"])
@pytest.mark.parametrize("case", _CASES[::5], ids=_IDS[::5])
def test_determinism_bitwise(cuda_device, case, algorithm) -> None:
    """
    Repeated calls (fresh and reused buffer) yield bitwise-identical
    output arrays, not merely equivalent partitions.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    case : tuple
        ``(shape, density, kind, seed)``.
    algorithm : str
        ``"bke_ic"`` or ``"bke"``.
    """
    shape, density, kind, seed = case
    image = _ccl_lib.random_image(shape, density, kind, seed)
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()

    first = concomtorch.connected_components(tensor, algorithm=algorithm).cpu().numpy()
    for _ in range(4):
        again = concomtorch.connected_components(tensor, algorithm=algorithm).cpu().numpy()
        np.testing.assert_array_equal(again, first)

    buffer = concomtorch.create_labels_buffer(shape)
    for _ in range(4):
        reused = concomtorch.connected_components(tensor, labels=buffer, algorithm=algorithm)
        np.testing.assert_array_equal(reused.cpu().numpy(), first)
