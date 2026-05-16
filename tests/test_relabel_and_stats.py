"""
``relabel_components`` / ``component_stats`` contract and the
``create_labels_buffer`` shape-validation guard.

``relabel_components`` is checked for the dense ``1..N`` invariant,
background preservation, partition equivalence with the sparse input, the
``dense=False`` passthrough, and validation. ``component_stats`` is checked
against an independent NumPy recomputation of area, inclusive bbox, and
centroid derived from the same GPU label map, on real CCL output and on
empty/degenerate inputs.
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


def _label_on_gpu(image: np.ndarray) -> torch.Tensor:
    """
    Label an image and return the int32 CUDA labels tensor.

    Parameters
    ----------
    image : np.ndarray
        2D array; nonzero is foreground.

    Returns
    -------
    torch.Tensor
        int32 CUDA labels of shape ``image.shape``.
    """
    tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()
    return concomtorch.connected_components(tensor)


def _expected_stats(labels_np: np.ndarray) -> dict[int, tuple[int, tuple[int, int, int, int], tuple[float, float]]]:
    """
    Recompute per-label area, inclusive bbox, and centroid from a label map.

    Parameters
    ----------
    labels_np : np.ndarray
        2D int label map; 0 is background.

    Returns
    -------
    dict
        Maps each nonzero label to ``(area, (min_row, min_col, max_row,
        max_col), (centroid_row, centroid_col))``.
    """
    out: dict[int, tuple[int, tuple[int, int, int, int], tuple[float, float]]] = {}
    for lab in np.unique(labels_np):
        if lab == 0:
            continue
        rows, cols = np.nonzero(labels_np == lab)
        out[int(lab)] = (
            int(rows.size),
            (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
            (float(rows.mean()), float(cols.mean())),
        )
    return out


def test_relabel_dense_invariant(cuda_device) -> None:
    """
    ``relabel_components`` yields a contiguous ``0..N`` value set with
    background 0 preserved and the partition unchanged.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((48, 53), 0.25, "dilated", seed=7)
    labels = _label_on_gpu(image)
    dense = concomtorch.relabel_components(labels)

    assert dense.dtype == torch.int32
    assert dense.shape == labels.shape

    dense_np = dense.cpu().numpy()
    labels_np = labels.cpu().numpy()

    uniq = np.unique(dense_np)
    nonzero = labels_np[labels_np != 0]
    n = int(np.unique(nonzero).size) if nonzero.size > 0 else 0
    np.testing.assert_array_equal(uniq, np.arange(n + 1))

    # Background stays background and only background maps to 0.
    np.testing.assert_array_equal((dense_np == 0), (labels_np == 0))

    # Partition is preserved: the relabeling is a bijection on component ids.
    pairs = np.stack([labels_np.ravel(), dense_np.ravel()], axis=1)
    assert len(np.unique(pairs[pairs[:, 0] != 0], axis=0)) == n


def test_relabel_dense_false_is_clone(cuda_device) -> None:
    """
    ``dense=False`` returns an equal but distinct tensor.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((32, 32), 0.2, "bernoulli", seed=3)
    labels = _label_on_gpu(image)
    passthrough = concomtorch.relabel_components(labels, dense=False)
    assert passthrough.data_ptr() != labels.data_ptr()
    torch.testing.assert_close(passthrough, labels)


def test_relabel_empty_and_all_background(cuda_device) -> None:
    """
    All-background and tiny inputs relabel to all zeros.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    labels = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    dense = concomtorch.relabel_components(labels)
    assert int(dense.max().item()) == 0
    assert int(dense.min().item()) == 0


def test_relabel_validation(cuda_device) -> None:
    """
    Non-int32 / non-2D / CPU inputs raise ``ValueError``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    with pytest.raises(ValueError):
        concomtorch.relabel_components(torch.zeros((4, 4), dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError):
        concomtorch.relabel_components(torch.zeros((4,), dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError):
        concomtorch.relabel_components(torch.zeros((4, 4), dtype=torch.int32))


def test_component_stats_matches_numpy(cuda_device) -> None:
    """
    ``component_stats`` area, inclusive bbox, and centroid equal an
    independent NumPy recomputation from the same label map.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    image = _ccl_lib.random_image((57, 49), 0.22, "dilated", seed=19)
    labels = _label_on_gpu(image)
    labels_np = labels.cpu().numpy()
    expected = _expected_stats(labels_np)

    stats = concomtorch.component_stats(labels)
    assert stats.labels.dtype == torch.int32
    assert stats.area.dtype == torch.int64
    assert stats.bbox.dtype == torch.int32
    assert stats.centroid.dtype == torch.float64
    assert stats.labels.numel() == len(expected)

    ids = stats.labels.cpu().numpy()
    area = stats.area.cpu().numpy()
    bbox = stats.bbox.cpu().numpy()
    centroid = stats.centroid.cpu().numpy()

    for i, lab in enumerate(ids):
        exp_area, exp_bbox, exp_centroid = expected[int(lab)]
        assert int(area[i]) == exp_area
        np.testing.assert_array_equal(bbox[i], np.array(exp_bbox, dtype=np.int32))
        np.testing.assert_allclose(centroid[i], np.array(exp_centroid), rtol=0, atol=1e-9)


def test_component_stats_empty(cuda_device) -> None:
    """
    An all-background map yields zero-length, correctly-typed tensors.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    labels = torch.zeros((24, 24), dtype=torch.int32, device="cuda")
    stats = concomtorch.component_stats(labels)
    assert stats.labels.numel() == 0
    assert stats.area.shape == (0,)
    assert stats.bbox.shape == (0, 4)
    assert stats.centroid.shape == (0, 2)
    assert stats.area.dtype == torch.int64
    assert stats.centroid.dtype == torch.float64


def test_component_stats_single_full_component(cuda_device) -> None:
    """
    A single all-foreground component spans the whole image.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    h, w = 8, 12
    labels = _label_on_gpu(np.ones((h, w), dtype=np.uint8))
    stats = concomtorch.component_stats(labels)
    assert stats.labels.numel() == 1
    assert int(stats.area[0]) == h * w
    np.testing.assert_array_equal(stats.bbox[0].cpu().numpy(), np.array([0, 0, h - 1, w - 1], dtype=np.int32))
    np.testing.assert_allclose(stats.centroid[0].cpu().numpy(), np.array([(h - 1) / 2.0, (w - 1) / 2.0]))


def test_create_labels_buffer_rejects_non_sized_shape(cuda_device) -> None:
    """
    A non-sized ``shape`` raises the documented ``ValueError``.

    Parameters
    ----------
    cuda_device : torch.device
        CUDA device fixture.
    """
    with pytest.raises(ValueError):
        concomtorch.create_labels_buffer(512)
    with pytest.raises(ValueError):
        concomtorch.create_labels_buffer((512, 512, 3))
    with pytest.raises(ValueError):
        concomtorch.create_labels_buffer((512, -1))
