"""
Shared, non-test helpers for the BKE correctness suite.

The leading underscore keeps pytest from collecting this module. It holds
three things every correctness module reuses:

- A dependency-free, trivially-correct 8-connectivity reference labeler
  (``reference_label``) used as ground truth for the fixed-fixture
  module so those tests do not depend on scipy/cc3d.
- ``scipy.ndimage.label`` / ``cc3d`` oracle wrappers, importorskip-guarded
  so the suite degrades gracefully when the oracle is absent.
- A label-permutation-invariant partition comparator and image
  generators (deterministic catalogue + seeded random factory).

All correctness assertions route through ``assert_same_partition`` so
failure output is uniform: BKE final labels are raster-root indices + 1,
never a dense ``1..N`` sequence, so raw label values must never be
compared directly.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import numpy as np
import pytest

# 8-connectivity neighbor offsets (the full 3x3 minus the center).
_NEIGHBORS_8 = tuple(
    (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dy == 0 and dx == 0)
)


def reference_label(image: np.ndarray) -> np.ndarray:
    """
    Label 8-connected components with a trivially-correct BFS.

    This is an independent ground truth (no scipy/cc3d) for the
    fixed-fixture tests. Any nonzero pixel is foreground.

    Parameters
    ----------
    image : np.ndarray
        2D array; nonzero entries are foreground.

    Returns
    -------
    np.ndarray
        int32 array of the same shape; background is 0, components are
        numbered 1, 2, ... in row-major discovery order.
    """
    fg = image != 0
    height, width = fg.shape
    out = np.zeros((height, width), dtype=np.int32)
    next_label = 0
    for start_y in range(height):
        for start_x in range(width):
            if not fg[start_y, start_x] or out[start_y, start_x] != 0:
                continue
            next_label += 1
            out[start_y, start_x] = next_label
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            while len(queue) > 0:
                y, x = queue.popleft()
                for dy, dx in _NEIGHBORS_8:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and fg[ny, nx] and out[ny, nx] == 0:
                        out[ny, nx] = next_label
                        queue.append((ny, nx))
    return out


def scipy_label(image: np.ndarray) -> np.ndarray:
    """
    Label with ``scipy.ndimage.label`` using the full 8-connectivity
    structuring element. Skips the test if scipy is unavailable.

    Parameters
    ----------
    image : np.ndarray
        2D array; nonzero entries are foreground.

    Returns
    -------
    np.ndarray
        int32 label array; background 0.
    """
    ndimage = pytest.importorskip("scipy.ndimage")
    structure = np.ones((3, 3), dtype=np.int32)
    labeled, _ = ndimage.label(image != 0, structure=structure)
    return labeled.astype(np.int32)


def cc3d_label(image: np.ndarray) -> np.ndarray:
    """
    Label with ``cc3d`` at 8-connectivity (2D). Skips if cc3d is absent.

    Parameters
    ----------
    image : np.ndarray
        2D array; nonzero entries are foreground.

    Returns
    -------
    np.ndarray
        int32 label array; background 0.
    """
    cc3d = pytest.importorskip("cc3d")
    labeled = cc3d.connected_components((image != 0).astype(np.uint8), connectivity=8)
    return np.asarray(labeled, dtype=np.int32)


def partition_diff(produced: np.ndarray, expected: np.ndarray) -> str | None:
    """
    Compare two label arrays as partitions, ignoring label identity.

    Two labelings are equivalent iff (a) their background pixel sets
    (label == 0) are identical and (b) the "same nonzero label"
    equivalence relation on foreground pixels is identical, i.e. the
    contingency table of produced-vs-expected labels restricted to
    foreground has exactly one nonzero entry per row and per column.

    Parameters
    ----------
    produced : np.ndarray
        Label array under test (background 0).
    expected : np.ndarray
        Reference label array (background 0).

    Returns
    -------
    str or None
        None when the partitions are equivalent; otherwise a
        human-readable description of the first divergence.
    """
    if produced.shape != expected.shape:
        return f"shape mismatch: produced {produced.shape} vs expected {expected.shape}"

    p_bg = produced == 0
    e_bg = expected == 0
    if not np.array_equal(p_bg, e_bg):
        ys, xs = np.where(p_bg != e_bg)
        coords = list(zip(ys[:10].tolist(), xs[:10].tolist()))
        return (
            f"background set differs at {len(ys)} pixel(s); "
            f"first: {coords} (produced bg={p_bg[ys[0], xs[0]]}, "
            f"expected bg={e_bg[ys[0], xs[0]]})"
        )

    fg = ~e_bg
    if not np.any(fg):
        return None

    p_flat = produced[fg].astype(np.int64)
    e_flat = expected[fg].astype(np.int64)

    # produced -> expected must be a function (one expected label per
    # produced label) and vice versa: that is exactly a bijection of the
    # induced partitions.
    for src, dst, name in ((p_flat, e_flat, "produced->expected"), (e_flat, p_flat, "expected->produced")):
        order = np.argsort(src, kind="stable")
        src_sorted = src[order]
        dst_sorted = dst[order]
        boundaries = np.flatnonzero(np.diff(src_sorted)) + 1
        for group in np.split(dst_sorted, boundaries):
            distinct = np.unique(group)
            if distinct.size > 1:
                bad_src = src_sorted[np.searchsorted(src_sorted, src_sorted[0])]
                return (
                    f"{name} not a bijection: one component split across "
                    f"labels {distinct[:8].tolist()} (a {name.split('->')[0]} "
                    f"label maps to multiple {name.split('->')[1]} labels; "
                    f"example source label {bad_src})"
                )
    return None


def assert_same_partition(produced: np.ndarray, expected: np.ndarray, *, context: str = "") -> None:
    """
    Assert two label arrays induce the same partition.

    Parameters
    ----------
    produced : np.ndarray
        Label array under test.
    expected : np.ndarray
        Reference label array.
    context : str, optional
        Extra text prepended to the failure message.

    Raises
    ------
    AssertionError
        When the partitions differ; the message names the first
        divergence.
    """
    diff = partition_diff(produced, expected)
    if diff is not None:
        prefix = f"{context}: " if context != "" else ""
        raise AssertionError(f"{prefix}{diff}")


def run_cc(
    image: np.ndarray,
    *,
    algorithm: str = "bke_ic",
    dtype: str = "uint8",
    labels=None,
):
    """
    Run the extension on a CPU image and return the labels as numpy.

    Parameters
    ----------
    image : np.ndarray
        2D array; nonzero entries are foreground.
    algorithm : str, optional
        ``"bke_ic"`` (default) or ``"bke"``.
    dtype : str, optional
        ``"uint8"`` or ``"bool"`` input tensor dtype.
    labels : torch.Tensor, optional
        Optional preallocated in-place buffer passed through.

    Returns
    -------
    np.ndarray
        int32 label array on CPU.
    """
    import torch

    import concomtorch

    if dtype == "uint8":
        tensor = torch.from_numpy((image != 0).astype(np.uint8)).cuda()
    elif dtype == "bool":
        tensor = torch.from_numpy((image != 0)).cuda()
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")

    out = concomtorch.connected_components(tensor, labels=labels, algorithm=algorithm)
    return out.cpu().numpy()


# --- deterministic fixture catalogue (section 2 of the design) ----------


def _ring(size: int, thickness: int = 1) -> np.ndarray:
    """
    Build a single hollow square ring of given outer size and thickness.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    img[:thickness, :] = 1
    img[-thickness:, :] = 1
    img[:, :thickness] = 1
    img[:, -thickness:] = 1
    return img


def _concentric_rings(size: int) -> np.ndarray:
    """
    Nested 1px rings separated by 1px gaps; each ring is its own component.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    layer = 0
    while layer * 2 + 1 < size:
        img[layer, layer:size - layer] = 1
        img[size - 1 - layer, layer:size - layer] = 1
        img[layer:size - layer, layer] = 1
        img[layer:size - layer, size - 1 - layer] = 1
        layer += 2
    return img


def _spiral(size: int) -> np.ndarray:
    """
    Single 1px-wide inward spiral path (one component); adjacent arms are
    separated by a 1px gap so a correct labeler keeps it as one piece
    without merging across the gap.
    """
    img = np.zeros((size, size), dtype=np.uint8)
    top, bottom, left, right = 0, size - 1, 0, size - 1
    while top <= bottom and left <= right:
        for x in range(left, right + 1):
            img[top, x] = 1
        for y in range(top, bottom + 1):
            img[y, right] = 1
        for x in range(right, left - 1, -1):
            img[bottom, x] = 1
        for y in range(bottom, top - 1, -1):
            img[y, left] = 1
        top += 2
        bottom -= 2
        left += 2
        right -= 2
    return img


def _comb(rows: int, cols: int) -> np.ndarray:
    """
    Vertical teeth on every other column, joined by no spine: each tooth
    is its own component, separated by 1px gaps.
    """
    img = np.zeros((rows, cols), dtype=np.uint8)
    img[:, ::2] = 1
    return img


def iter_fixed_images() -> Iterator[tuple[str, np.ndarray]]:
    """
    Yield ``(id, image)`` for the deterministic fixture catalogue.

    Covers degenerate shapes, all four H/W parity classes, single-pixel
    placement (including the bottom-right odd/odd corner), the (0,0)
    union-find-root case, diagonal/checkerboard/ring/spiral/comb
    connectivity patterns, and block-boundary adjacencies.

    Yields
    ------
    tuple[str, np.ndarray]
        A short id and a 2D uint8 image.
    """
    # Degenerate / all-background / all-foreground across parity classes.
    for h, w in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (4, 4), (3, 4), (4, 3), (5, 7), (8, 8)):
        yield f"empty_{h}x{w}", np.zeros((h, w), dtype=np.uint8)
        yield f"full_{h}x{w}", np.ones((h, w), dtype=np.uint8)

    # 1x1 foreground.
    yield "single_1x1_fg", np.ones((1, 1), dtype=np.uint8)

    # Single foreground pixel at each corner + interior, on an odd/odd
    # image (bottom-right is the odd/odd corner case).
    base = np.zeros((5, 5), dtype=np.uint8)
    for name, (y, x) in (
        ("tl", (0, 0)),
        ("tr", (0, 4)),
        ("bl", (4, 0)),
        ("br_oddodd", (4, 4)),
        ("interior", (2, 3)),
    ):
        img = base.copy()
        img[y, x] = 1
        yield f"single_{name}_5x5", img

    # Bottom-right corner of several odd/odd sizes.
    for n in (1, 3, 5, 7):
        img = np.zeros((n, n), dtype=np.uint8)
        img[n - 1, n - 1] = 1
        yield f"br_corner_{n}x{n}", img

    # 1xN / Nx1 rows and columns, even and odd, alternating + full.
    for n in (4, 5, 8, 9):
        row = np.zeros((1, n), dtype=np.uint8)
        row[0, ::2] = 1
        yield f"row_alt_1x{n}", row
        col = np.zeros((n, 1), dtype=np.uint8)
        col[::2, 0] = 1
        yield f"col_alt_{n}x1", col

    # (0,0) is the union-find root and foreground: must not merge into bg.
    img = np.zeros((6, 6), dtype=np.uint8)
    img[0, 0] = 1
    img[0, 1] = 1
    img[1, 0] = 1
    yield "root_at_origin_6x6", img

    # Diagonal-only pair (8-connected single component) at four placements.
    for name, (y, x) in (
        ("in_block", (1, 1)),
        ("h_boundary", (1, 2)),
        ("v_boundary", (2, 1)),
        ("block_corner", (1, 1)),
    ):
        img = np.zeros((5, 5), dtype=np.uint8)
        img[y, x] = 1
        img[y + 1, x + 1] = 1
        yield f"diag_pair_{name}", img

    # Checkerboards (8-connected -> one component), even and odd.
    for n in (4, 5, 8):
        board = np.indices((n, n)).sum(axis=0) % 2
        yield f"checker_{n}x{n}", board.astype(np.uint8)

    # Concentric rings, ring, spiral, comb.
    yield "rings_9x9", _concentric_rings(9)
    yield "rings_11x11", _concentric_rings(11)
    yield "ring_6x6", _ring(6)
    yield "spiral_11x11", _spiral(11)
    yield "spiral_12x12", _spiral(12)
    yield "comb_7x9", _comb(7, 9)

    # Block-boundary adjacency probes: minimal pairs connecting only
    # through the right (S), top (Q), upper-right (R), upper-left (P)
    # inter-block relations.
    for name, pixels in (
        ("S_right", ((2, 1), (2, 2))),
        ("Q_top", ((1, 2), (2, 2))),
        ("R_upper_right", ((1, 3), (2, 2))),
        ("P_upper_left", ((1, 1), (2, 2))),
    ):
        img = np.zeros((4, 4), dtype=np.uint8)
        for y, x in pixels:
            img[y, x] = 1
        yield f"block_adj_{name}", img


def random_image(shape: tuple[int, int], density: float, kind: str, seed: int) -> np.ndarray:
    """
    Generate a reproducible random binary image.

    Parameters
    ----------
    shape : tuple[int, int]
        ``(H, W)``.
    density : float
        Approximate foreground fraction for the ``bernoulli`` kind.
    kind : str
        ``"bernoulli"``, ``"rectangles"``, ``"walk"``, or
        ``"dilated"``.
    seed : int
        RNG seed; logged into the test id so failures reproduce.

    Returns
    -------
    np.ndarray
        2D uint8 image.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    if kind == "bernoulli":
        return (rng.random((height, width)) < density).astype(np.uint8)
    if kind == "dilated":
        seed_img = rng.random((height, width)) < density / 4
        img = seed_img.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                img |= np.roll(np.roll(seed_img, dy, axis=0), dx, axis=1)
        return img.astype(np.uint8)
    if kind == "rectangles":
        img = np.zeros((height, width), dtype=np.uint8)
        for _ in range(max(1, int(density * 20))):
            y0 = rng.integers(0, height)
            x0 = rng.integers(0, width)
            y1 = min(height, y0 + rng.integers(1, max(2, height // 3)))
            x1 = min(width, x0 + rng.integers(1, max(2, width // 3)))
            img[y0:y1, x0:x1] = 1
        return img
    if kind == "walk":
        img = np.zeros((height, width), dtype=np.uint8)
        y, x = height // 2, width // 2
        for _ in range(int(density * height * width)):
            img[y, x] = 1
            y = int(np.clip(y + rng.integers(-1, 2), 0, height - 1))
            x = int(np.clip(x + rng.integers(-1, 2), 0, width - 1))
        return img
    raise ValueError(f"unknown kind {kind!r}")
