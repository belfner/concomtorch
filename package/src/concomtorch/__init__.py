"""
ConComTorch: GPU-accelerated connected component labeling for PyTorch tensors.

Implements the state-of-the-art Block-based Komura Equivalence (BKE) algorithm
from "Optimized Block-Based Algorithms to Label Connected Components on GPUs"
(IEEE TPDS 2019).

All operations are CUDA-only, non-differentiable, and run on the caller's
current CUDA stream and the input tensor's device. See README.md for usage
examples, the buffer-reuse aliasing contract, and limitations.
"""

import importlib.metadata
import importlib.util
from dataclasses import dataclass

import torch

_EXTENSION_LOADED = False


def _ensure_extension() -> None:
    """
    Load the compiled native extension on first use.

    The extension is loaded lazily so importing :mod:`concomtorch` succeeds
    even when the compiled op is unavailable (for example a source build on a
    host without a CUDA toolkit); the precise error is raised only when an
    operation that needs the kernels is actually called.

    Raises
    ------
    RuntimeError
        If the compiled ``concomtorch._C`` extension cannot be located.
    """
    global _EXTENSION_LOADED
    if _EXTENSION_LOADED:
        return
    spec = importlib.util.find_spec("concomtorch._C")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "The compiled concomtorch._C extension is not available. This "
            "happens when the package was built on a host without a CUDA "
            "toolkit (nvcc), which produces a Python-only install. Install a "
            "published wheel matching your CUDA/torch build, or build from "
            "source on a machine with a CUDA toolkit."
        )
    torch.ops.load_library(spec.origin)
    _EXTENSION_LOADED = True


def _require_cuda_available() -> None:
    """
    Ensure a CUDA runtime is present.

    Raises
    ------
    RuntimeError
        If CUDA is not available in this process.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. concomtorch requires CUDA for connected component labeling.")


def _validate_image(input: torch.Tensor) -> None:
    """
    Validate a binary input image.

    Parameters
    ----------
    input : torch.Tensor
        Candidate input tensor.

    Raises
    ------
    ValueError
        If the tensor is not a 2D CUDA uint8/bool tensor.
    """
    if not input.is_cuda:
        raise ValueError(
            f"Input tensor must be on a CUDA device, got {input.device}. Use input.cuda() to move the tensor to GPU."
        )
    if input.dim() != 2:
        raise ValueError(f"Input tensor must be 2D (H, W), got shape {tuple(input.shape)}")
    if input.dtype not in (torch.uint8, torch.bool):
        raise ValueError(f"Input tensor must be uint8 or bool, got {input.dtype}")


def _same_cuda_device(a: torch.device, b: torch.device) -> bool:
    """
    Test whether two CUDA devices refer to the same physical device.

    An index-less ``cuda`` device resolves to the current CUDA device, so
    ``torch.device("cuda")`` and the concrete ``cuda:0`` of a ``.cuda()``
    tensor compare equal.

    Parameters
    ----------
    a : torch.device
        First device.
    b : torch.device
        Second device.

    Returns
    -------
    bool
        True when both are CUDA devices resolving to the same index.
    """
    if a.type != "cuda" or b.type != "cuda":
        return a == b
    current = torch.cuda.current_device()
    a_index = a.index if a.index is not None else current
    b_index = b.index if b.index is not None else current
    return a_index == b_index


def _validate_labels_buffer(labels: torch.Tensor, input: torch.Tensor) -> None:
    """
    Validate a preallocated labels buffer against its input image.

    Parameters
    ----------
    labels : torch.Tensor
        Caller-supplied output buffer.
    input : torch.Tensor
        The validated input image the buffer will be written for.

    Raises
    ------
    ValueError
        If the buffer is not a contiguous int32 CUDA tensor on the same
        device and shape as ``input``.
    """
    if not labels.is_cuda:
        raise ValueError(f"Labels buffer must be on a CUDA device, got {labels.device}")
    if labels.dtype != torch.int32:
        raise ValueError(f"Labels buffer must be int32, got {labels.dtype}")
    if not _same_cuda_device(labels.device, input.device):
        raise ValueError(
            f"Labels buffer device {labels.device} does not match input "
            f"device {input.device}; they must be on the same CUDA device."
        )
    if tuple(labels.shape) != tuple(input.shape):
        raise ValueError(f"Labels buffer shape {tuple(labels.shape)} does not match input shape {tuple(input.shape)}")
    if not labels.is_contiguous():
        raise ValueError(
            "Labels buffer must be contiguous (it is written in place). Allocate it with create_labels_buffer()."
        )


def _validate_label_map(labels: torch.Tensor, name: str = "labels") -> None:
    """
    Validate a labeled component map produced by :func:`connected_components`.

    Parameters
    ----------
    labels : torch.Tensor
        Candidate label map.
    name : str, default='labels'
        Argument name used in error messages.

    Raises
    ------
    ValueError
        If the tensor is not a 2D int32 CUDA tensor.
    """
    if not labels.is_cuda:
        raise ValueError(f"{name} tensor must be on a CUDA device, got {labels.device}")
    if labels.dtype != torch.int32:
        raise ValueError(f"{name} tensor must be int32, got {labels.dtype}")
    if labels.dim() != 2:
        raise ValueError(f"{name} tensor must be 2D (H, W), got shape {tuple(labels.shape)}")


def connected_components(
    input: torch.Tensor, labels: torch.Tensor | None = None, algorithm: str = "bke_ic"
) -> torch.Tensor:
    """
    Label connected components in a 2D binary image (CUDA only).

    Uses Block-based Komura Equivalence (BKE), the state-of-the-art GPU CCL
    algorithm from IEEE TPDS 2019, with 8-connectivity on 2x2 blocks. The
    kernels launch on the caller's current CUDA stream and on the input
    tensor's device. The operation is non-differentiable; call it under
    ``torch.no_grad()`` in training code.

    Parameters
    ----------
    input : torch.Tensor
        Binary CUDA tensor of shape (H, W), dtype uint8 or bool. Any non-zero
        value is foreground; zero is background. A non-contiguous input is
        copied internally (the buffer-reuse fast path then only avoids the
        output allocation, not the input copy).

    labels : torch.Tensor, optional
        Preallocated contiguous int32 CUDA tensor of shape (H, W) on the same
        device as ``input``, used as the output buffer to avoid an allocation.
        The returned tensor *is this buffer* (an alias, not a copy): a
        subsequent call that reuses the buffer overwrites previously returned
        results. To retain a result across reuse, store ``result.clone()``.
        See the README "Efficiency tips" section.

    algorithm : str, default='bke_ic'
        Variant: ``'bke_ic'`` (BKE with InlineCompression, recommended) or
        ``'bke'`` (standard BKE).

    Returns
    -------
    torch.Tensor
        int32 CUDA tensor of shape (H, W). Background is 0; component labels
        are positive int32 values. Labels are **not** guaranteed to be dense
        or sequential (they derive from each component's root raster index).
        Use :func:`relabel_components` for dense 1..N ids.

    Raises
    ------
    ValueError
        If ``algorithm`` is invalid, or ``input`` / ``labels`` is not a valid
        CUDA tensor of the required dtype, rank, shape, or device.
    RuntimeError
        If CUDA is not available, or the compiled extension is missing.

    Notes
    -----
    Degenerate inputs are well-defined: an empty (0-sized), 1x1, or
    all-background image returns an all-zero / correctly shaped int32 result.
    Labels are deterministic in value set for a given input and algorithm.
    """
    if algorithm not in ("bke_ic", "bke"):
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Valid options: 'bke_ic', 'bke'")

    _ensure_extension()
    _require_cuda_available()
    _validate_image(input)
    if labels is not None:
        _validate_labels_buffer(labels, input)

    if algorithm == "bke_ic":
        return torch.ops.concomtorch.connected_components_bke_ic(input, labels)
    return torch.ops.concomtorch.connected_components_bke(input, labels)


def create_labels_buffer(
    shape: tuple[int, int], device: torch.device | str = "cuda", zero_fill: bool = False
) -> torch.Tensor:
    """
    Create a reusable output buffer for :func:`connected_components`.

    The buffer is uninitialized by default (``torch.empty``): the BKE pipeline
    writes every output cell on every call, so initialization is unnecessary
    for correctness and is omitted for speed. ``zero_fill=True`` returns a
    zeroed buffer for debugging (it makes a leaked stale value visible as 0).

    Parameters
    ----------
    shape : tuple[int, int]
        Buffer shape (H, W), matching the images it will be used with.

    device : torch.device or str, default='cuda'
        CUDA device to allocate the buffer on.

    zero_fill : bool, default=False
        If True, zero-initialize the buffer (debugging aid).

    Returns
    -------
    torch.Tensor
        int32 CUDA tensor of shape (H, W) suitable for the ``labels``
        parameter of :func:`connected_components`.

    Raises
    ------
    ValueError
        If ``shape`` is not a 2-tuple of non-negative ints, or ``device`` is
        not a CUDA device.
    """
    if (
        not isinstance(shape, (tuple, list, torch.Size))
        or len(shape) != 2
        or not all(isinstance(s, int) and s >= 0 for s in shape)
    ):
        raise ValueError(f"shape must be a 2-tuple (H, W) of non-negative ints, got {shape!r}")
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"create_labels_buffer requires a CUDA device, got {dev}")
    factory = torch.zeros if zero_fill else torch.empty
    return factory(shape, dtype=torch.int32, device=dev)


def get_unique_labels(
    labels: torch.Tensor, exclude_background: bool = True, collapse_consecutive: bool = True
) -> torch.Tensor:
    """
    Get unique component labels efficiently (all work stays on GPU).

    Parameters
    ----------
    labels : torch.Tensor
        Label map from :func:`connected_components`, shape (H, W), int32, on
        CUDA.

    exclude_background : bool, default=True
        If True, drop the background label 0 from the result.

    collapse_consecutive : bool, default=True
        If True, collapse adjacent equal values before the final unique,
        which is materially faster when labels form long contiguous runs
        (the common CCL case). For scattered/noisy maps it adds a pass with
        little benefit; set False there.

    Returns
    -------
    torch.Tensor
        Sorted unique int32 label values on CUDA. Empty if no labels remain.

    Raises
    ------
    ValueError
        If ``labels`` is not a 2D int32 CUDA tensor.
    RuntimeError
        If the compiled extension is missing.
    """
    _ensure_extension()
    _validate_label_map(labels)
    return torch.ops.concomtorch.get_unique_labels(labels, exclude_background, collapse_consecutive)


def get_component_masks(
    labels: torch.Tensor,
    unique_labels: torch.Tensor | None = None,
    exclude_background: bool = True,
    collapse_consecutive: bool = True,
) -> torch.Tensor:
    """
    Convert a labeled component map to per-component binary masks.

    Parameters
    ----------
    labels : torch.Tensor
        Label map from :func:`connected_components`, shape (H, W), int32, on
        CUDA.

    unique_labels : torch.Tensor, optional
        Precomputed unique int32 label values, shape (N,), on the same CUDA
        device as ``labels``. When given, masks are produced for exactly
        these labels in this order; ``exclude_background`` and
        ``collapse_consecutive`` then do not apply. Passing
        ``unique_labels`` together with a non-default ``exclude_background``
        or ``collapse_consecutive`` is rejected, because the flags would have
        no effect and the combination is ambiguous.

    exclude_background : bool, default=True
        Auto-compute mode only: drop background label 0.

    collapse_consecutive : bool, default=True
        Auto-compute mode only: see :func:`get_unique_labels`.

    Returns
    -------
    torch.Tensor
        uint8 CUDA tensor of shape (N, H, W); ``out[i]`` is 1 where component
        ``i`` is present. Shape (0, H, W) if there are no components.

    Raises
    ------
    ValueError
        If ``labels`` is not a 2D int32 CUDA tensor; if ``unique_labels`` is
        not a 1D int32 CUDA tensor on the same device as ``labels``; or if
        ``unique_labels`` is combined with non-default flags.
    RuntimeError
        If the compiled extension is missing.

    Notes
    -----
    The result is a dense ``(N, H, W)`` uint8 tensor: it is only more compact
    than a ``max_label + 1`` one-hot when labels are sparse, and its size is
    ``N * H * W`` bytes, which can be very large for many components. Filter
    ``unique_labels`` to the components you need before calling.
    """
    _ensure_extension()
    _validate_label_map(labels)

    if unique_labels is not None:
        if exclude_background is not True or collapse_consecutive is not True:
            raise ValueError(
                "unique_labels was provided together with a non-default "
                "exclude_background/collapse_consecutive. Those flags only "
                "apply in auto-compute mode and would be ignored here. Pass "
                "unique_labels alone (it is the sole source of truth), or "
                "omit it to use the flags."
            )
        if not unique_labels.is_cuda:
            raise ValueError(f"unique_labels must be on a CUDA device, got {unique_labels.device}")
        if unique_labels.dtype != torch.int32:
            raise ValueError(f"unique_labels must be int32, got {unique_labels.dtype}")
        if unique_labels.dim() != 1:
            raise ValueError(f"unique_labels must be 1D, got shape {tuple(unique_labels.shape)}")
        if unique_labels.device != labels.device:
            raise ValueError(
                f"unique_labels device {unique_labels.device} does not match labels device {labels.device}"
            )

    return torch.ops.concomtorch.get_component_masks(labels, unique_labels, exclude_background, collapse_consecutive)


def relabel_components(labels: torch.Tensor, dense: bool = True) -> torch.Tensor:
    """
    Relabel a component map so ids are dense and sequential.

    :func:`connected_components` returns positive but sparse labels. This maps
    them to ``1, 2, ..., N`` (background stays 0), entirely on GPU via
    ``torch.unique(..., return_inverse=True)`` (a single sort/dedup/scatter;
    no custom kernel is faster for arbitrary sparse ids).

    Parameters
    ----------
    labels : torch.Tensor
        Label map from :func:`connected_components`, shape (H, W), int32, on
        CUDA.

    dense : bool, default=True
        If True, return densely renumbered labels. If False, return a copy of
        the input unchanged (provided for API symmetry).

    Returns
    -------
    torch.Tensor
        int32 CUDA tensor of shape (H, W). When ``dense=True``, background is
        0 and components are ``1..N``. When ``dense=False``, an unchanged copy
        of ``labels`` (still sparse) is returned.

    Raises
    ------
    ValueError
        If ``labels`` is not a 2D int32 CUDA tensor.
    """
    _validate_label_map(labels)
    if not dense:
        return labels.clone()

    unique, inverse = torch.unique(labels, return_inverse=True)
    relabeled = inverse.to(torch.int32)
    if unique.numel() > 0:
        # If background (0) is present it is unique[0] -> inverse 0, already
        # correct. If absent, shift by 1 so 0 stays reserved for background.
        bg_present = unique[0] == 0
        relabeled = relabeled + (~bg_present).to(torch.int32)
    return relabeled.view_as(labels)


@dataclass
class ComponentStats:
    """
    Per-component statistics returned by :func:`component_stats`.

    Attributes
    ----------
    labels : torch.Tensor
        The unique original (sparse) component label values, shape (N,),
        int32, on CUDA. ``area[i]``/``bbox[i]``/``centroid[i]`` describe
        ``labels[i]``.
    area : torch.Tensor
        Pixel count per component, shape (N,), int64, on CUDA.
    bbox : torch.Tensor
        Inclusive bounding box per component, shape (N, 4), int32, on CUDA,
        ordered ``[min_row, min_col, max_row, max_col]``.
    centroid : torch.Tensor
        Coordinate mean per component, shape (N, 2), float64, on CUDA,
        ordered ``[row, col]``.
    """

    labels: torch.Tensor
    area: torch.Tensor
    bbox: torch.Tensor
    centroid: torch.Tensor


def component_stats(labels: torch.Tensor) -> ComponentStats:
    """
    Compute per-component area, bounding box, and centroid on the GPU.

    Background (label 0) is excluded. The label map is first densified on GPU
    (a ``get_unique_labels`` reduction plus a ``searchsorted`` gather into a
    temporary ``(H, W)`` int32 id map) so that a single fused CUDA kernel can
    then accumulate area, bbox, and centroid sums in one DRAM pass over that
    id map using per-component atomics. Results are mapped back to the
    original sparse label values. Atomic accumulation serializes when a few
    components dominate the image; it is efficient for the typical
    many-component CCL case.

    Parameters
    ----------
    labels : torch.Tensor
        Label map from :func:`connected_components`, shape (H, W), int32, on
        CUDA.

    Returns
    -------
    ComponentStats
        Per-component statistics aligned with ``ComponentStats.labels``.

    Raises
    ------
    ValueError
        If ``labels`` is not a 2D int32 CUDA tensor.
    RuntimeError
        If the compiled extension is missing.
    """
    _ensure_extension()
    _validate_label_map(labels)

    unique = get_unique_labels(labels, exclude_background=True)
    num_components = unique.numel()
    height, width = labels.shape

    if num_components == 0:
        return ComponentStats(
            labels=unique,
            area=torch.empty(0, dtype=torch.int64, device=labels.device),
            bbox=torch.empty((0, 4), dtype=torch.int32, device=labels.device),
            centroid=torch.empty((0, 2), dtype=torch.float64, device=labels.device),
        )

    # Dense ids in [0, N) for the foreground; background and any value not in
    # `unique` map to -1, which the kernel skips. searchsorted + an exact
    # match check keeps everything on GPU.
    idx = torch.searchsorted(unique, labels)
    idx_clamped = idx.clamp(max=num_components - 1)
    is_component = idx < num_components
    is_component = is_component & (unique[idx_clamped] == labels)
    dense = torch.where(
        is_component,
        idx.to(torch.int32),
        torch.full_like(labels, -1, dtype=torch.int32),
    ).contiguous()

    area, bbox, centroid = torch.ops.concomtorch.component_stats(dense, int(num_components))
    return ComponentStats(labels=unique, area=area, bbox=bbox, centroid=centroid)


def _register_fake_kernels() -> None:
    """
    Register Meta/FakeTensor implementations for shape propagation.

    Allows ``torch.compile`` and meta/fake-tensor tracing to reason about
    output shapes/dtypes without launching CUDA kernels. The connected
    components ops cannot infer the exact label values on meta tensors;
    they return a correctly shaped/typed empty meta tensor. Ops whose output
    extent is data-dependent (``get_unique_labels``, ``get_component_masks``,
    ``component_stats``) are left unregistered so a graph break occurs rather
    than a wrong static shape.
    """
    try:
        lib_register_fake = torch.library.register_fake
    except AttributeError:
        return

    def _cc_fake(input, labels=None):
        return input.new_empty(input.shape, dtype=torch.int32)

    for name in (
        "connected_components",
        "connected_components_bke",
        "connected_components_bke_ic",
    ):
        try:
            lib_register_fake(f"concomtorch::{name}", _cc_fake)
        except (RuntimeError, AttributeError):
            # Op not registered (extension missing) or already has a fake
            # impl; shape-prop simply graph-breaks in that case.
            pass


try:
    _ensure_extension()
    _register_fake_kernels()
except RuntimeError:
    # Extension unavailable (e.g. source build without a CUDA toolkit).
    # Import still succeeds; the precise error is raised on first op use.
    pass


def _resolve_version() -> str:
    """
    Resolve the installed package version, including any local build tag.

    Returns
    -------
    str
        The distribution version (e.g. ``0.1.0+cu124torch2.6``), or
        ``'0.0.0+unknown'`` when package metadata is unavailable.
    """
    try:
        return importlib.metadata.version("concomtorch")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()


class ConnectedComponentsLabeler:
    """
    Reusable fixed-size connected-component labeler.

    Preallocates one int32 output buffer for a fixed (H, W) and device. This
    instance is **not** thread- or stream-safe: the single shared buffer is
    overwritten on every call, so concurrent calls from multiple threads,
    CUDA streams, or overlapping async work race. Use one instance per
    thread/stream/in-flight operation. The returned tensor aliases the shared
    buffer; clone a result you need to retain across calls.

    Parameters
    ----------
    image_size : tuple[int, int]
        Fixed image dimensions (H, W).
    device : torch.device or str, default='cuda'
        CUDA device for the buffer.
    algorithm : str, default='bke_ic'
        Variant: ``'bke_ic'`` (recommended) or ``'bke'``.

    Attributes
    ----------
    image_size : tuple[int, int]
        Configured (H, W).
    device : torch.device
        Device the buffer is on.
    labels_buffer : torch.Tensor
        The shared preallocated int32 buffer.
    """

    def __init__(self, image_size: tuple[int, int], device: torch.device | str = "cuda", algorithm: str = "bke_ic"):
        if algorithm not in ("bke_ic", "bke"):
            raise ValueError(f"Unknown algorithm: {algorithm!r}. Valid options: 'bke_ic', 'bke'")
        self.image_size = image_size
        self.device = torch.device(device)
        self.algorithm = algorithm
        self.labels_buffer = create_labels_buffer(image_size, self.device)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """
        Label connected components in ``input`` using the shared buffer.

        Parameters
        ----------
        input : torch.Tensor
            Binary CUDA tensor whose shape and device match the configured
            ``image_size`` and ``device``.

        Returns
        -------
        torch.Tensor
            The shared int32 buffer (an alias; clone to retain across calls).

        Raises
        ------
        ValueError
            If ``input`` shape or device does not match the configuration.
        """
        if tuple(input.shape) != tuple(self.image_size):
            raise ValueError(
                f"Input shape {tuple(input.shape)} does not match configured image_size {tuple(self.image_size)}"
            )
        if not _same_cuda_device(input.device, self.device):
            raise ValueError(f"Input device {input.device} does not match the labeler device {self.device}")
        return connected_components(input, labels=self.labels_buffer, algorithm=self.algorithm)

    def __repr__(self) -> str:
        return (
            f"ConnectedComponentsLabeler(image_size={self.image_size}, "
            f"device={self.device}, algorithm={self.algorithm!r})"
        )


__all__ = [
    "connected_components",
    "create_labels_buffer",
    "get_unique_labels",
    "get_component_masks",
    "relabel_components",
    "component_stats",
    "ComponentStats",
    "ConnectedComponentsLabeler",
]
