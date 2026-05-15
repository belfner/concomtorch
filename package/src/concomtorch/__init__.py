"""
ConComTorch: GPU-accelerated connected component labeling for PyTorch tensors.

Implements the state-of-the-art Block-based Komura Equivalence (BKE) algorithm
from "Optimized Block-Based Algorithms to Label Connected Components on GPUs"
(IEEE TPDS 2019).
"""

import importlib.util

import torch


# Load the custom C++ extension
def _load_extension():
    """Load compiled native extension."""
    spec = importlib.util.find_spec('concomtorch._C')
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:
        raise RuntimeError(
            'Failed to load concomtorch native extension. '
            'Please ensure the package is properly installed with: pip install -e .'
        )


_load_extension()


def connected_components(
    input: torch.Tensor,
    labels: torch.Tensor | None = None,
    algorithm: str = 'bke_ic'
) -> torch.Tensor:
    """
    Label connected components in a 2D binary image (CUDA only).

    Uses Block-based Komura Equivalence (BKE), the state-of-the-art GPU CCL
    algorithm from IEEE TPDS 2019. Supports 8-connectivity and operates on
    2×2 blocks for maximum performance.

    Parameters
    ----------
    input : torch.Tensor
        Binary CUDA tensor of shape (H, W) with dtype uint8 or bool.
        Foreground pixels should have non-zero values. Must be on CUDA device.

    labels : torch.Tensor, optional
        Pre-allocated, contiguous int32 CUDA tensor of shape (H, W).
        If provided, avoids allocation overhead (~4-9% faster for repeated
        calls with same image size). The tensor is modified in-place; a
        non-contiguous buffer is rejected rather than silently replaced.

    algorithm : str, default='bke_ic'
        Which algorithm variant to use:
        - 'bke_ic': BKE with InlineCompression (recommended)
        - 'bke': Standard BKE

    Returns
    -------
    torch.Tensor
        Integer CUDA tensor of shape (H, W) with dtype int32.
        Background pixels are labeled 0, components labeled 1, 2, 3, ...

    Raises
    ------
    RuntimeError
        If CUDA is not available or input tensor is not on CUDA device.

    ValueError
        If algorithm name is invalid or labels tensor shape/dtype mismatches.

    Examples
    --------
    Basic usage with automatic allocation:

    >>> import torch
    >>> from concomtorch import connected_components
    >>> img = torch.tensor([[1, 1, 0, 0],
    ...                     [0, 0, 0, 1],
    ...                     [0, 1, 1, 1]], dtype=torch.uint8, device='cuda')
    >>> labels = connected_components(img)
    >>> print(labels)

    Avoid allocation overhead with pre-allocated buffer (recommended for loops):

    >>> labels_buffer = torch.empty((512, 512), dtype=torch.int32, device='cuda')
    >>> for img in image_batch:
    ...     labels = connected_components(img, labels=labels_buffer)
    ...     # Process labels...

    Get component count:

    >>> from concomtorch import connected_components, get_unique_labels
    >>> labels = connected_components(img)
    >>> num_components = len(get_unique_labels(labels))
    >>> print(f'Found {num_components} objects')

    Notes
    -----
    Performance tips:
    - Use buffer reuse (`labels` parameter) for repeated calls with same size
    - BKE_IC (default) is recommended for best performance
    - Expect 4-9% speedup from removing batch processing overhead vs batched version

    Algorithm details:
    The BKE algorithm operates on 2×2 blocks rather than individual pixels,
    dramatically reducing memory accesses and atomic operations. It uses
    5 CUDA kernels: Init, Compress, Reduce, Compress, FinalLabel.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is not available. concomtorch requires CUDA for connected component labeling.'
        )

    if not input.is_cuda:
        raise RuntimeError(
            f'Input tensor must be on CUDA device, got {input.device}. '
            'Use input.cuda() to move tensor to GPU.'
        )

    # Validate labels buffer if provided
    if labels is not None:
        if not labels.is_cuda:
            raise ValueError(f'Labels tensor must be on CUDA device, got {labels.device}')
        if labels.dtype != torch.int32:
            raise ValueError(f'Labels tensor must be int32, got {labels.dtype}')
        # Shape validation happens in C++ for better error messages

    # Select algorithm
    if algorithm == 'bke_ic':
        return torch.ops.concomtorch.connected_components_bke_ic(input, labels)
    elif algorithm == 'bke':
        return torch.ops.concomtorch.connected_components_bke(input, labels)
    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            "Valid options are: 'bke', 'bke_ic'"
        )


def create_labels_buffer(
    shape: tuple[int, ...],
    device: torch.device | str = 'cuda'
) -> torch.Tensor:
    """
    Create a reusable buffer for connected components labels.

    Useful for avoiding allocation overhead when processing multiple images
    of the same size in a loop. Typical speedup: 4-9% when reusing buffers.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the labels tensor (H, W), matching your input images.

    device : torch.device or str, default='cuda'
        Device to allocate the buffer on.

    Returns
    -------
    torch.Tensor
        Uninitialized int32 tensor that can be passed to connected_components()
        via the `labels` parameter.

    Examples
    --------
    >>> import torch
    >>> from concomtorch import connected_components, create_labels_buffer
    >>>
    >>> # Create buffer once
    >>> labels_buf = create_labels_buffer((512, 512))
    >>>
    >>> # Reuse in loop
    >>> for img in image_batch:
    ...     labels = connected_components(img, labels=labels_buf)
    ...     process(labels)
    """
    return torch.empty(shape, dtype=torch.int32, device=device)


def get_unique_labels(
    labels: torch.Tensor,
    exclude_background: bool = True,
    collapse_consecutive: bool = True
) -> torch.Tensor:
    """
    Get unique component labels efficiently.

    Helper function to extract unique label values from a connected components
    output. Optimized for CCL outputs using consecutive value collapsing.

    Parameters
    ----------
    labels : torch.Tensor
        Labeled image from connected_components(), shape (H, W), dtype int32.
        Must be on CUDA device.

    exclude_background : bool, default=True
        If True, exclude background pixels (label 0) from result.
        If False, include background in the unique labels.

    collapse_consecutive : bool, default=True
        If True, use torch.unique(torch.unique_consecutive(labels.flatten()))
        for faster computation. Recommended for connected component labels
        which have large contiguous regions of identical values.
        If False, use standard torch.unique(labels) directly.

    Returns
    -------
    torch.Tensor
        Unique label values as int32 tensor on CUDA, sorted in ascending order.
        Returns empty tensor if no labels found.

    Raises
    ------
    RuntimeError
        If labels tensor is not on CUDA device.

    ValueError
        If labels tensor has wrong dtype or dimensionality.

    Notes
    -----
    - collapse_consecutive=True is significantly faster for CCL outputs
      (e.g., 512x512 image: ~262K → ~100 elements to unique)
    - Use this to count components: len(get_unique_labels(labels))
    - Returned labels are sorted, but order doesn't reflect spatial position

    Examples
    --------
    Get unique component labels:

    >>> labels = connected_components(img)
    >>> unique = get_unique_labels(labels)
    >>> num_components = len(unique)

    Include background in results:

    >>> unique_with_bg = get_unique_labels(labels, exclude_background=False)

    Use standard unique (slower but works for any tensor):

    >>> unique = get_unique_labels(labels, collapse_consecutive=False)
    """
    if not labels.is_cuda:
        raise RuntimeError(
            f'Labels tensor must be on CUDA device, got {labels.device}'
        )

    if labels.dtype != torch.int32:
        raise ValueError(f'Labels tensor must be int32, got {labels.dtype}')

    if labels.dim() != 2:
        raise ValueError(
            f'Labels tensor must be 2D (H, W), got shape {labels.shape}'
        )

    # Call the CUDA implementation
    return torch.ops.concomtorch.get_unique_labels(labels, exclude_background, collapse_consecutive)


def get_component_masks(
    labels: torch.Tensor,
    unique_labels: torch.Tensor | None = None,
    exclude_background: bool = True,
    collapse_consecutive: bool = True
) -> torch.Tensor:
    """
    Convert labeled component map to binary masks.

    Takes the output of connected_components() and returns individual binary
    masks for each detected component. More memory-efficient than creating
    one-hot encodings and supports non-sequential label values.

    Parameters
    ----------
    labels : torch.Tensor
        Labeled image from connected_components(), shape (H, W), dtype int32.
        Must be on CUDA device.

    unique_labels : torch.Tensor, optional
        Pre-computed unique label values, shape (N,), dtype int32, on CUDA.
        If provided, creates masks only for these specific labels, enabling:
        - Avoiding redundant torch.unique() calls when already computed
        - Filtering to subset of components (e.g., by size threshold)
        - Custom ordering of output masks
        When provided, exclude_background and collapse_consecutive are ignored.
        If None, computes unique labels automatically.

    exclude_background : bool, default=True
        If True, exclude background pixels (label 0) from output masks.
        If False, include background as the first mask.
        Only applies when unique_labels is None (auto-compute mode).

    collapse_consecutive : bool, default=True
        If True, use torch.unique(torch.unique_consecutive(labels)) for faster
        unique label computation. Recommended for connected component labels
        which have large contiguous regions of identical values.
        If False, use standard torch.unique(labels) directly.
        Only applies when unique_labels is None (auto-compute mode).

    Returns
    -------
    torch.Tensor
        Binary masks of shape (N, H, W) with dtype uint8, where N is the
        number of components. Each mask[i] is 1 where that component exists,
        0 elsewhere. Returns empty tensor (0, H, W) if no components found.

    Raises
    ------
    RuntimeError
        If labels tensor is not on CUDA device.

    ValueError
        If labels tensor has wrong dtype or dimensionality.
        If unique_labels provided but has wrong dtype, device, or shape.

    Notes
    -----
    - Output uses uint8 (not bool) for better compatibility with image ops
    - Handles non-sequential labels efficiently (no remapping needed)
    - Components ordered by label value (arbitrary, based on spatial position)
    - Typical performance: ~0.1-0.5ms for images with 5-100 components
    - All data stays on GPU (no CPU copies)
    - collapse_consecutive=True significantly faster for CCL (512x512: ~262K → ~100 elements to sort)

    Examples
    --------
    Basic usage (auto-compute unique labels with fast collapse):

    >>> labels = connected_components(img)
    >>> masks = get_component_masks(labels)  # (N, H, W), uses collapse_consecutive=True

    Reuse unique labels to avoid redundant computation:

    >>> from concomtorch import get_unique_labels
    >>> unique = get_unique_labels(labels)
    >>> num_components = len(unique)
    >>> masks = get_component_masks(labels, unique)  # faster

    Filter to specific components (e.g., by size):

    >>> unique = get_unique_labels(labels, exclude_background=False)
    >>> sizes = torch.bincount(labels.flatten())[unique]
    >>> large_labels = unique[sizes > 100]
    >>> masks = get_component_masks(labels, large_labels)  # only large components
    """
    if not labels.is_cuda:
        raise RuntimeError(
            f'Labels tensor must be on CUDA device, got {labels.device}'
        )

    if labels.dtype != torch.int32:
        raise ValueError(f'Labels tensor must be int32, got {labels.dtype}')

    if labels.dim() != 2:
        raise ValueError(
            f'Labels tensor must be 2D (H, W), got shape {labels.shape}'
        )

    if unique_labels is not None:
        if not unique_labels.is_cuda:
            raise ValueError(
                f'unique_labels must be on CUDA device, got {unique_labels.device}'
            )
        if unique_labels.dtype != torch.int32:
            raise ValueError(
                f'unique_labels must be int32, got {unique_labels.dtype}'
            )
        if unique_labels.dim() != 1:
            raise ValueError(
                f'unique_labels must be 1D, got shape {unique_labels.shape}'
            )

    return torch.ops.concomtorch.get_component_masks(labels, unique_labels, exclude_background, collapse_consecutive)


class ConnectedComponentsLabeler:
    """
    Helper class for efficient repeated connected component labeling.

    Pre-allocates a label buffer for a specific image size and device,
    avoiding allocation overhead on every call. Recommended for processing
    multiple images of the same size.

    Parameters
    ----------
    image_size : tuple[int, int]
        Image dimensions (H, W) to allocate buffer for.
    device : torch.device or str, default='cuda'
        Device to allocate buffer on.
    algorithm : str, default='bke_ic'
        Algorithm variant to use: 'bke_ic' (recommended) or 'bke'.

    Attributes
    ----------
    image_size : tuple[int, int]
        The (H, W) dimensions this labeler is configured for.
    device : torch.device
        The device the label buffer is allocated on.
    labels_buffer : torch.Tensor
        Pre-allocated int32 tensor of shape (H, W).

    Examples
    --------
    >>> from concomtorch import ConnectedComponentsLabeler
    >>> import torch
    >>>
    >>> # Create labeler for 512×512 images
    >>> labeler = ConnectedComponentsLabeler((512, 512))
    >>>
    >>> # Process multiple images efficiently
    >>> for img in image_batch:
    ...     labels = labeler(img)
    ...     # Process labels...
    >>>
    >>> # Get component count
    >>> from concomtorch import get_unique_labels
    >>> labels = labeler(img)
    >>> num_components = len(get_unique_labels(labels))
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        device: torch.device | str = 'cuda',
        algorithm: str = 'bke_ic'
    ):
        self.image_size = image_size
        self.device = torch.device(device) if isinstance(device, str) else device
        self.algorithm = algorithm
        self.labels_buffer = create_labels_buffer(image_size, device)

    def __call__(
        self,
        input: torch.Tensor
    ) -> torch.Tensor:
        """
        Label connected components in input image.

        Parameters
        ----------
        input : torch.Tensor
            Binary tensor of shape (H, W) matching the configured image_size.

        Returns
        -------
        torch.Tensor
            Label tensor of shape (H, W).

        Raises
        ------
        ValueError
            If input shape doesn't match the configured image_size.
        """
        if input.shape != self.image_size:
            raise ValueError(
                f'Input shape {input.shape} does not match configured '
                f'image_size {self.image_size}'
            )

        return connected_components(
            input,
            labels=self.labels_buffer,
            algorithm=self.algorithm
        )

    def __repr__(self) -> str:
        return (
            f'ConnectedComponentsLabeler(image_size={self.image_size}, '
            f'device={self.device}, algorithm={self.algorithm!r})'
        )


__version__ = '0.1.0'
__all__ = [
    'connected_components',
    'create_labels_buffer',
    'get_unique_labels',
    'get_component_masks',
    'ConnectedComponentsLabeler'
]
