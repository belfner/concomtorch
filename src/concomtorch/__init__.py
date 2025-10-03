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
    algorithm: str = 'bke_ic',
    return_num_components: bool = False
) -> torch.Tensor | tuple[torch.Tensor, int]:
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
        Pre-allocated int32 CUDA tensor of shape (H, W).
        If provided, avoids allocation overhead (~4-9% faster for repeated
        calls with same image size). The tensor will be modified in-place.

    algorithm : str, default='bke_ic'
        Which algorithm variant to use:
        - 'bke_ic': BKE with InlineCompression (recommended)
        - 'bke': Standard BKE

    return_num_components : bool, default=False
        If True, also return the number of connected components found.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, int]
        Integer CUDA tensor of shape (H, W) with dtype int32.
        Background pixels are labeled 0, components labeled 1, 2, 3, ...
        If return_num_components=True, returns (labels, num_components).

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

    >>> labels, num_components = connected_components(img, return_num_components=True)
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
        result = torch.ops.concomtorch.connected_components_bke_ic(input, labels)
    elif algorithm == 'bke':
        result = torch.ops.concomtorch.connected_components_bke(input, labels)
    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            "Valid options are: 'bke', 'bke_ic'"
        )

    if return_num_components:
        num_components = result.max().item()
        return result, num_components

    return result


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
    >>> labels, num_components = labeler(img, return_num_components=True)
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
        input: torch.Tensor,
        return_num_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, int]:
        """
        Label connected components in input image.

        Parameters
        ----------
        input : torch.Tensor
            Binary tensor of shape (H, W) matching the configured image_size.
        return_num_components : bool, default=False
            If True, also return the number of connected components.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, int]
            Label tensor of shape (H, W).
            If return_num_components=True, returns (labels, num_components).

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
            algorithm=self.algorithm,
            return_num_components=return_num_components
        )

    def __repr__(self) -> str:
        return (
            f'ConnectedComponentsLabeler(image_size={self.image_size}, '
            f'device={self.device}, algorithm={self.algorithm!r})'
        )


__version__ = '0.1.0'
__all__ = ['connected_components', 'create_labels_buffer', 'ConnectedComponentsLabeler']
