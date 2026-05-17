# API Reference

For task-oriented examples see the [Usage section of the
README](README.md#usage).

## `connected_components(input, labels=None, algorithm='bke_ic')`

Label connected components in a binary image using 8-connectivity (CUDA only).

- `input` (`torch.Tensor`): CUDA tensor, shape `(H, W)`, dtype `uint8` or
  `bool`. Any non-zero value is foreground; `0` is background. Non-contiguous
  input is copied to a contiguous temporary.
- `labels` (`torch.Tensor`, optional): pre-allocated `int32` CUDA tensor of
  shape `(H, W)` on the same device as `input`. Modified in-place and returned
  (aliased). Size/dtype/device validated.
- `algorithm` (`str`): `'bke_ic'` (default) or `'bke'`. Other values raise
  `ValueError`.

Returns an `int32` CUDA tensor `(H, W)`; background `0`, component IDs positive
and **sparse** (not dense `1..N`). Runs under `no_grad` semantics (integer
output, no autograd).

Raises `RuntimeError` if CUDA is unavailable or the compiled extension is
missing; `ValueError` for an invalid algorithm or a mismatched `input`/`labels`
tensor.

## `create_labels_buffer(shape, device='cuda', zero_fill=False)`

Allocate a reusable `int32` output buffer.

- `shape` (`tuple[int, int]`): `(H, W)`, non-negative ints, matching your
  inputs.
- `device`: CUDA device for the buffer.
- `zero_fill` (`bool`): `False` (default) returns `torch.empty` (fast);
  `True` returns `torch.zeros` for debugging buffer-overwrite assumptions.

The fast path relies on every output cell being written each call; that
contract is enforced by tests across odd / empty / all-background sizes. Use
`zero_fill=True` if you suspect a leak from a prior image.

## `get_unique_labels(labels, exclude_background=True, collapse_consecutive=True)`

Unique component IDs as a sorted `int32` CUDA tensor. Fully GPU-side (no host
sync). `collapse_consecutive=True` first applies
`torch.unique_consecutive(labels.flatten())`; this is much faster when equal
labels form long contiguous runs (typical CCL output) and adds a redundant pass
for scattered/noisy label fields.

## `get_component_masks(labels, unique_labels=None, exclude_background=True, collapse_consecutive=True)`

`uint8` masks (values `0`/`1`), shape `(N, H, W)`, one plane per component. The
mask stack is filled by a single fused grid-stride kernel; when `unique_labels`
is not supplied a unique-label reduction runs first to determine `N` and the
label set. Memory is dense: `N * H * W` bytes regardless of
component size. This beats a `(max_label + 1, H, W)` one-hot only when labels
are sparse; for many components it can be large. Passing `unique_labels`
explicitly makes it the sole source of truth; combining it with non-default
`exclude_background` / `collapse_consecutive` raises `ValueError`.

## `relabel_components(labels, dense=True)`

Pure-torch remap of sparse IDs to a contiguous range, fully on-device. With
`dense=True`, background `0` (if present) stays `0` and components become
`1..N`. Returns `int32`, same shape as `labels`.

## `component_stats(labels) -> ComponentStats`

Per-component area, bounding box, and centroid. The label map is densified
on GPU (a `get_unique_labels` reduction plus a `searchsorted` gather into a
temporary `(H, W)` id map), then a single fused CUDA kernel accumulates all
three statistics in one DRAM pass over that id map via per-component atomics.
Returns a `ComponentStats` dataclass:

- `labels` `int32 (N,)` original IDs ascending
- `area` `int64 (N,)` pixel counts
- `bbox` `int32 (N, 4)` `[min_row, min_col, max_row, max_col]`, inclusive
- `centroid` `float64 (N, 2)` `[row, col]`

Empty input yields zero-length tensors with these dtypes.

## `class ConnectedComponentsLabeler(image_size, device='cuda', algorithm='bke_ic')`

Stateful helper holding one fixed-size internal buffer. `__call__(input)`
validates that `input.shape` and `input.device` match the configured values.
Strictly fixed-size by design. Not thread-/stream-safe (see the [Reusable
labeler](README.md#performance-and-reuse) note in the README).
