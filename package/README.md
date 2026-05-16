# ConComTorch

GPU-accelerated connected component labeling for 2D PyTorch tensors using the
**Block-based Komura Equivalence (BKE)** algorithm (IEEE TPDS 2019). The package
is a thin Python wrapper over a compiled CUDA extension registered through
PyTorch's operator dispatcher.

## Features

- **BKE algorithm** with two variants: `bke_ic` (InlineCompression, default) and `bke` (standard)
- **8-connectivity** on 2x2 blocks (diagonal neighbors are connected)
- **Buffer reuse** via a caller-supplied output tensor to skip the output allocation
- **uint8 and bool** input tensors
- **Stream- and device-correct** launches (`CUDAGuard` + current CUDA stream)
- **Typed**: ships `py.typed`; meta/fake kernels registered for the shape-stable ops

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.4.0
- CUDA >= 11.8 runtime, and a CUDA toolkit (`nvcc`) for source builds
- C++ compiler with C++17 support
- NVIDIA GPU with compute capability >= 7.5 (Turing or newer)

The published wheels are built for SASS on the architectures listed per channel
plus a PTX fallback for the highest target, so newer GPUs run via JIT PTX
compilation. Source builds default to compute capabilities `>= 7.5` available
from the local toolkit (override with `TORCH_CUDA_ARCH_LIST` or
`CONCOMTORCH_COMPUTE_MIN`).

### Install the published wheel

Wheels are published behind a two-layer simple index keyed by CUDA variant and
torch minor (`<cuda>/<torch_tag>/`). The wheel's compiled extension is built
against a specific PyTorch CUDA build; **the channel you install from must match
the CUDA build of the PyTorch already in your environment** (check
`torch.version.cuda` and `torch.__version__`). A mismatch fails at
`import concomtorch` with an opaque symbol/ABI error rather than at install
time.

Pick the directory matching your installed PyTorch. For CUDA 12.6 with torch 2.6:

```bash
pip install concomtorch --index-url https://belfner.github.io/concomtorch/cu126/torch2_6/
```

`--index-url` restricts resolution to the project index. If you need PyPI for
other dependencies in the same command, use `--extra-index-url` instead and be
aware pip will also consider PyPI for a package named `concomtorch`; prefer a
separate, isolated install step for the supply-chain-sensitive package.

Browse `https://belfner.github.io/concomtorch/` for the available CUDA variants
and, under each, the torch channels.

### Install from source

The buildable package lives in `package/`. The repository root is the CI
orchestration environment, not the package, so install the subdirectory:

```bash
git clone <repository-url>
cd concomtorch
pip install -e ./package
```

A working CUDA toolkit (`nvcc`) must be on `PATH` (or discoverable via
`CUDA_HOME`). Without it the build installs a Python-only package and
`import concomtorch` raises a `RuntimeError` explaining the extension is
missing.

### Build with specific CUDA architectures

`TORCH_CUDA_ARCH_LIST` is a semicolon-separated list of dotted compute
capabilities:

```bash
TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' pip install -e ./package
```

## Usage

### Basic usage

```python
import torch
from concomtorch import connected_components

# Binary image on CUDA (required). Any non-zero value is foreground; 0 is background.
img = torch.tensor(
    [[1, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 1, 1, 1]],
    dtype=torch.uint8,
    device='cuda',
)

labels = connected_components(img)
# int32 (H, W). Background pixels are 0. Component IDs are positive but
# sparse (root-derived), not a dense 1..N range. See relabel_components.
```

### Buffer reuse

Supply a pre-allocated output tensor to skip the per-call output allocation:

```python
import torch
from concomtorch import connected_components, create_labels_buffer

labels_buffer = create_labels_buffer((512, 512))

results = []
for img in image_batch:
    labels = connected_components(img, labels=labels_buffer)
    # `labels` IS `labels_buffer` (aliased). Retaining it across iterations
    # requires a copy, or every stored result points at the last image.
    results.append(labels.clone())
```

The return value aliases the supplied buffer. The buffer only removes the
output allocation; a non-contiguous input is still copied to a contiguous
temporary internally.

### Counting and enumerating components

```python
from concomtorch import connected_components, get_unique_labels

labels = connected_components(img)
ids = get_unique_labels(labels)          # sorted int32 IDs, background excluded
num_components = ids.numel()
```

### Densifying labels

`connected_components` produces sparse positive IDs. `relabel_components`
remaps them to a contiguous `0..N` range (0 stays background if present):

```python
from concomtorch import relabel_components

dense = relabel_components(labels)       # pure-torch, stays on GPU
```

### Per-component statistics

```python
from concomtorch import component_stats

stats = component_stats(labels)
stats.labels    # int32 (N,)  original sparse IDs, ascending
stats.area      # int64 (N,)  pixel count per component
stats.bbox      # int32 (N,4) [min_row, min_col, max_row, max_col] inclusive
stats.centroid  # float64 (N,2) [row, col]
```

### Component masks

```python
from concomtorch import get_component_masks

masks = get_component_masks(labels)      # bool (N, H, W), one plane per component
```

### Reusable labeler helper

```python
from concomtorch import ConnectedComponentsLabeler

labeler = ConnectedComponentsLabeler((512, 512), algorithm='bke_ic')
labels = labeler(img)                    # input shape and device must match config
```

`ConnectedComponentsLabeler` owns one internal buffer and is **not**
thread-safe or stream-safe. Use one instance per device, per CUDA stream, and
per in-flight operation; serialize concurrent use or give each worker its own
labeler. Output aliases the internal buffer (same `.clone()` rule as buffer
reuse).

### Algorithm variants

```python
labels = connected_components(img, algorithm='bke_ic')  # default, fastest for 2D
labels = connected_components(img, algorithm='bke')      # standard BKE
```

## API Reference

### `connected_components(input, labels=None, algorithm='bke_ic')`

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

### `create_labels_buffer(shape, device='cuda', zero_fill=False)`

Allocate a reusable `int32` output buffer.

- `shape` (`tuple[int, int]`): `(H, W)`, non-negative ints, matching your
  inputs.
- `device`: CUDA device for the buffer.
- `zero_fill` (`bool`): `False` (default) returns `torch.empty` (fast);
  `True` returns `torch.zeros` for debugging buffer-overwrite assumptions.

The fast path relies on every output cell being written each call; that
contract is enforced by tests across odd / empty / all-background sizes. Use
`zero_fill=True` if you suspect a leak from a prior image.

### `get_unique_labels(labels, exclude_background=True, collapse_consecutive=True)`

Unique component IDs as a sorted `int32` CUDA tensor. Fully GPU-side (no host
sync). `collapse_consecutive=True` first applies
`torch.unique_consecutive(labels.flatten())`; this is much faster when equal
labels form long contiguous runs (typical CCL output) and adds a redundant pass
for scattered/noisy label fields.

### `get_component_masks(labels, exclude_background=True, collapse_consecutive=True, unique_labels=None)`

Boolean masks, shape `(N, H, W)`, one plane per component, produced by a single
fused grid-stride kernel. Memory is dense: `N * H * W` bytes regardless of
component size. This beats a `(max_label + 1, H, W)` one-hot only when labels
are sparse; for many components it can be large. Passing `unique_labels`
explicitly makes it the sole source of truth; combining it with non-default
`exclude_background` / `collapse_consecutive` raises `ValueError`.

### `relabel_components(labels, dense=True)`

Pure-torch remap of sparse IDs to a contiguous range, fully on-device. With
`dense=True`, background `0` (if present) stays `0` and components become
`1..N`. Returns `int32`, same shape as `labels`.

### `component_stats(labels) -> ComponentStats`

Per-component area, bounding box, and centroid via a single fused CUDA kernel
(one DRAM pass). Returns a `ComponentStats` dataclass:

- `labels` `int32 (N,)` original IDs ascending
- `area` `int64 (N,)` pixel counts
- `bbox` `int32 (N, 4)` `[min_row, min_col, max_row, max_col]`, inclusive
- `centroid` `float64 (N, 2)` `[row, col]`

Empty input yields zero-length tensors with these dtypes.

### `class ConnectedComponentsLabeler(shape, device='cuda', algorithm='bke_ic')`

Stateful helper holding one fixed-size internal buffer. `__call__(input)`
validates that `input.shape` and `input.device` match the configured values.
Strictly fixed-size by design. Not thread-/stream-safe (see Usage).

## Algorithm

ConComTorch implements the **Block-based Komura Equivalence (BKE)** algorithm:

> Stefano Allegretti, Federico Bolelli, Michele Cancilla, Costantino Grana.
> "Optimized Block-Based Algorithms to Label Connected Components on GPUs."
> *IEEE Transactions on Parallel and Distributed Systems*, 2019.

BKE operates on **2x2 blocks** rather than individual pixels, reducing memory
accesses and atomic operations. The pipeline is five kernels: Init (block
connectivity via a 16-bit BitSet) -> Compress (path compression) -> Reduction
(union for remaining connections) -> Compress -> FinalLabel (block labels to
pixels). The `bke_ic` variant updates the parent at each traversal step
(InlineCompression) for faster convergence. Intermediate state is packed into
the output tensor to avoid extra allocations. The 2x2 block structure makes
8-connectivity the native, fixed connectivity.

## Limitations and Semantics

- **CUDA-only.** CPU tensors are rejected; there is no CPU fallback.
- **2D single-image only.** No batched `(N, H, W)` API and no 3D volumes;
  iterate in Python for batches.
- **Fixed 8-connectivity.** 4-connectivity is not provided (the 2x2 block
  structure is intrinsically 8-connected).
- **Sparse labels.** Component IDs are positive but root-derived, not dense.
  Use `relabel_components` for a contiguous range.
- **int32 label space.** Pixel-index arithmetic is `int32`; images with more
  than ~2^31 pixels are unsupported.
- **Non-contiguous input is copied.** A contiguous temporary is allocated
  internally, so buffer reuse is allocation-free only for contiguous input.
- **Contiguous, same-device buffers.** A supplied `labels` (or
  `unique_labels`) tensor must be on the same CUDA device as the input.
- **Concurrency.** `ConnectedComponentsLabeler` and any shared reuse buffer are
  single-context: one per device, per stream, per in-flight op.
- **Stream/device.** Launches use the current CUDA stream and a `CUDAGuard`
  bound to the tensor's device; correctness under CUDA graph capture or
  multi-stream pipelines requires the caller to manage stream/event ordering.
- **Determinism.** Final positive label *values* are derived from union-find
  roots and may differ run-to-run, across devices, and between `bke`/`bke_ic`;
  the *partition* into components is stable. Densify with `relabel_components`
  if you need stable IDs.

## Efficiency Tips

- **Reuse a buffer** with `create_labels_buffer` + the `labels=` parameter for
  repeated same-size calls. Remember the return aliases the buffer; `.clone()`
  any result you retain across iterations.
- **Keep inputs contiguous** (`img.contiguous()` once, upstream) so buffer
  reuse actually avoids all allocations.
- **One labeler per stream/device.** For concurrent pipelines, give each
  worker/stream its own `ConnectedComponentsLabeler` or reuse buffer.
- **`collapse_consecutive=True`** (default) is fastest for typical CCL output
  with long equal-label runs; benchmark against `False` if your label field is
  scattered or noisy.
- **`bke_ic`** (default) is generally the better 2D variant.
- **`component_stats` / `get_component_masks`** are single fused kernels; for
  thousands of components the dense `(N, H, W)` mask memory dominates, so
  prefer `component_stats` when you only need area/bbox/centroid.

### Measuring on your workload

Performance depends on image size, component count and size, label-field
contiguity, allocation mode, and GPU architecture. Benchmark with your data and
a warmed device:

```python
import torch, time
from concomtorch import connected_components, create_labels_buffer

img = your_binary_image_cuda
buf = create_labels_buffer(img.shape)
for _ in range(10):                       # warm up
    connected_components(img, labels=buf)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    connected_components(img, labels=buf)
torch.cuda.synchronize()
print((time.perf_counter() - t0) / 100 * 1e3, 'ms/call')
```

Compare against `scipy.ndimage.label`, `cc3d`, or `kornia` on your inputs for a
meaningful baseline.

## Development

```bash
pip install -e './package[dev]'
ruff check package/src/
ruff format package/src/
```

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@article{allegretti2019optimized,
  title={Optimized Block-Based Algorithms to Label Connected Components on GPUs},
  author={Allegretti, Stefano and Bolelli, Federico and Cancilla, Michele and Grana, Costantino},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={31},
  number={2},
  pages={423--438},
  year={2019},
  publisher={IEEE}
}
```

## Contributing

Contributions are welcome. Please open a Pull Request.
