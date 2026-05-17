# ConComTorch

GPU-accelerated connected component labeling for 2D PyTorch tensors using the
**Block-based Komura Equivalence (BKE)** algorithm (IEEE TPDS 2019). The package
is a thin Python wrapper over a compiled CUDA extension registered through
PyTorch's operator dispatcher.

**Prebuilt wheels are available** for Python 3.10+, PyTorch 2.6+ (every torch
minor from 2.6 upward), every CUDA variant PyTorch publishes for those torch
versions, and NVIDIA GPUs with compute capability >= 7.5 (Turing or newer). This
matrix expands as new PyTorch and CUDA releases are built and published, so the
exact set of channels is whatever the index currently lists. You install one the
same way you install PyTorch: match your environment, then install from the
matching channel. See [Installation](#installation) for the version-matching
steps and the index URL; browse `https://belfner.github.io/concomtorch/` for the
live list of channels.

## Features

- **BKE algorithm** with two variants: `bke_ic` (InlineCompression, default) and `bke` (standard)
- **8-connectivity** on 2x2 blocks (diagonal neighbors are connected)
- **Buffer reuse** via a caller-supplied output tensor to skip the output allocation
- **uint8 and bool** input tensors
- **Typed**: ships `py.typed`; meta/fake kernels registered for the shape-stable ops

## Installation

The normal path is a prebuilt wheel, the same way you install PyTorch: pick the
build that matches your environment, then install from the matching channel. You
do not build from source for a normal install.

### Requirements

- Python >= 3.10
- PyTorch >= 2.6.0
- CUDA >= 11.8 runtime
- NVIDIA GPU with compute capability >= 7.5 (Turing or newer)

###### GPU architecture coverage

Published wheels carry SASS for the architectures listed per channel plus a PTX fallback for the highest target, so newer GPUs run via JIT PTX compilation.

### Step 1: identify the channel you need

The compiled extension is built against one specific PyTorch CUDA build, so the
channel you install from **must match the PyTorch already in your environment**.
Read the two values off your environment:

```python
import torch
print(torch.__version__)      # e.g. 2.6.1+cu126  -> torch minor 2.6
print(torch.version.cuda)     # e.g. 12.6         -> CUDA variant cu126
```

The index is a two-layer simple index keyed by CUDA variant and torch minor:

```
https://belfner.github.io/concomtorch/<cuda>/<torch_tag>/
                                       cu126  torch2_6
```

### Step 2: install the wheel

The commands below use the `<cuda>/<torch_tag>` placeholder. Replace it with the
channel you identified in step 1 (for example, `cu126/torch2_6`).

#### pip

```bash
pip install concomtorch --index-url https://belfner.github.io/concomtorch/<cuda>/<torch_tag>/
```

`--index-url` restricts resolution to the project index. If the same command
must also reach PyPI for other dependencies, use `--extra-index-url`; be aware
pip will then also consider PyPI for a package named `concomtorch`, so prefer a
separate isolated install step for this package.

#### uv

For a uv-managed project, first add the index and source pin to your
`pyproject.toml`. `explicit = true` scopes the index to only the packages routed
to it, so the rest of your dependency resolution still uses PyPI:

```toml
[[tool.uv.index]]
name = "concomtorch"
url = "https://belfner.github.io/concomtorch/<cuda>/<torch_tag>/"
explicit = true

[tool.uv.sources]
concomtorch = { index = "concomtorch" }
```

Then add the package:

```bash
uv add concomtorch
```

`uv` resolves `concomtorch` from that channel and everything else from PyPI. To
change CUDA/torch, edit the `url` to the matching channel and re-run `uv sync`.

For a non-project (pip-compatible) uv install:

```bash
uv pip install concomtorch --index-url https://belfner.github.io/concomtorch/<cuda>/<torch_tag>/
```

### Install from source

<details>
<summary>Source build and containerized one-off wheel (most users do not need this)</summary>

#### From source

Source builds require a CUDA toolkit (`nvcc`) and a C++17 compiler. The
buildable package lives in `package/`; the repository root is the CI
orchestration environment, not the package, so install the subdirectory:

```bash
git clone <repository-url>
cd concomtorch
pip install -e ./package
```

The build locates `nvcc` at `$CUDA_HOME/bin/nvcc` when `CUDA_HOME` is set,
falling back to `PATH`; finding it compiles the CUDA extension. When the toolkit
is absent the build produces a Python-only package: `import concomtorch` still
succeeds, and the missing-extension error surfaces on the first operator call.

Source builds default to the compute capabilities `>= 7.5` available from the
local toolkit. Override with `CONCOMTORCH_COMPUTE_MIN`, or pass an explicit
semicolon-separated list of dotted compute capabilities via
`TORCH_CUDA_ARCH_LIST`:

```bash
TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0' pip install -e ./package
```

#### Build one wheel yourself in a container

If you want a single wheel for an exact `(torch, cuda, python)` combination
built locally and then installed once, the CI build path produces one. This
needs Docker and is run from the repository root. Warm the manylinux+CUDA image,
build the wheel, then install the artifact:

```bash
python ci/docker_pool.py ensure <cuda>
python ci/build_wheel.py --torch <torch> --cuda <cuda> --py <py> --project-dir package
pip install wheelhouse/concomtorch-*+<cuda>torch<torch>*.whl
```

Fill in the combination you need, for example `--cuda cu126 --torch 2.6.1 --py
cp311`. The wheel is produced in `wheelhouse/` with a PEP 440 local version
(`+<cuda>torch<torch>`, e.g. `+cu126torch2.6.1`) and is verified by an
in-container pytest run with GPU passthrough before it lands.

</details>

## Usage

| Goal | Function |
|------|----------|
| Label an image | `connected_components` |
| Get contiguous `0..N` labels | `relabel_components` |
| Count / list component IDs | `get_unique_labels` |
| Area, bbox, centroid per component | `component_stats` |
| One binary mask per component | `get_component_masks` |
| Skip the per-call output allocation | `create_labels_buffer` + `labels=` |
| Repeated fixed-size calls | `ConnectedComponentsLabeler` |

### Core workflow

Labeling an image and turning the result into something you can index by
component. Most code needs only this section.

#### Label an image

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
# sparse (root-derived), not a dense 1..N range. Densify with relabel_components.
```

#### Densify labels to a contiguous range

`connected_components` produces sparse positive IDs. `relabel_components`
remaps them to a contiguous `0..N` range (0 stays background if present):

```python
from concomtorch import relabel_components

dense = relabel_components(labels)       # pure-torch, stays on GPU
```

### Inspecting components

Operations that consume a label map and report on the components in it.

#### Count and list components

```python
from concomtorch import connected_components, get_unique_labels

labels = connected_components(img)
ids = get_unique_labels(labels)          # sorted int32 IDs, background excluded
num_components = ids.numel()
```

#### Per-component statistics

```python
from concomtorch import component_stats

stats = component_stats(labels)
stats.labels    # int32 (N,)  original sparse IDs, ascending
stats.area      # int64 (N,)  pixel count per component
stats.bbox      # int32 (N,4) [min_row, min_col, max_row, max_col] inclusive
stats.centroid  # float64 (N,2) [row, col]
```

#### Component masks

```python
from concomtorch import get_component_masks

masks = get_component_masks(labels)      # uint8 (N, H, W), values 0/1, one plane per component
```

For thousands of components the dense `(N, H, W)` stack is large; prefer
`component_stats` when you only need area/bbox/centroid.

### Performance and reuse

Opt-in once the basic workflow works and you are calling it repeatedly.

#### Buffer reuse

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

#### Reusable labeler

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

#### Algorithm variants

```python
labels = connected_components(img, algorithm='bke_ic')  # default (inline compression)
labels = connected_components(img, algorithm='bke')      # standard BKE
```

## API Reference

Full per-function signatures, parameters, return types, and raised exceptions
are in **[`REFERENCE.md`](REFERENCE.md)**.

## Algorithm

ConComTorch implements the **Block-based Komura Equivalence (BKE)** algorithm:

> Stefano Allegretti, Federico Bolelli, Michele Cancilla, Costantino Grana.
> "Optimized Block-Based Algorithms to Label Connected Components on GPUs."
> *IEEE Transactions on Parallel and Distributed Systems*, 31(2):423-438, 2019.
> [doi:10.1109/TPDS.2019.2934683](https://doi.org/10.1109/TPDS.2019.2934683)
> ([open-access PDF](https://iris.unimore.it/bitstream/11380/1179616/1/2018_TPDS_Optimized_Block_Based_Algorithms_to_Label_Connected_Components_on_GPUs.pdf))

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
- **Determinism.** For a given input and algorithm the output is reproducible
  run-to-run. Label *values* are union-find-root-derived: sparse, and specific
  to the device and algorithm, so they can differ across devices and between
  `bke`/`bke_ic`. The *partition* into components is stable. Densify with
  `relabel_components` if you need contiguous IDs.

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
- **`bke_ic`** is the default (inline-compression) variant; benchmark it
  against `bke` on your workload if variant choice matters.
- **`component_stats`** runs a GPU densification step then one fused
  accumulation kernel; **`get_component_masks`** materializes the mask stack
  with one fused kernel (preceded by a unique-label reduction when
  `unique_labels` is not supplied), but for thousands of components the dense
  `(N, H, W)` mask memory dominates, so prefer `component_stats` when you only
  need area/bbox/centroid.

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

Development uses [uv](https://docs.astral.sh/uv/). Install the package editable
with its dev extras into the project environment, then run lint and format
through `uv run` so they use that environment's pinned tools:

```bash
uv pip install -e './package[dev]'
uv run ruff check package/src/
uv run ruff format package/src/
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
  publisher={IEEE},
  doi={10.1109/TPDS.2019.2934683}
}
```

## Contributing

Contributions are welcome. Please open a Pull Request.
