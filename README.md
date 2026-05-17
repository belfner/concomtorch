# ConComTorch

**GPU-accelerated connected component labeling for 2D PyTorch tensors.** Label
the connected components of a binary CUDA tensor in a single fast kernel
pipeline, with the result handed straight back as a PyTorch tensor.

ConComTorch implements the **Block-based Komura Equivalence (BKE)** algorithm
([Allegretti et al., IEEE TPDS 2019](https://doi.org/10.1109/TPDS.2019.2934683))
as a CUDA extension registered through PyTorch's operator dispatcher. It is a
thin Python wrapper over compiled CUDA, so it stays on the GPU and out of your
way.

## Highlights

- **Fast.** BKE works on 2x2 blocks, cutting memory traffic and atomics; two
  variants (`bke_ic` inline-compression default, and standard `bke`).
- **8-connectivity** on 2x2 blocks (diagonal neighbors connected).
- **Zero-copy friendly.** Reuse a caller-supplied output tensor to skip the
  per-call allocation.
- **uint8 and bool** inputs; `int32` labels out, background `0`.
- **Typed**, ships `py.typed`, with meta/fake kernels for the shape-stable ops.

## Quick start

Prebuilt wheels are published the same way PyTorch ships: install the build that
matches your environment. Replace `<cuda>/<torch_tag>` with your channel (for
example `cu126/torch2_6`); see the installation guide below for how to find it:

```bash
pip install concomtorch --index-url https://belfner.github.io/concomtorch/<cuda>/<torch_tag>/
```

```python
import torch
from concomtorch import connected_components

img = torch.tensor(
    [[1, 1, 0, 0],
     [0, 0, 0, 1],
     [0, 1, 1, 1]],
    dtype=torch.uint8,
    device="cuda",
)

labels = connected_components(img)  # int32 (H, W), background 0
```

The channel must match your installed PyTorch CUDA build. The
**[full installation guide](package/README.md#installation)** walks through
identifying your channel, the pip and uv (`pyproject.toml`) setups, source
builds, and the live list of available CUDA x torch channels. Prebuilt wheels
cover Python 3.10+, PyTorch 2.6+, every CUDA variant PyTorch publishes for those
versions, and GPUs with compute capability >= 7.5; the matrix expands as new
PyTorch and CUDA releases are built.

## Documentation

- **Using the package** (install, full API, algorithm, semantics, limitations):
  **[`package/README.md`](package/README.md)**.
- **Operating the wheel-build/publish service** (bootstrap, GitHub auth, config,
  building a single wheel by hand): [`ci/README.md`](ci/README.md).

## Repository layout

Two distinct Python projects live in this repository:

- **`package/`**: the buildable wheel (`concomtorch`). Its own `pyproject.toml`
  and `setup.py`; CUDA extension source in `package/csrc/`, Python package in
  `package/src/concomtorch/`.
- **repo root**: the CI orchestration environment (`[tool.uv] package = false`).
  The `ci/*.py` scripts run a self-hosted wheel-build tick (detect -> plan ->
  build -> publish -> evict -> notify) over the CUDA x PyTorch x Python matrix.
  The default publish mode is GitHub Pages plus GitHub Releases and requires a
  `GH_TOKEN` in a repo-root `.env`; see the "Publish modes and GitHub auth"
  section of [`ci/README.md`](ci/README.md).

## License

MIT. See [LICENSE](LICENSE).
