# concomtorch

GPU-accelerated connected component labeling for PyTorch tensors using the
Block-based Komura Equivalence (BKE) algorithm. CUDA-only PyTorch C++
extension.

## Repository layout

This repository contains two distinct Python projects:

- **`package/`**: the buildable wheel (`concomtorch`). Has its own
  `pyproject.toml` and `setup.py`; the CUDA extension source is in
  `package/csrc/` and the Python package in `package/src/concomtorch/`.
  Install from source with `pip install -e ./package`.
- **repo root**: the CI orchestration environment (`concomtorch-ops`,
  `[tool.uv] package = false`). The `ci/*.py` scripts run a daily self-hosted
  wheel-build tick (detect -> plan -> build -> publish -> evict -> notify) over
  the CUDA x PyTorch x Python build matrix.

## Where to go next

- **Using the package** (install, API, algorithm): see
  [`package/README.md`](package/README.md).
- **Operating the wheel-build/publish service** (bootstrap, GitHub auth,
  config, building a single wheel by hand): see [`ci/README.md`](ci/README.md).

The default publish mode is GitHub Pages plus GitHub Releases, which requires a
`GH_TOKEN` in a repo-root `.env`; see the "Publish modes and GitHub auth"
section of [`ci/README.md`](ci/README.md).

## License

MIT. See [LICENSE](LICENSE).
