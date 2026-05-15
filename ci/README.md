# concomtorch CI

Self-hosted wheel build automation. Lives on one server, runs once a day, builds any
(torch, cuda, py) combination that PyTorch publishes which is in `matrix.yaml` and not yet in the
wheelhouse. Wheels carry PEP 440 local versions (`+cu121torch2.4`) and are published behind
per-cuda PEP 503 simple indexes (one channel per cuda variant, mirroring PyTorch's
`download.pytorch.org/whl/cu121` layout) that pip consumes via `--extra-index-url`. The torch
minor pin lives in each wheel's `Requires-Dist`, so pip auto-selects the right wheel for the
user's installed torch.

## Files

| File | Role |
|---|---|
| `matrix.yaml` | Source of truth: cuda variants, py ABIs, torch_min, exclusions, docker pool sizing |
| `detect.py` | Query `torch-wheel-index`, intersect with matrix.yaml, emit WANTED set |
| `plan.py` | Diff WANTED against wheelhouse contents, emit build plan grouped by (torch, cuda) |
| `docker_pool.py` | List, build (parallel), and LRU-evict the manylinux+CUDA image cache |
| `build_wheel.py` | Run cibuildwheel against a pre-built image for one (torch, cuda, py-abis) group |
| `publish.py` | Move new wheels into the public serve root and regenerate per-cuda PEP 503 indexes |
| `notify.py` | POST to ntfy.sh (or any compatible endpoint) on failure |
| `run.py` | Orchestrator: detect -> plan -> warm images -> build -> publish -> evict -> notify |
| `systemd/concomtorch-wheels.service` | Oneshot service that runs `run.py` |
| `systemd/concomtorch-wheels.timer` | Daily timer triggering the service |

## Layout on the server

```
/srv/concomtorch/              # git clone of this repo
  .venv/                       # uv sync (outer env: pyyaml, torch-wheel-index, cibuildwheel)
  pyproject.toml               # outer CI env (sets [tool.uv] package = false)
  ci/                          # orchestration scripts (this directory)
  package/                     # the buildable wheel (own pyproject + setup.py)
    pyproject.toml             # [project] dynamic = ["version","dependencies"]
    setup.py                   # composes PEP 440 local version from env vars
    csrc/
    src/concomtorch/
  wheelhouse/                  # cibuildwheel output staging
  public/                      # web-served root
    index.html                 # human landing page
    cu121/
      index.html               # PEP 503 channel root
      concomtorch/
        index.html             # PEP 503 project page
    cu124/
      ...
    files/*.whl
```

caddy (or nginx) is pointed at `/srv/concomtorch/public/`. Users select the channel matching
their installed torch CUDA build:

```
pip install concomtorch --extra-index-url https://wheels.<your-domain>/cu121/
```

The user picks the cuda variant; pip picks the torch minor automatically via the
`torch==X.Y.*` requirement baked into each wheel at build time.

## Bootstrap

```bash
# On the server:
sudo useradd -r -m -d /srv/concomtorch -s /bin/bash concomtorch
sudo usermod -aG docker concomtorch
sudo -u concomtorch git clone https://github.com/<you>/concomtorch /srv/concomtorch
cd /srv/concomtorch
sudo -u concomtorch uv sync

sudo cp ci/systemd/concomtorch-wheels.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now concomtorch-wheels.timer

# Optional: failure notifications via ntfy.sh
sudo systemctl edit concomtorch-wheels.service
# Add:
#   [Service]
#   Environment=CONCOMTORCH_NTFY_URL=https://ntfy.sh/<your-topic>
```

## Manual operations

```bash
# Dry-run plan
python ci/run.py --dry-run

# Cap builds this tick (useful for the first big backfill)
python ci/run.py --limit 2

# Detect only
python ci/detect.py --format text

# Plan only
python ci/plan.py

# Inspect the image pool
python ci/docker_pool.py list

# Pre-warm images for a set of cuda variants (parallel)
python ci/docker_pool.py ensure --max-parallel 2 cu121 cu124 cu128

# Trim the pool to N images, never evicting the cuda variants in --keep
python ci/docker_pool.py evict --max-resident 3 --keep cu129 cu130
```

## Image pool

`matrix.yaml.docker.max_parallel_builds` caps how many `docker build` invocations run
simultaneously during the warmup phase of a tick. `matrix.yaml.docker.max_resident_images`
caps how many manylinux+CUDA images live on disk at the end of a tick. Images for cuda
variants in the active plan are never evicted; among the rest, the oldest is removed first.
Each image is roughly 5-15 GB.

## Why this shape

- One server. No CI server. Releases are infrequent and the state (wheel exists / doesn't) lives
  on disk where the wheels live. systemd handles scheduling, journald handles logs.
- `build_wheel.py` is the per-group build engine: one cibuildwheel invocation per (torch, cuda, py-abis) tuple, against a pre-built manylinux+CUDA image.
- `torch-wheel-index` is the detection layer.
- PyG-style flat HTML index is the simplest self-hostable, pip-friendly publish format.
