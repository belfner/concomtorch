# concomtorch CI

Self-hosted wheel build automation. Lives on one server, runs once a day, builds any
(torch, cuda, py) combination that PyTorch publishes which is in `matrix.yaml` and not yet in the
wheelhouse. Wheels are served as PEP 440 local versions (`+cu121torch2.4`) from a static HTML
index that pip consumes via `--extra-index-url` or `--find-links`.

## Files

| File | Role |
|---|---|
| `matrix.yaml` | Source of truth: cuda variants, py ABIs, torch_min, exclusions |
| `detect.py` | Query `torch-wheel-index`, intersect with matrix.yaml, emit WANTED set |
| `plan.py` | Diff WANTED against wheelhouse contents, emit build plan grouped by (torch, cuda) |
| `build_wheel.py` | Build manylinux+CUDA docker image and run cibuildwheel for one group |
| `publish.py` | Move new wheels into the public serve root and regenerate HTML indexes |
| `notify.py` | POST to ntfy.sh (or any compatible endpoint) on failure |
| `run.py` | Orchestrator: detect to plan to build to publish to notify |
| `systemd/concomtorch-wheels.service` | Oneshot service that runs `run.py` |
| `systemd/concomtorch-wheels.timer` | Daily timer triggering the service |

## Layout on the server

```
/srv/concomtorch/              # git clone of this repo
  .venv/                       # uv sync --group build
  ci/
  wheelhouse/                  # cibuildwheel output staging
  public/                      # web-served root
    index.html
    torch-2.4+cu121.html
    torch-2.4+cu124.html
    ...
    files/*.whl
```

caddy (or nginx) is pointed at `/srv/concomtorch/public/`. Users install with:

```
pip install --extra-index-url https://wheels.<your-domain>/ concomtorch
# or pin explicitly:
pip install concomtorch --find-links https://wheels.<your-domain>/torch-2.4+cu121.html
```

## Bootstrap

```bash
# On the server:
sudo useradd -r -m -d /srv/concomtorch -s /bin/bash concomtorch
sudo usermod -aG docker concomtorch
sudo -u concomtorch git clone https://github.com/<you>/concomtorch /srv/concomtorch
cd /srv/concomtorch
sudo -u concomtorch uv sync --group build

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
```

## Why this shape

- One server. No CI server. Releases are infrequent and the state (wheel exists / doesn't) lives
  on disk where the wheels live. systemd handles scheduling, journald handles logs.
- `gen_docker.py` (now `build_wheel.py`) is the build engine, unchanged in spirit.
- `torch-wheel-index` is the detection layer.
- PyG-style flat HTML index is the simplest self-hostable, pip-friendly publish format.
