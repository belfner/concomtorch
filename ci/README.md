# concomtorch CI

Self-hosted wheel build automation. Lives on one server, runs once a day, builds every
(torch, cuda, py) combination PyTorch publishes at or above the floors in `matrix.yaml`
that is not yet in the wheelhouse. Wheels carry PEP 440 local versions (`+cu126torch2.6.1`) and are published behind
a two-layer PEP 503 simple index keyed by cuda variant then torch minor
(`<cuda>/<torch_tag>/`, e.g. `cu126/torch2_6/`) that pip consumes via `--extra-index-url`.
The user selects the cuda variant and torch minor matching their installed PyTorch and passes
that two-layer URL; the wheel for that bucket is the only `concomtorch` candidate, so a bare
`pip install concomtorch` resolves it.

## Files

| File | Role |
|---|---|
| `matrix.yaml` | Build floors (`torch_min`, `python_min`, `compute_min`) and docker pool sizing |
| `detect.py` | Query `torch-wheel-index`, intersect with matrix.yaml, emit WANTED set |
| `plan.py` | Diff WANTED against wheelhouse contents, emit build plan grouped by (torch, cuda) |
| `docker_pool.py` | List, build (parallel), and LRU-evict the manylinux+CUDA image cache |
| `build_wheel.py` | Run cibuildwheel against a pre-built image for one (torch, cuda, py-abis) group |
| `publish.py` | Move new wheels into the public serve root and regenerate the two-layer PEP 503 index tree |
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
    index.html                 # human landing page (lists cuda variants)
    cu126/
      index.html               # browsable page (lists torch channels)
      torch2_6/
        index.html             # PEP 503 channel root
        concomtorch/
          index.html           # PEP 503 project page
      torch2_7/
        ...
    cu128/
      ...
    files/*.whl
```

caddy (or nginx) is pointed at `/srv/concomtorch/public/`. Users select the
`<cuda>/<torch_tag>/` directory matching their installed PyTorch:

```
pip install concomtorch --extra-index-url https://wheels.<your-domain>/cu126/torch2_6/
```

The user picks both the cuda variant and the torch minor. That bucket holds only the wheels
built for that (cuda, torch minor), so a bare `pip install concomtorch` resolves the right
one. The `torch==X.Y.*` requirement baked into each wheel still pins the torch minor at
install time.

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

## Publish modes and GitHub auth

`ci/run.py` defaults to `--publish-mode github-pages`. In this mode each tick
uploads the built wheels as GitHub Release assets and pushes the regenerated
PEP 503 index to the `gh-pages` branch; GitHub Pages serves that branch as the
simple index.

### Required: GH_TOKEN via .env

`ci/release.py` sources a personal access token from `/srv/concomtorch/.env`
(the repo-root `.env`, loaded before any `gh` or `git push` invocation). Create
it after `uv sync`:

```bash
sudo -u concomtorch tee /srv/concomtorch/.env >/dev/null <<'EOF'
GH_TOKEN=ghp_your_token_here
EOF
sudo -u concomtorch chmod 600 /srv/concomtorch/.env
```

`.env` is gitignored; keep it owned by the service user with mode `600`. The
token is used both for the `gh release` asset upload and for the `gh-pages`
push (supplied through a transient `GIT_ASKPASS` helper, so it never appears in
argv or logs).

Token scope:

- Classic PAT: `repo` scope (release asset upload plus `gh-pages` branch push).
- Fine-grained PAT: repository `Contents: Read and write`, scoped to this repo.

The repo slug is parsed from `git remote get-url origin`, so the clone must
have an `origin` remote pointing at the GitHub repository. Enable GitHub Pages
for the repository with the source set to the `gh-pages` branch; the published
index is then served at `https://<owner>.github.io/<name>/<cuda>/<torch_tag>/`.

Without a usable token (or pre-configured `gh`/git credentials) the build
phase still succeeds but the publish step fails the tick and notifies.

### Alternative: local publish mode

To self-host the index without GitHub, run with `--publish-mode local`. Wheels
are moved into `<serve-root>/files` and the two-layer PEP 503 index tree is
regenerated there after each build (see the layout above); point caddy/nginx
at `<serve-root>`. No `GH_TOKEN` is needed in this mode. Set it in the systemd
unit, e.g.:

```
ExecStart=/srv/concomtorch/.venv/bin/python /srv/concomtorch/ci/run.py --publish-mode local
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
