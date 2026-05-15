#!/usr/bin/env python3
"""
Orchestrate one tick of the wheel build loop.

Steps:
  1. Detect what PyTorch publishes and matrix.yaml allows.
  2. Diff against the on-disk wheelhouse to produce a plan grouped by (torch, cuda).
  3. Warm up docker images for the cuda variants the plan needs, in parallel.
  4. For each (torch, cuda) group, invoke ci/build_wheel.py serially (GPU box).
  5. Move resulting wheels into the public serve root and regenerate HTML indexes.
  6. Evict least-recently-used images down to max_resident_images.
  7. Notify on failure.

Designed to be invoked by a systemd timer or cron.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from detect import (
    enumerate_wanted,
    fetch_catalog,
    load_matrix,
)
from docker_pool import (
    ensure_images_parallel,
    evict_lru,
)
from notify import notify
from plan import (
    compute_plan,
    group_by_torch_cuda,
    scan_wheelhouse,
)
from publish import (
    collect,
    move_new_wheels,
    write_root_index,
    write_torch_cuda_page,
)

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent


def build_one(torch_version: str, cuda_variant: str, py_abis: list[str], output_dir: Path) -> bool:
    """
    Invoke ci/build_wheel.py for a single (torch, cuda) group. Returns True on success.
    """
    cmd = [
        sys.executable, str(CI_DIR / 'build_wheel.py'),
        '--torch', torch_version,
        '--cuda', cuda_variant,
        '--py', *py_abis,
        '--project-dir', str(REPO_ROOT / 'package'),
        '--output-dir', str(output_dir),
    ]
    print('>>', ' '.join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def publish(output_dir: Path, serve_root: Path) -> None:
    """
    Move any wheels in output_dir into serve_root and regenerate indexes.
    """
    moved = move_new_wheels(output_dir, serve_root / 'files')
    print(f'Moved {len(moved)} wheels into {serve_root / "files"}', flush=True)
    groups = collect(serve_root)
    for (torch_minor, cuda), wheels in groups.items():
        write_torch_cuda_page(serve_root, torch_minor, cuda, wheels)
    write_root_index(serve_root, groups)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=CI_DIR / 'matrix.yaml')
    parser.add_argument('--wheelhouse', type=Path, default=REPO_ROOT / 'wheelhouse')
    parser.add_argument('--serve-root', type=Path, default=REPO_ROOT / 'public')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the plan but do not build.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap the number of (torch, cuda) groups built this tick.')
    parser.add_argument('--skip-image-warmup', action='store_true',
                        help='Assume images exist; skip parallel docker build phase.')
    parser.add_argument('--skip-eviction', action='store_true',
                        help='Skip LRU eviction at the end of the tick.')
    args = parser.parse_args()

    started = time.monotonic()

    matrix = load_matrix(args.matrix)
    docker_cfg = matrix.get('docker', {})
    max_parallel = int(docker_cfg.get('max_parallel_builds', 2))
    max_resident = int(docker_cfg.get('max_resident_images', 3))

    catalog = fetch_catalog()
    wanted = enumerate_wanted(matrix, catalog)
    present = scan_wheelhouse(args.serve_root / 'files') | scan_wheelhouse(args.wheelhouse)
    plan = compute_plan(wanted, present)
    groups = group_by_torch_cuda(plan)

    if args.limit is not None:
        groups = groups[:args.limit]

    if len(groups) == 0:
        print('Nothing to build. Exiting.')
        return 0

    print(f'Plan: {len(groups)} (torch, cuda) groups, {sum(len(g[2]) for g in groups)} wheels total.')
    for torch, cuda, pys in groups:
        print(f'  - {torch} / {cuda} / {" ".join(pys)}')

    if args.dry_run:
        return 0

    active_cuda = sorted({cuda for _, cuda, _ in groups})
    print(f'\nActive cuda variants this tick: {active_cuda}', flush=True)

    image_failures: list[str] = []
    if not args.skip_image_warmup:
        _success, image_failures = ensure_images_parallel(active_cuda, max_parallel=max_parallel)
        if len(image_failures) > 0:
            print(f'Image warmup failed for: {image_failures}. Groups using those variants will be skipped.',
                  flush=True)

    skip_cuda = set(image_failures)
    failures: list[tuple[str, str]] = []
    for torch, cuda, pys in groups:
        if cuda in skip_cuda:
            failures.append((torch, cuda))
            print(f'  - skipping {torch} / {cuda}: image unavailable', flush=True)
            continue
        ok = build_one(torch, cuda, pys, args.wheelhouse)
        if not ok:
            failures.append((torch, cuda))
            print(f'  ! build failed for {torch} / {cuda}', flush=True)
            continue
        publish(args.wheelhouse, args.serve_root)

    if not args.skip_eviction:
        evicted = evict_lru(max_resident, keep=set(active_cuda))
        if len(evicted) > 0:
            print(f'Evicted {len(evicted)} image(s): {evicted}', flush=True)

    elapsed = time.monotonic() - started
    summary = (
        f'concomtorch tick: {len(groups) - len(failures)} groups ok, '
        f'{len(failures)} failed, {elapsed:.0f}s'
    )
    print(summary)

    if len(failures) > 0:
        notify(
            f'{summary}\nFailures:\n' + '\n'.join(f'  {t} / {c}' for t, c in failures),
            title='concomtorch wheel build failed',
            priority='high',
        )
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
