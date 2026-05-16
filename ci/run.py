#!/usr/bin/env python3
"""
Orchestrate one tick of the wheel build loop.

Steps:
  1. Detect what PyTorch publishes and matrix.yaml allows.
  2. Diff against the on-disk wheelhouse to produce a plan grouped by (torch, cuda).
  3. Warm up docker images for the cuda variants the plan needs, in parallel.
  4. For each (torch, cuda) group, invoke ci/build_wheel.py serially (GPU box).
  5. Publish. In local mode wheels move into the serve root after each build. In
     github-pages mode the release/index-sync runs whenever the wheelhouse holds
     wheels, even on a tick that built nothing, so a release or gh-pages push that
     failed on a prior tick is retried and a drifted index re-synced rather than
     wedged forever behind an empty plan. ci/release.py is idempotent, so the
     unconditional retry is a no-op when the index is already in sync.
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

from packaging.version import Version

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
    render_index_tree,
)

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent


def render_plan_table(groups: list[tuple[str, str, list[str]]]) -> str:
    """
    Render the build plan as an aligned text table sorted by torch version then cuda.

    Parameters
    ----------
    groups : list of tuple
        (torch_version, cuda_variant, [py_abi, ...]) tuples from group_by_torch_cuda.

    Returns
    -------
    str
        The formatted table including header, rows, and a totals footer.
    """
    rows = sorted(groups, key=lambda g: (Version(g[0]), g[1]))
    headers = ('Torch', 'CUDA', 'Python ABIs', 'Wheels')
    cells = [
        (torch, cuda, ' '.join(pys), str(len(pys)))
        for torch, cuda, pys in rows
    ]
    widths = [
        max(len(headers[i]), *(len(c[i]) for c in cells)) if len(cells) > 0 else len(headers[i])
        for i in range(4)
    ]

    def fmt(c: tuple[str, str, str, str]) -> str:
        return '  '.join([
            c[0].ljust(widths[0]),
            c[1].ljust(widths[1]),
            c[2].ljust(widths[2]),
            c[3].rjust(widths[3]),
        ])

    sep = '  '.join('-' * w for w in widths)
    total_wheels = sum(len(pys) for _, _, pys in rows)
    lines = [fmt(headers), sep]
    lines += [fmt(c) for c in cells]
    lines += [sep, f'Total: {len(rows)} (torch, cuda) groups, {total_wheels} wheels']
    return '\n'.join(lines)


def build_one(
    torch_version: str,
    cuda_variant: str,
    py_abis: list[str],
    output_dir: Path,
    compute_min: str,
) -> bool:
    """
    Invoke ci/build_wheel.py for a single (torch, cuda) group. Returns True on success.
    """
    cmd = [
        sys.executable, str(CI_DIR / 'build_wheel.py'),
        '--torch', torch_version,
        '--cuda', cuda_variant,
        '--py', *py_abis,
        '--compute-min', compute_min,
        '--project-dir', str(REPO_ROOT / 'package'),
        '--output-dir', str(output_dir),
    ]
    print('>>', ' '.join(cmd), flush=True)
    result = subprocess.run(cmd)
    return result.returncode == 0


def publish(output_dir: Path, serve_root: Path) -> None:
    """
    Move any wheels in output_dir into serve_root and regenerate the two-layer PEP 503 tree.
    """
    moved = move_new_wheels(output_dir, serve_root / 'files')
    print(f'Moved {len(moved)} wheels into {serve_root / "files"}', flush=True)
    groups = collect(serve_root)
    render_index_tree(serve_root, groups)


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
    parser.add_argument('--publish-mode', choices=['local', 'github-pages'], default='github-pages',
                        help='local: move wheels to <serve-root>/files and regenerate HTML there '
                             'after each build. github-pages: leave wheels in the wheelhouse and '
                             'run ci/release.py once at the end of the tick to upload to GitHub '
                             'Releases and push the index to the gh-pages branch.')
    parser.add_argument('--release-tag-prefix', default='wheels',
                        help='Prefix for per-(cuda, torch_minor) GitHub Release tags used by '
                             "publish-mode=github-pages. Each combination gets a release named "
                             "'<prefix>-<cuda>-torch<minor>'.")
    args = parser.parse_args()

    started = time.monotonic()

    matrix = load_matrix(args.matrix)
    docker_cfg = matrix.get('docker', {})
    max_parallel = int(docker_cfg.get('max_parallel_builds', 2))
    max_resident = int(docker_cfg.get('max_resident_images', 3))
    compute_min = str(matrix['compute_min'])

    catalog = fetch_catalog()
    wanted = enumerate_wanted(matrix, catalog)
    # Scan both publish destinations: local mode moves wheels into serve-root/files while
    # github-pages mode leaves them in the wheelhouse. Taking the union keeps a tick that
    # follows a publish-mode switch from rescheduling wheels that already exist in the
    # other location.
    present = scan_wheelhouse(args.serve_root / 'files') | scan_wheelhouse(args.wheelhouse)
    plan = compute_plan(wanted, present)
    groups = group_by_torch_cuda(plan)

    if args.limit is not None:
        groups = groups[:args.limit]

    have_builds = len(groups) > 0
    publishes_independently = args.publish_mode == 'github-pages'

    if have_builds:
        print(render_plan_table(groups))
    else:
        print('Nothing to build.')
        if not publishes_independently:
            # local mode republishes inside the build loop, so there is nothing to
            # retry across ticks when the plan is empty.
            return 0

    if args.dry_run:
        return 0

    failures: list[tuple[str, str]] = []
    active_cuda: list[str] = []

    if have_builds:
        active_cuda = sorted({cuda for _, cuda, _ in groups})
        print(f'\nActive cuda variants this tick: {active_cuda}', flush=True)

        image_failures: list[str] = []
        if not args.skip_image_warmup:
            _success, image_failures = ensure_images_parallel(active_cuda, max_parallel=max_parallel)
            if len(image_failures) > 0:
                print(f'Image warmup failed for: {image_failures}. Groups using those variants will be skipped.',
                      flush=True)

        skip_cuda = set(image_failures)
        for torch, cuda, pys in groups:
            if cuda in skip_cuda:
                failures.append((torch, cuda))
                print(f'  - skipping {torch} / {cuda}: image unavailable', flush=True)
                continue
            ok = build_one(torch, cuda, pys, args.wheelhouse, compute_min)
            if not ok:
                failures.append((torch, cuda))
                print(f'  ! build failed for {torch} / {cuda}', flush=True)
                continue
            if args.publish_mode == 'local':
                publish(args.wheelhouse, args.serve_root)

    publish_failed = False
    if publishes_independently:
        # Run release/index-sync whenever the wheelhouse holds any wheel file,
        # independent of whether this tick built anything. This retries a prior failed
        # release or gh-pages push and re-syncs a drifted index; ci/release.py is
        # idempotent so an already-in-sync index is a no-op. The gate matches raw
        # '*.whl' rather than parseable signatures so a build that emitted only
        # malformed wheel filenames still reaches ci/release.py, whose unparsable-
        # filename validation fails the tick loudly instead of silently skipping.
        if args.wheelhouse.is_dir() and any(args.wheelhouse.glob('*.whl')):
            release_cmd = [
                sys.executable, str(CI_DIR / 'release.py'),
                '--wheelhouse', str(args.wheelhouse),
                '--tag-prefix', args.release_tag_prefix,
            ]
            print('>>', ' '.join(release_cmd), flush=True)
            release_result = subprocess.run(release_cmd)
            if release_result.returncode != 0:
                publish_failed = True
                print('  ! release step failed', flush=True)
        else:
            print('github-pages mode: wheelhouse empty, nothing to publish.', flush=True)

    if have_builds and not args.skip_eviction:
        evicted = evict_lru(max_resident, keep=set(active_cuda))
        if len(evicted) > 0:
            print(f'Evicted {len(evicted)} image(s): {evicted}', flush=True)

    elapsed = time.monotonic() - started
    summary = (
        f'concomtorch tick: {len(groups) - len(failures)} groups ok, '
        f'{len(failures)} build failed, '
        f'publish {"failed" if publish_failed else "ok"}, {elapsed:.0f}s'
    )
    print(summary)

    if len(failures) > 0 or publish_failed:
        lines = [f'  {t} / {c}' for t, c in failures]
        if publish_failed:
            lines.append('  release / gh-pages')
        notify(
            f'{summary}\nFailures:\n' + '\n'.join(lines),
            title='concomtorch wheel build failed',
            priority='high',
        )
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
