#!/usr/bin/env python3
"""
Diff the wanted set (from detect.py) against the on-disk wheelhouse to produce the build plan.

The wheelhouse is the source of truth for state. For each (torch_minor, cuda, py) triple,
``select_latest_patch_per_minor`` narrows the wanted set to the highest available patch (so a
2.6.0 wheelhouse with 2.6.1 newly upstream triggers a 2.6.1 build), and ``compute_plan``
schedules a build whenever a wheel for that exact full patch is missing. The wheel's PEP 440
local version encodes the full patch (``+cu124torch2.6.1``) so the wheelhouse signature can
distinguish patches; the wheel's Requires-Dist pins ``torch==X.Y.*`` so a single wheel still
satisfies any patch of its minor on the install side.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

from packaging.version import Version

from detect import (
    Combo,
    enumerate_wanted,
    fetch_catalog,
    load_matrix,
)

WHEEL_RE = re.compile(
    r'^(?P<name>[^-]+)-(?P<version>[^+]+)\+(?P<local>[^-]+)-(?P<py>cp\d+)-(?P=py)-(?P<plat>.+)\.whl$'
)
LOCAL_RE = re.compile(r'^(?P<cuda>cu\d+)torch(?P<torch>\d+\.\d+\.\d+)$')


@dataclass(frozen=True)
class WheelKey:
    torch: str
    cuda: str
    py: str


def parse_wheel(filename: str) -> WheelKey | None:
    """
    Extract the (torch, cuda, py) signature from a wheel filename.

    Recognizes only wheels whose local version follows '+cu{N}torch{X.Y.Z}'.

    Parameters
    ----------
    filename : str
        Wheel filename basename.

    Returns
    -------
    WheelKey or None
        Parsed key, or None if the filename does not match.
    """
    m = WHEEL_RE.match(filename)
    if m is None:
        return None
    lm = LOCAL_RE.match(m.group('local'))
    if lm is None:
        return None
    return WheelKey(torch=lm.group('torch'), cuda=lm.group('cuda'), py=m.group('py'))


def scan_wheelhouse(path: Path) -> set[WheelKey]:
    """
    Return the set of (torch, cuda, py) signatures already built.
    """
    if not path.is_dir():
        return set()
    keys: set[WheelKey] = set()
    for wheel in path.glob('*.whl'):
        key = parse_wheel(wheel.name)
        if key is not None:
            keys.add(key)
    return keys


def torch_minor(version: str) -> str:
    """
    Reduce '2.4.1' to '2.4'.
    """
    parts = version.split('.')
    return f'{parts[0]}.{parts[1]}'


def select_latest_patch_per_minor(wanted: list[Combo]) -> list[Combo]:
    """
    Keep only the highest torch patch per (torch_minor, cuda, py) triple.

    Upstream publishes every patch of every minor; we only ever build the latest patch of
    each minor. When a new patch (e.g. ``2.6.1`` after ``2.6.0``) appears upstream this
    function selects it for the next plan diff, and ``compute_plan`` then schedules a build
    because the new full-patch signature is not yet present in the wheelhouse.

    Parameters
    ----------
    wanted : list[Combo]
        Output of :func:`detect.enumerate_wanted`.

    Returns
    -------
    list[Combo]
        One combo per (minor, cuda, py), sorted by (Version(torch), cuda, py).
    """
    best: dict[tuple[str, str, str], Combo] = {}
    for combo in wanted:
        key = (torch_minor(combo.torch), combo.cuda, combo.py)
        prev = best.get(key)
        if prev is None or Version(combo.torch) > Version(prev.torch):
            best[key] = combo
    return sorted(best.values(), key=lambda c: (Version(c.torch), c.cuda, c.py))


def compute_plan(wanted: list[Combo], present: set[WheelKey]) -> list[Combo]:
    """
    Filter wanted combos to those whose exact (torch, cuda, py) signature is missing.

    Wanted is narrowed to one combo per minor (highest patch) before diffing so the plan
    never tries to backfill superseded patches.
    """
    latest = select_latest_patch_per_minor(wanted)
    out = []
    for combo in latest:
        key = WheelKey(torch=combo.torch, cuda=combo.cuda, py=combo.py)
        if key not in present:
            out.append(combo)
    return out


def group_by_torch_cuda(plan: list[Combo]) -> list[tuple[str, str, list[str]]]:
    """
    Group plan rows into (torch_version, cuda_variant, [py_abi, ...]) tuples so one container build
    handles all requested Python ABIs in a single cibuildwheel invocation.

    Returns
    -------
    list of tuple
        Sorted by torch version then cuda.
    """
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in plan:
        groups[(c.torch, c.cuda)].append(c.py)
    out = [(torch, cuda, sorted(pys)) for (torch, cuda), pys in groups.items()]
    out.sort(key=lambda g: (Version(g[0]), g[1]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path,
                        default=Path(__file__).parent / 'matrix.yaml')
    parser.add_argument('--wheelhouse', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'wheelhouse')
    parser.add_argument('--format', choices=['json', 'text', 'grouped'], default='grouped')
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    catalog = fetch_catalog()
    wanted = enumerate_wanted(matrix, catalog)
    present = scan_wheelhouse(args.wheelhouse)
    plan = compute_plan(wanted, present)

    if args.format == 'json':
        json.dump([asdict(c) for c in plan], sys.stdout, indent=2)
        sys.stdout.write('\n')
    elif args.format == 'text':
        for c in plan:
            print(f'{c.torch}\t{c.cuda}\t{c.py}')
    else:
        for torch, cuda, pys in group_by_torch_cuda(plan):
            print(f'{torch}\t{cuda}\t{" ".join(pys)}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
