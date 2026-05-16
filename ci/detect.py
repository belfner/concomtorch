#!/usr/bin/env python3
"""
Enumerate (torch_version, cuda_variant, py_abi) combinations that PyTorch publishes and the
local matrix.yaml allows.

Uses the torch-wheel-index package to query download.pytorch.org. The output is the WANTED set;
ci/plan.py derives the build plan from it by subtracting wheels already present in the wheelhouse.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import yaml
from packaging.version import Version


@dataclass(frozen=True)
class Combo:
    torch: str
    cuda: str
    py: str

    def key(self) -> tuple[str, str, str]:
        return (self.torch, self.cuda, self.py)


def dotted_to_cu_tag(dotted: str) -> str:
    """
    Translate a dotted CUDA version to its PyTorch ``cuXYZ`` tag.

    PyTorch's tag scheme encodes the CUDA minor as a single trailing digit
    (``12.1`` -> ``cu121``, ``11.8`` -> ``cu118``). A minor of two or more
    digits has no representation in that scheme, so it is rejected here, at
    catalog-enumeration time, rather than silently producing a tag that the
    downstream ``cu<major><minor>`` parsers would decode back to the wrong
    version.

    Parameters
    ----------
    dotted : str
        Dotted CUDA version, e.g. ``'12.1'``.

    Returns
    -------
    str
        The ``cuXYZ`` tag, e.g. ``'cu121'``.

    Raises
    ------
    ValueError
        When the minor component is not exactly one digit.
    """
    parts = dotted.split('.')
    if len(parts) != 2 or len(parts[1]) != 1:
        raise ValueError(
            f"CUDA version {dotted!r} does not fit PyTorch's cu<major><minor> "
            'tag scheme (minor must be a single digit)'
        )
    return 'cu' + dotted.replace('.', '')


def dotted_to_cp(dotted: str) -> str:
    """
    Translate '3.10' to 'cp310'.
    """
    return 'cp' + dotted.replace('.', '')


def load_matrix(path: Path) -> dict:
    """
    Load and validate the build matrix YAML.

    Parameters
    ----------
    path : Path
        Path to matrix.yaml.

    Returns
    -------
    dict
        Parsed matrix configuration.
    """
    with path.open() as fh:
        data = yaml.safe_load(fh)
    required = {'torch_min', 'python_min', 'compute_min'}
    missing = required - set(data.keys())
    if len(missing) > 0:
        raise ValueError(f'matrix.yaml missing required keys: {sorted(missing)}')
    return data


def fetch_catalog() -> list[dict]:
    """
    Run `torch-wheel-index find --compute-type cuda --platform linux` and parse the catalog.

    Returns
    -------
    list[dict]
        Catalog rows.
    """
    result = subprocess.run(
        ['torch-wheel-index', 'find', '--format', 'json',
         '--compute-type', 'cuda', '--platform', 'linux'],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def enumerate_wanted(matrix: dict, catalog: list[dict]) -> list[Combo]:
    """
    Select every catalog row at or above the torch and python floors.

    For each torch version >= ``torch_min``, every cuda variant and every python ABI
    >= ``python_min`` that PyTorch publishes is included.

    Parameters
    ----------
    matrix : dict
        Parsed matrix.yaml.
    catalog : list[dict]
        Output of `torch-wheel-index find`.

    Returns
    -------
    list[Combo]
        Sorted list of valid combinations.
    """
    torch_min = Version(matrix['torch_min'])
    python_min = Version(matrix['python_min'])

    seen: set[tuple[str, str, str]] = set()
    out: list[Combo] = []
    for row in catalog:
        torch_ver = row['version']
        cuda_dotted = row['compute_version']
        py_dotted = row['python_version']

        cuda = dotted_to_cu_tag(cuda_dotted)
        py = dotted_to_cp(py_dotted)

        if Version(torch_ver) < torch_min:
            continue
        if Version(py_dotted) < python_min:
            continue
        combo = Combo(torch=torch_ver, cuda=cuda, py=py)
        if combo.key() in seen:
            continue
        seen.add(combo.key())
        out.append(combo)

    out.sort(key=lambda c: (Version(c.torch), c.cuda, c.py))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path,
                        default=Path(__file__).parent / 'matrix.yaml')
    parser.add_argument('--format', choices=['json', 'text'], default='json')
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    catalog = fetch_catalog()
    wanted = enumerate_wanted(matrix, catalog)

    if args.format == 'json':
        json.dump([asdict(c) for c in wanted], sys.stdout, indent=2)
        sys.stdout.write('\n')
    else:
        for c in wanted:
            print(f'{c.torch}\t{c.cuda}\t{c.py}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
